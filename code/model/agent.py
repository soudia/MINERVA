import ops
import random
import itertools
import collections
import numpy as np
import tensorflow as tf
from dqna_cell import DQNALSTMCell
from collections import defaultdict


class Agent(object):

    def __init__(self, params):

        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']

        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)

        if params['use_entity_embeddings']:
            self.entity_initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.entity_initializer = tf.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        self.attention_size = self.hidden_size
        self.belief_inp_dim = self.m * self.embedding_size
        self.belief_out_dim = 2 * self.hidden_size

        self.action_inp_dim = 1 * self.m * self.embedding_size + self.belief_out_dim
        self.action_out_dim = 4 * self.m * self.hidden_size

        with tf.variable_scope("action_lookup_table"):
            self.action_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            self.relation_lookup_table = tf.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.contrib.layers.xavier_initializer(),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.variable_scope("entity_lookup_table"):
            self.entity_embedding_placeholder = tf.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        with tf.variable_scope("DQNA"):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(
                      DQNALSTMCell(self.m * self.hidden_size, self.action_out_dim, self.belief_out_dim,
                                   use_peepholes=True, state_is_tuple=True))
            self.reasoning_step = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.belief_MLP = ops.initBeliefEstimator(self.batch_size, self.belief_inp_dim, self.belief_out_dim,
                                                  [100, 150], 2 * self.hidden_size, self.attention_size)


    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)


    def policy(self, state, observation, belief, relation_embeddings, next_relations, range_arr):
        dim_input = 2 * self.embedding_size
        w_omega = tf.Variable(tf.random_normal([dim_input, dim_input], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([dim_input], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([dim_input], stddev=0.1))

        def attend(inputs):
            with tf.name_scope('v'):
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape
            output = inputs * tf.expand_dims(alphas, -1)
            return alphas, output

        scores, _ = attend(relation_embeddings) # use alphas as scores
        _, _, chosen = self.random_categorical(range_arr, scores, next_relations)
        chosen = tf.nn.embedding_lookup(self.relation_lookup_table, chosen)
        cosine_sim = tf.keras.losses.CosineSimilarity(axis=1)
        alpha = 1. - cosine_sim(belief, chosen)
        beta = 1. - cosine_sim(observation, chosen)
        with tf.variable_scope("policy", reuse=tf.AUTO_REUSE):
            inputs = tf.concat([state, alpha * belief, beta * observation], -1)
            hidden = tf.layers.Dense(2 * self.action_inp_dim, activation=tf.nn.relu)(inputs)
            output = tf.layers.Dense(self.action_out_dim, activation=tf.nn.relu)(hidden)
        return output

    def forward(self, state, scope="feed_forward"):
        with tf.variable_scope(scope):
            hidden = tf.layers.dense(state, 4 * self.hidden_size, activation=tf.nn.relu)
            output = tf.layers.dense(hidden, self.m * self.embedding_size, activation=tf.nn.relu)
        return output


    def action_encoder(self, next_relations, next_entities):
        with tf.variable_scope("lookup_table_edge_encoder"):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding


    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
             amplitude, belief, label_action, range_arr, first_step_of_test):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)

        prev_state = tuple([prev_state[i].set_action(amplitude) for i in range(self.LSTM_Layers)])

        output, new_state = self.reasoning_step(prev_action_embedding, prev_state)  # output: [B, 2D]
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)

        # Get state vector
        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        if self.use_entity_embeddings:
            state = tf.concat([output, prev_entity], axis=-1)
        else:
            state = output
        state_query_concat = tf.concat([state, query_embedding], axis=-1)
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        amplitude = self.policy(state_query_concat, prev_action_embedding, belief,
                                candidate_action_embeddings, next_relations, range_arr) # [B, 4D]

        output = self.forward(state_query_concat)
        output_expanded = tf.expand_dims(output, axis=1)  # [B, 1, 2D]
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        mask = tf.equal(next_relations, comparison_tensor)  # The mask
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = tf.where(mask, dummy_scores, prelim_scores)  # [B, MAX_NUM_ACTIONS]

        loss, action_idx, chosen_relation = self.random_categorical(range_arr, scores, next_relations)

        return loss, output, new_state, tf.nn.log_softmax(scores), action_idx, chosen_relation, amplitude


    def random_categorical(self, range_arr, scores, next_relations, num_samples=1):
        action = tf.cast(tf.random.categorical(logits=scores, num_samples=num_samples), tf.int32)  # [B, 1]
        label_action =  tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]
        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(tf.stack([range_arr, action_idx])))

        return loss, action_idx, chosen_relation


    def init_belief(self):
        return tf.zeros((self.batch_size, 2 * self.hidden_size))

    def init_action(self):
        return tf.ones((self.batch_size, self.action_out_dim))


    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities,
                 path_label, query_relation, range_arr, first_step_of_test, T=3, entity_sequence=0):

        self.baseline_inputs = []
        # get the query vector
        query_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)  # [B, 2D]
        state = self.reasoning_step.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        prev_relation = self.dummy_start_label

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []
        all_hiddens = []
        all_beliefs = []

        with tf.variable_scope("policy_steps_unroll") as scope:
            for t in range(T):
                if t == 0:
                    belief = self.init_belief()
                    amplitude = self.init_action()
                else:
                    scope.reuse_variables()
                    hiddens = tf.concat(all_hiddens, axis=-1)
                    beliefs = tf.concat(all_beliefs, axis=-1)
                    _, h = tf.split(state[0], num_or_size_splits=2, axis=0)
                    belief = self.belief_MLP(tf.squeeze(h, 0), hiddens, beliefs) # belief: [B, 2D]
                next_possible_relations = candidate_relation_sequence[t]  # [B, MAX_NUM_ACTIONS, MAX_EDGE_LENGTH]
                next_possible_entities = candidate_entity_sequence[t]
                current_entities_t = current_entities[t]

                path_label_t = path_label[t]  # [B]

                loss, output, state, logits, idx, chosen_relation, amplitude \
                                                        = self.step(next_possible_relations,
                                                                    next_possible_entities,
                                                                    state, prev_relation, query_embedding,
                                                                    current_entities_t, amplitude, belief,
                                                                    label_action=path_label_t,
                                                                    range_arr=range_arr,
                                                                    first_step_of_test=first_step_of_test)

                all_hiddens.append(output)
                all_beliefs.append(belief)
                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(idx)
                prev_relation = chosen_relation

            # [(B, T), 4D]

        return all_loss, all_logits, action_idx
