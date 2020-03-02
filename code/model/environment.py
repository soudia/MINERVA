from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging

logger = logging.getLogger()

import tensorflow as tf


class Episode(object):

    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, mode, batcher, reward_shaper = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts

        self.current_hop = 0
        self.reward_shaper = reward_shaper
        start_entities, query_relation, end_entities, all_answers, all_paths, all_lengths = data
        self.no_examples = start_entities.shape[0]
        self.positive_reward = reward_shaper.positive_reward
        self.negative_reward = reward_shaper.negative_reward

        start_entities = np.repeat(start_entities, self.num_rollouts)
        all_paths = np.repeat(all_paths, self.num_rollouts)
        all_lengths = np.repeat(all_lengths, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)

        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        self.relational_lengths = all_lengths
        self.relational_paths = np.reshape(all_paths, [self.batch_size * self.num_rollouts, self.path_len])

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        last_step = self.current_hop == self.path_len - 1
        return self.reward_shaper.get_reward(self.current_entities, self.end_entities, self.relational_paths, last_step)

    def __call__(self, action):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action]

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts )

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state


class RewardShaper(object):
    def __init__(self, params, scope='reward_shaping'):
        self.path_length = params['path_length']
        self.batch_size  = params['batch_size']
        self.LSTM_Layers = params['LSTM_layers']
        self.hidden_size = params['hidden_size']
        self._intrinsic_reward = params['intrinsic_reward']

        self.inference_path = []
        self.num_choices = params['num_choices']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']

        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        if self._intrinsic_reward:
            with tf.variable_scope(scope):
                for _ in range(self.LSTM_Layers):
                    cells.append(tf.nn.rnn_cell.BasicLSTMCell(self.m * self.hidden_size))
            self.reward_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            self.state = self.reward_shaper.zero_state(self.batch_size, dtype=tf.float32)

        lookup_table = params['entity_lookup_table']
        self.ent_embedding = lambda entities: tf.nn.embedding_lookup(lookup_table, entities)

        lookup_table = params['relation_lookup_table']
        self.rel_embedding = lambda relations: tf.nn.embedding_lookup(lookup_table, relations)

    def evaluate(self, path_embeddings, sequence_lengths=None):
        enc_outputs, _ = tf.nn.dynamic_rnn(self.reward_rnn_cell, path_embeddings,
                                 dtype=tf.float32, sequence_length=sequence_lengths)
        return enc_outputs


    def cross_entropy(self, path_embeddings, sequence_lengths, labels):
        enc_outputs = self.evaluate(path_embeddings, sequence_lengths)
        xent_loss = \
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=enc_outputs, labels=labels)
        return xent_loss


    def intrinsic_reward(self, time, path_embeddings, relational_paths):
        id_time = tf.tile([time], (self.batch_size, ))
        id_time = tf.expand_dims(id_time, 1)
        indices = tf.range(self.batch_size)
        indices = tf.expand_dims(indices, 1)
        indices = tf.concat([indices, id_time], 1)

        targets = tf.gather_nd(relational_paths, indices)
        targets = tf.cast(targets, tf.int32)

        gather_fn = lambda inputs : tf.gather(inputs[0], inputs[1])
        prob_fn = lambda logits, targets, activation_fn=tf.nn.softmax: \
            tf.map_fn(gather_fn, elems=(activation_fn(logits), targets),
                      parallel_iterations=self.batch_size, dtype=tf.float32)

        enc_outputs = self.evaluate(path_embeddings)
        rewards = tf.reduce_prod(tf.map_fn(fn=reward_fn, elems=enc_outputs), 0)

        return rewards


    def extrinsic_reward(self, current_entities, end_entities):
        reward = (current_entities == end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward


    def get_reward(self, current_entities, end_entities, relational_paths, last_step=False):
        if self._intrinsic_reward and not last_step:
            self.inference_path.append(current_entities)
            path_embeddings = tf.map_fn(fn=self.ent_embedding, elems=self.inference_path, dtype=tf.float32)
            path_embeddings = tf.reshape(path_embeddings, [self.batch_size, self.path_length, -1])
            time = tf.constant(len(self.inference_path))
            reward = self.intrinsic_reward(time, path_embeddings, relational_paths)
        else:
            reward = self.extrinsic_reward(current_entities, end_entities)
            self.inference_path.clear()
        return reward



class env(object):
    def __init__(self, params, mode='train'):

        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        input_dir = params['data_input_dir']
        self.intrinsic_reward = params['intrinsic_reward'] and params['LSTM_layers'] > 0

        if mode == 'train':
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 batch_size=params['batch_size'],
                                                 path_length=params['path_length'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'])
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 mode=mode,
                                                 batch_size=params['batch_size'],
                                                 path_length=params['path_length'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'])

            self.total_no_examples = self.batcher.store.shape[0]

        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'])

        self.reward_shaper = RewardShaper(params)


    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.mode, self.batcher, self.reward_shaper
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():

                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
