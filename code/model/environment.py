from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging

logger = logging.getLogger()


class Episode(object):

    def __init__(self, graph, data, params):

        start_entities, query_relation, end_entities, all_answers, all_paths, all_lengths = data
        self.batch_size, self.path_len, num_rollouts, test_rollouts, mode, _, reward_shaper = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts

        self.grapher = graph
        self.current_hop = 0
        self.reward_shaper = reward_shaper
        self.no_examples = start_entities.shape[0]
        self.positive_reward = reward_shaper.positive_reward
        self.negative_reward = reward_shaper.negative_reward
        self.rollout_size = self.batch_size * self.num_rollouts

        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)

        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers

        self.all_paths, self.all_lengths = all_paths, all_lengths

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
        last_step = self.current_hop == self.path_len # because self.current_hop starts at 1 in __call__
        return self.reward_shaper.get_reward(self.current_entities, self.end_entities, self.all_paths, last_step)

    def __call__(self, action):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(
            self.no_examples*self.num_rollouts), action]

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state


class RewardShaper(object):
    def __init__(self, params, mode='train', scope='reward_shaping'):
        self.path_length = params['path_length']
        self.batch_size = params['batch_size']
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

        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = params['num_rollouts']
        else:
            self.num_rollouts = params['test_rollouts']

        self.rollout_size = self.batch_size * self.num_rollouts

        if self._intrinsic_reward:
            cells = []
            with tf.variable_scope(scope):
                for _ in range(self.LSTM_Layers):
                    cells.append(tf.nn.rnn_cell.BasicLSTMCell(
                        self.m * self.hidden_size))
                self.reward_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
                    cells, state_is_tuple=True)

        self.lookup_table = params['entity_lookup_table']
        self.embed = lambda entities: tf.nn.embedding_lookup(self.lookup_table, entities)

    def encode(self, inference_path, sequence_length=None):
        def lookup(inputs): return self.embed(inputs)
        path_embeddings = tf.map_fn(fn=lookup, elems=inference_path, dtype=tf.float32)
        enc_outputs, _ = tf.nn.dynamic_rnn(self.reward_rnn_cell, path_embeddings,
                                           dtype=tf.float32, sequence_length=sequence_length)
        return tf.reshape(enc_outputs, [self.batch_size, -1])

    def intrinsic_reward(self, correct, inference_path):
        id_time, results = len(inference_path), []
        indices = tf.expand_dims(tf.range(self.batch_size), 1)

        def get_target(time):
            id_time = tf.tile([time], (self.batch_size,))
            id_time = tf.expand_dims(id_time, 1)
            indexes = tf.concat([indices, id_time], 1)

            targets = tf.gather_nd(correct, indexes)
            targets = tf.cast(targets, tf.int32)
            targets = tf.expand_dims(targets, 1)

            return tf.concat([indices, targets], 1)

        def projection(inputs, reuse=tf.AUTO_REUSE):
            with tf.variable_scope("transform", reuse=reuse):
                h = tf.layers.Dense(
                    self.num_choices, activation=tf.nn.softmax)(inputs)
            return h

        def reward_fn(inputs): return tf.gather_nd(
            inputs[0], get_target(inputs[1]))

        enc_outputs = self.encode(inference_path)
        enc_outputs = tf.reshape(enc_outputs, [self.num_rollouts, id_time, self.batch_size, -1])
        for i in range(self.num_rollouts):
            probs = [projection(enc_outputs[i][j]) for j in range(id_time)]
            probs = tf.reshape(probs, [id_time, self.batch_size, -1])
            results.append(tf.map_fn(reward_fn, dtype=tf.float32,
                                     elems=(probs, tf.range(id_time)), parallel_iterations=id_time))

        reward = tf.reshape(results, [self.rollout_size, id_time])
        return tf.reduce_prod(reward, axis=1)

    def extrinsic_reward(self, current_entities, end_entities):
        reward = (current_entities == end_entities)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        reward = np.select(condlist, choicelist)  # [B,]
        return reward

    def get_reward(self, current_entities, end_entities, paths, last_step=False):
        if self._intrinsic_reward and not last_step:
            self.inference_path.append(current_entities)
            reward = self.intrinsic_reward(paths, self.inference_path)
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

        scope = params.get('shaper_scope', 'reward_shaping')
        self.reward_shaper = RewardShaper(params, mode=mode, scope=scope)

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
