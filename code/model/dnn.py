import math
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import collections
import utils

FLAGS = flags.FLAGS


class DNN(object):
    def __init__(self, dim_input, n_hiddens, dim_output, **kwargs):
        """
           Base Deep neural network with some util methods

           Args:
               dim_input: input dimensionality.
               n_hiddens: List containing dim of hidden layers.
               dim_output: output dimensionality.
               num_units: dimensionality of the hidden codes.
               kwargs: additional args for constructing particle nets.
        """
        self.dim_input = dim_input
        self.n_hiddens = n_hiddens
        self.dim_output = dim_output
        self.width = self.height = kwargs.get("width", 64)
        self.flatten = (int(math.ceil\
           (self.width / (2**len(self.n_hiddens))
           ))**2) * self.n_hiddens[-1]
   

    def nth_hidden(self, WW, X, activation_fn=tf.nn.relu, reuse=None, scope=""):
        """ Last activation layer """
        forward = self.conv_forward if 'conv' in WW else self.fc_forward
        return forward(WW, X, activation_fn, reuse=reuse, scope=scope)


    def conv_weights(self, stddev=.1, name="", reuse=None):
        params = collections.OrderedDict()
        conv_init = tf.contrib.layers.xavier_initializer_conv2d(seed=config.seed, dtype=tf.float32)
        fc_init = tf.contrib.layers.xavier_initializer(seed=config.seed, dtype=tf.float32)

        with tf.variable_scope(name, reuse=reuse) as scope:
            k, n_channels = config.kernelSize, config.channels
            params['conv_w1'] = tf.get_variable('conv_w1', [k, k, n_channels, self.n_hiddens[0]],
                           initializer=conv_init, dtype=tf.float32)
            params['conv_b1'] = tf.Variable(tf.zeros([self.n_hiddens[0]]), name='conv_b1')
            for i in range(1, len(self.n_hiddens)):
                params['conv_w'+str(i+1)] = tf.get_variable(
                           'conv_w'+str(i+1), 
                           [k, k, self.n_hiddens[i-1], self.n_hiddens[i-1]],
                           initializer=conv_init, dtype=tf.float32)
                params['conv_b'+str(i+1)] = tf.Variable(tf.zeros([self.n_hiddens[i-1]]),
                                            name='conv_v'+str(i+1))
            params['w1'] = tf.get_variable('w1', [self.flatten, self.dim_output], initializer=fc_init)
            params['b1'] = tf.Variable(tf.zeros([self.dim_output], name='b1'))

        return params

 
    def fc_weights(self, stddev=.1, name="", reuse=None):
        params = collections.OrderedDict()
        n_hiddens = [self.dim_output] if not self.n_hiddens else self.n_hiddens

        init = tf.truncated_normal_initializer(stddev=stddev)

        with tf.variable_scope(name, reuse=reuse) as scope:
            params['w1'] = tf.get_variable('w1', (self.dim_input, n_hiddens[0]), initializer=init)
            params['b1'] = tf.get_variable('b1', (n_hiddens[0]), initializer=tf.zeros_initializer)

            if not self.n_hiddens: return params

            for i in range(1, len(self.n_hiddens)):
                params['w'+str(i+1)] = tf.get_variable('w'+str(i+1), (self.n_hiddens[i-1],
                                                       self.n_hiddens[i]), initializer=init)
                params['b'+str(i+1)] = tf.get_variable('b'+str(i+1), (self.n_hiddens[i]),
                                                       initializer=tf.zeros_initializer)

            n_hiddens = len(self.n_hiddens) + 1
            params['w'+str(n_hiddens)] = tf.get_variable('w'+str(n_hiddens),
                                           (self.n_hiddens[-1], self.dim_output), initializer=init)
            params['b'+str(n_hiddens)] = tf.get_variable('b'+str(n_hiddens),
                                           (self.dim_output), initializer=tf.zeros_initializer)
        return params


    def fc_forward(self, weights, inputs, activation=tf.nn.relu, reuse=False, scope=""):
        """ 
           Feed forward of an input on a fully connected neural network 

           Args:
               weights (dict): neural network weights and biases
               inputs (tensor):  inputs to the neural network
               activation: activation function for the layers

           Returns:
               Logits (the activation function is not applied)
        """
        hidden = tf.matmul(inputs, weights['w1']) + weights['b1']
        hidden = utils.normalize(hidden, activation=activation,
                                 reuse=reuse, scope=scope)

        if not self.n_hiddens: return hidden

        for i in range(1, len(self.n_hiddens)):
            hidden = utils.normalize(
                         tf.matmul(hidden, weights['w'+str(i+1)]) + \
                              weights['b'+str(i+1)],
                         activation=activation, reuse=reuse,
                         scope=scope+str(i+1))

        return tf.matmul(hidden, weights['w'+str(len(self.n_hiddens)+1)]) + \
                                 weights['b'+str(len(self.n_hiddens)+1)] 


    def conv_forward(self, WW, X, scope='', activation=tf.nn.relu, reuse=False):
        X = tf.reshape(X, [-1, self.height, self.width, self.n_channels])
        hidden = utils.conv_block(X, WW['conv_w1'], WW['conv_b1'], reuse, scope=scope+"_1")

        for i in range(1, len(self.n_hiddens)):
            hidden = utils.conv_block(hidden, WW['conv_w'+str(i+1)], WW['conv_b'+str(i+1)],
                                      reuse, scope=scope+"_"+str(i+1))
        f_hidden = tf.reshape(hidden, [-1, self.flatten])
        return tf.matmul(f_hidden, WW['w1']) + WW['b1']
