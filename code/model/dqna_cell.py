import cell_state
import numpy as np
import collections
import tensorflow as tf

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

LSTMActionTuple = collections.namedtuple("LSTMActionTuple", ("weight_c", "weight_i", "weight_f"))


class DQNALSTMCell(tf.nn.rnn_cell.LSTMCell):
    def __init__(self, 
                 num_units,
                 forget_bias = 1.0,
                 input_size = None,
                 state_is_tuple = True,
                 activation = tf.nn.tanh,
                 name = None,
                 reuse = None,
                 noisin = None):

        self.numUnits = num_units
        self.forgetBias = forget_bias
        self.activation = \
            tf.nn.tanh if not activation else activation
        self._noisin = noisin
        self.batchSize = FLAGS.batchSize
        self._stateIsTuple = state_is_tuple

        super(DQNALSTMCell, self).__init__(num_units,
              name=name, reuse=reuse)


    @property
    def state_size(self):
        return self.zero_state(self.batchSize)


    @property
    def output_size(self):
        return self.numUnits


    def mulWeights(self, inp, inDim, outDim, name = ""):
        with tf.variable_scope("weights", reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name, shape = (inDim, outDim),
                initializer = tf.contrib.layers.xavier_initializer(), trainable=True)
        num_samples = int(self.batchSize)
        jitter = self._distrib().sample(num_samples)
        output = tf.matmul(inp, W) + jitter
        return output


    def addBiases(self, inp1, inp2, dim, name = ""):
        with tf.variable_scope("additiveBiases" + name, reuse=tf.AUTO_REUSE):
            b = tf.get_variable("biases", shape = (dim,), 
                initializer = tf.zeros_initializer())
        with tf.variable_scope("multiplicativeBias" + name, reuse=tf.AUTO_REUSE):
            beta = tf.get_variable("biases", shape = (3 * dim,), 
                initializer = tf.ones_initializer())

        Wx, Uh, inter = tf.split(beta * tf.concat([inp1, inp2, inp1 * inp2], axis = 1), 
            num_or_size_splits = 3, axis = 1)
        output = Wx + Uh + inter + b
        return output


    def _distrib(self):
        class ZeroDistrib:
            def __init__(self, numUnits):
                self.numUnits = numUnits

            def sample(self, num_samples):
                return tf.zeros(shape=(num_samples, self.numUnits))
        if not self._noisin:
            return ZeroDistrib(self.numUnits)
        # see https://arxiv.org/pdf/1805.01500.pdf for good noisin (.41)    
        gamma  = self._noisin * (1. - self._noisin)
        import tensorflow_probability as tfp
        return tfp.distributions.Bernouilli(probs=[gamma]*self.numUnits)


    def __call__(self, inputs, state, scope = None):
        """
           :param inputs: input data to pass to the LSTM cell
           :param state: previous state of the LSTM cell
        """
        scope = scope or type(self).__name__
        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            c, h, a = state.c, state.h, state.a
            inputSize = int(inputs.shape[1])

            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxi")
            Uh = self.mulWeights(h, self.numUnits, self.numUnits, name = "Uhi")
 
            i = self.addBiases(Wx, Uh, self.numUnits, name = "i")

            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxj")
            Uh = self.mulWeights(h, self.numUnits, self.numUnits, name = "Uhj")
            
            j = self.addBiases(Wx, Uh, self.numUnits, name = "l")

            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxf")
            Uh = self.mulWeights(h, self.numUnits, self.numUnits, name = "Uhf")
            
            f = self.addBiases(Wx, Uh, self.numUnits, name = "f")

            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxo")
            Uh = self.mulWeights(h, self.numUnits, self.numUnits, name = "Uho")
            
            o = self.addBiases(Wx, Uh, self.numUnits, name = "o")
            concat = tf.multiply(tf.concat([c, f, i, o], axis=1), a)
            c, f, i, o = tf.split(value=concat, num_or_size_splits=4, axis = 1)

            newC = (c * tf.nn.sigmoid(f + self.forgetBias) + tf.nn.sigmoid(i) \
                 * self.activation(j))
            newH = self.activation(newC) * tf.nn.sigmoid(o)

            newH = newH + tf.random_normal(shape=tf.shape(newC), stddev=.1) \
                 if FLAGS.noisin else newH

            newState = cell_state.DQNACellState(newC, newH)
        return newH, newState


    def zero_state(self, batchSize, dtype = tf.float32):
        jitter = self._distrib().sample(batchSize)
        return cell_state.DQNACellState(
            tf.zeros((batchSize, self.numUnits), dtype = dtype) + jitter,
            tf.zeros((batchSize, self.numUnits), dtype = dtype) + jitter)


    def init_cell_state(self, batchSize):
        return self.zero_state(batchSize)
