import cell_state
import numpy as np
import collections
import tensorflow as tf


LSTMActionTuple = collections.namedtuple("LSTMActionTuple", ("c_weight", "i_weight",
                                         "f_weight", "o_weight"))


class DQNALSTMCell(tf.nn.rnn_cell.LSTMCell):
    def __init__(self, 
                 num_units,
                 action_units,
                 belief_units,
                 forget_bias = 1.0,
                 input_size = None,
                 numHiddens = [],
                 use_peepholes=True,
                 state_is_tuple = True,
                 activation = tf.nn.tanh,
                 name = None,
                 reuse = None,
                 noisin = None):

        self.numUnits = num_units
        self.actionUnits = action_units
        self.beliefUnits = belief_units
        self.numHiddens = numHiddens

        self.forgetBias = forget_bias
        self.activation = \
            tf.nn.tanh if not activation else activation

        self._noisin = noisin
        self._stateIsTuple = state_is_tuple


        super(DQNALSTMCell, self).__init__(num_units,
              name=name, reuse=reuse)


    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self.numUnits, self.numUnits)


    @property
    def output_size(self):
        return self.numUnits

    def nth_hidden(self, weights, inputs, activation_fn=tf.nn.tanh):
        hidden = activation_fn(tf.matmul(inputs, weights["w1"]) + weights["b1"])

        if not self.numHiddens: return hidden

        for i in range(1, len(self.numHiddens)):
            hidden = activation_fn(tf.matmul(hidden, weights["w" + str(i+1)])) + weights["b" + str(i+1)]

        return tf.matmul(hidden, weights["w" + str(len(self.numHiddens)+1)]) + \
            weights["b" + str(len(self.numHiddens)+1)]


    def mulWeights(self, inp, inDim, outDim, name = ""):
        with tf.variable_scope("weights", reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name, shape = (inDim, outDim),
                initializer = tf.initializers.glorot_uniform(), trainable=True)
        output = tf.matmul(inp, W)
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

        inputSize = int(inputs.shape[1])
        scope = scope or type(self).__name__

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxi")
            Uh = self.mulWeights(state.h, self.numUnits, self.numUnits, name = "Uhi")
 
            i = self.addBiases(Wx, Uh, self.numUnits, name = "i")

            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxj")
            Uh = self.mulWeights(state.h, self.numUnits, self.numUnits, name = "Uhj")
            
            j = self.addBiases(Wx, Uh, self.numUnits, name = "l")

            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxf")
            Uh = self.mulWeights(state.h, self.numUnits, self.numUnits, name = "Uhf")
            
            f = self.addBiases(Wx, Uh, self.numUnits, name = "f")

            Wx = self.mulWeights(inputs, inputSize, self.numUnits, name = "Wxo")
            Uh = self.mulWeights(state.h, self.numUnits, self.numUnits, name = "Uho")
            
            o = self.addBiases(Wx, Uh, self.numUnits, name = "o")
            action = LSTMActionTuple(*tf.split(state.a, 4, axis=1))

            newC = (action.c_weight * state.c) \
                 * (action.f_weight * tf.nn.sigmoid(f + self.forgetBias)) \
                 + (action.i_weight * tf.nn.sigmoid(i) * self.activation(j))

            newH = self.activation(newC) * action.o_weight * tf.nn.sigmoid(o)
            newState = cell_state.DQNACellState(newC, newH)

        return newH, newState


    def zero_state(self, batchSize, dtype = tf.float32):
        return cell_state.DQNACellState(
            tf.zeros((batchSize, self.numUnits), dtype = dtype),
            tf.zeros((batchSize, self.numUnits), dtype = dtype),
            tf.zeros((batchSize, self.actionUnits), dtype = dtype),
            tf.zeros((batchSize, self.beliefUnits), dtype = dtype))


    def init_cell_state(self, batchSize):
        return self.zero_state(batchSize)
