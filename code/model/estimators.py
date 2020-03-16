import os
import ops
import dnn
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import collections

FLAGS = flags.FLAGS


class PolicyEstimator(dnn.DNN):
    """
       Policy Function approximator. Given a observation, returns probabilities
       over all possible actions.

       Args:
          num_outputs: Size of the action space.
          reuse: If true, an existing shared network will be re-used.
          trainable: If true we add train ops to the network.
             Actor threads that don't update their local models and don't need
              train ops would set this to false.
    """
    def __init__(self, dim_input, dim_output, n_hiddens=[], scope=None):
        super(PolicyEstimator, self).__init__(dim_input, n_hiddens, dim_output)


    def __call__(self, inp, belief, activation_fn=tf.nn.relu):
        inp = tf.concat([inp, belief], axis=1)
        self.weights = self.fc_weights(name="policy_step")
        action = self.nth_hidden(self.weights, inp, activation_fn, scope="policy_step")
        return action

    def gradient_update(self, loss, lr=1e-2):
        gradients = tf.gradients(loss, list(self.weights.values()))
        self.weights = ops.gradient_update(self.weights, gradients, lr)


    def update(self, state, target, action, eps=1e-8):
        probs = tf.nn.softmax(action) + eps #TODO: softmax or exponential distr?
        # we add entropy to the loss to encourage exploration
        entropy = -tf.reduce_sum(probs * tf.log(probs), 1, name="entropy")
        entropy_mean = tf.reduce_mean(entropy, name="entropy_mean")

        # get the predictions for the chosen actions only
        gather_indices = tf.range(FLAGS.batchSize) * tf.shape(probs)[1]
        picked_action_probs = tf.gather(tf.reshape(probs, [-1]), gather_indices)

        losses = - (tf.log(picked_action_probs) * target + 0.01 * entropy)
        loss = tf.reduce_sum(losses, name="policy_loss")
        gradients = tf.gradients(loss, list(self.weights.values()))
        self.weights = ops.gradient_update(self.weights, gradients)

        return loss


class ValueEstimator(dnn.DNN):
    """
       Value function approximator. Returns a value estimator for a batch of observations

       Args:
         dim_input: size of the input dimension
         n_hiddens: hidden layers size (num units at each layer)
         dim_output: output dimension
    """
    def __init__(self, dim_input, dim_output, n_hiddens=[]):
        self.statistics = []
        super(ValueEstimator, self).__init__(dim_input, n_hiddens, dim_output)


    def __call__(self, inp, activation_fn=tf.nn.relu):
        #inp = tf.concat([inp, belief], axis=1)
        self.weights = self.fc_weights(name="value_net", reuse=tf.AUTO_REUSE)
        logits = self.nth_hidden(self.weights, inp, activation_fn)
        return tf.squeeze(logits, squeeze_dims=[1], name="logits")


    def update(self, state, target, value):
        loss = tf.squared_difference(value, target)
        gradients = tf.gradients(loss, list(self.weights.values()))
        self.weights = ops.gradient_update(self.weights, gradients)
        return loss

    def episodic_info(self, stats):
        self.statistics = stats



class BeliefEstimator(dnn.DNN):
    """
       Belief approximator. Returns beliefs for a batch of observations
    """
    def __init__(self,
                 dim_input,
                 dim_output,
                 n_hiddens=[200, 150],
                 h_size=100,
                 a_size=100,
                 stddev=.1,
                 attend_belief=False,
                 name="attention"):

        self.h_size = h_size
        dim_input = dim_input + a_size
        super(BeliefEstimator, self).__init__(dim_input, n_hiddens, dim_output)

        self.rec_attn_net = collections.OrderedDict()
        self.blf_attn_net = collections.OrderedDict()

        with tf.variable_scope(name, reuse=False) as scope:
            self.belief_net = self.fc_weights(name="belief_net", reuse=tf.AUTO_REUSE)

            self.rec_attn_net['w'] = tf.get_variable('cw_omega', (h_size, a_size),
               initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=True)
            self.rec_attn_net['b'] = tf.get_variable('cb_omega', (a_size),
                                         initializer=tf.zeros_initializer, trainable=True)
            self.rec_attn_net['u'] = tf.get_variable('cu_omega', (a_size),
                                         initializer=tf.zeros_initializer, trainable=True)
        
            self.blf_attn_net['w'] = tf.get_variable('bw_omega', (h_size, a_size),
               initializer=tf.truncated_normal_initializer(stddev=stddev), trainable=True)
            self.blf_attn_net['b'] = tf.get_variable('bb_omega', (a_size),
                                         initializer=tf.zeros_initializer, trainable=True)
            self.blf_attn_net['u'] = tf.get_variable('bu_omega', (a_size),
                                         initializer=tf.zeros_initializer, trainable=True)


    # https://github.com/TobiasLee/Text-Classification/blob/master/models/modules/attention.py
    def attention(self, inputs, time_major=False, return_alphas=True):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
         for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article

        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        """

        def attend(inputs, attn_net, scope='_context'):
            hidden_size = inputs.shape[1].value  # D value - hidden size of the RNN layer

            with tf.name_scope('v' + scope):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inputs, attn_net['w'], axes=1) + attn_net['b'])

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, attn_net['u'], axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

            output = inputs * tf.expand_dims(alphas, -1)

            return output, alphas

        if isinstance(inputs, tuple):
            # In case an attention mechanism is required for the beliefs as well
            contexts, beliefs = inputs

            ctx_results = attend(contexts, self.rec_attn_net, scope='_context')
            blf_results = attend(beliefs, self.blf_attn_net, scope='_belief')

            if not return_alphas:
                return ctx_results[0], blf_results[0]
            else:
                return ctx_results, blf_results
        else:
            output, alphas = attend(contexts, self.rec_attn_net, scope='_context')

            return output if not return_alphas else output, alphas


    def __call__(self, observation, contexts, beliefs, activation_fn=tf.nn.tanh):
        """
           Belief sampling using soft attention

           Args:
               observation: output of previous LSTM cell
               contexts: list of bottlenecks h
               beliefs: list of previously sampled beliefs
               activation_fn: activation function

           Returns:
               the belief tensor

        """
        contexts = tf.reshape(contexts, [-1, self.h_size])
        beliefs = tf.reshape(beliefs, [-1, self.h_size])
        ctx_results, blf_results = self.attention((contexts, beliefs))
        ctx_outputs, ctx_alphas = ctx_results
        blf_outputs, blf_alphas = blf_results

        x_product = tf.multiply(ctx_outputs, blf_outputs)
        x_product = tf.reshape(x_product, [-1, FLAGS.batchSize, self.h_size])
        x_product = tf.reduce_sum(x_product, axis=0)
        inputs = tf.concat([observation, x_product], axis=1)
        belief = self.nth_hidden(self.belief_net, inputs, activation_fn)

        return activation_fn(belief) if activation_fn else belief
