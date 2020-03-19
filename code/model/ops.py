from __future__ import division
import math
import numpy as np
import collections
import tensorflow as tf

from dqna_cell import DQNALSTMCell
from mi_gru_cell import MiGRUCell
from mi_lstm_cell import MiLSTMCell

from estimators import ValueEstimator
from estimators import PolicyEstimator
from estimators import BeliefEstimator

eps = 1e-20
inf = 1e30

####################################### variables ########################################

def clip_grad(grads_vars, configs):
    gradients, variables = zip(*grads_vars)
    gradients, _ = tf.clip_by_global_norm(
                       gradients,
                       configs["grad_clip_norm"],
                       use_norm=tf.linalg.global_norm(gradients))
    grads_vars = zip(gradients, variables)
    return grads_vars

def gradient_update(weights, grads, configs, lr=1e-3):
    if not isinstance(grads, dict):
        grads = collections.OrderedDict(zip(weights.keys(), grads))
    if configs["clip_grad"]:
        grads = collections.OrderedDict(zip(weights.keys(), 
           [clip_if_not_none(grads[key], -10., 10.) for key in grads.keys()]))
    lr = configs.get("lr", lr)
    weights = collections.OrderedDict(zip(weights.keys(),
        [weights[key] - lr*clip_if_not_none(grads[key], -10., 10.) \
                   for key in weights.keys()]))
    return weights

def clip_if_not_none(grad, min_value, max_value):
    if grad is None:
        grad = tf.constant([0.])
    grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
    return tf.clip_by_value(grad, min_value, max_value)

'''
Initializes a weight matrix variable given a shape and a name. 
Uses random_normal initialization if 1d, otherwise uses xavier. 
'''
def getWeight(shape, name = ""):
    with tf.variable_scope("weights"):               
        initializer = tf.initializers.GlorotUniform()
        W = tf.get_variable("weight" + name, shape = shape, initializer = initializer)
        
    return W

'''
Initializes a weight matrix variable given a shape and a name. Uses xavier
'''
def getKernel(shape, name = ""):
    with tf.variable_scope("kernels"):               
        initializer = tf.initializers.GlorotUniform()
        W = tf.get_variable("kernel" + name, shape = shape, initializer = initializer)
    return W

'''
Initializes a bias variable given a shape and a name.
'''
def getBias(shape, name = ""):
    with tf.variable_scope("biases"):              
        initializer = tf.zeros_initializer()
        b = tf.get_variable("bias" + name, shape = shape, initializer = initializer)
    return b

######################################### basics #########################################

'''
Multiplies input inp of any depth by a 2d weight matrix.  
'''
def multiply(inp, W):
    inDim = tf.shape(W)[0]
    outDim = tf.shape(W)[1] 
    newDims = tf.concat([tf.shape(inp)[:-1], tf.fill((1,), outDim)], axis = 0)
    
    inp = tf.reshape(inp, (-1, inDim))
    output = tf.matmul(inp, W)
    output = tf.reshape(output, newDims)

    return output

'''
Concatenates x and y. Support broadcasting. 
Optionally concatenate multiplication of x * y
'''
def concat(x, y, dim, mul = False, expandY = False):
    if expandY:
        y = tf.expand_dims(y, axis = -2)
        # broadcasting to have the same shape
        y = tf.zeros_like(x) + y

    if mul:
        out = tf.concat([x, y, x * y], axis = -1)
        dim *= 3
    else:
        out = tf.concat([x, y], axis = -1)
        dim *= 2
    
    return out, dim

def expand(x, y):
    y = tf.expand_dims(y, axis = -2)
    y = tf.zeros_like(x) + y
    return y

'''
Adds L2 regularization for weight and kernel variables. 
'''
# add l2 in the tf way
def L2RegularizationOp(l2):
    l2Loss = 0
    names = ["weight", "kernel"]
    for var in tf.trainable_variables():
        if any((name in var.name.lower()) for name in names):
            l2Loss += tf.nn.l2_loss(var)
    return l2 * l2Loss

######################################### attention #########################################

'''
Transform vectors to scalar logits.

Args:
    interactions: input vectors
    [batchSize, N, dim]

    dim: dimension of input vectors

    sumMod: LIN for linear transformation to scalars.
            SUM to sum up vectors entries to get scalar logit.

    dropout: dropout value over inputs (for linear case)

Return matching scalar for each interaction.
[batchSize, N]
'''
sumMod = ["LIN", "SUM"]
def inter2logits(interactions, dim, sumMod = "LIN", dropout = 1.0, name = "", reuse = None):
    with tf.variable_scope("inter2logits" + name, reuse = reuse): 
        if sumMod == "SUM":
            logits = tf.reduce_sum(interactions, axis = -1)
        else: # "LIN"
            logits = linear(interactions, dim, 1, dropout = dropout, name = "logits")
    return logits

'''
Transforms vectors to probability distribution. 
Calls inter2logits and then softmax over these.

Args:
    interactions: input vectors
    [batchSize, N, dim]

    dim: dimension of input vectors

    sumMod: LIN for linear transformation to scalars.
            SUM to sum up vectors entries to get scalar logit.

    dropout: dropout value over inputs (for linear case)

Return attention distribution over interactions.
[batchSize, N]
'''
def inter2att(interactions, dim, dropout = 1.0, mask = None, sumMod = "LIN", name = "", reuse = None):
    with tf.variable_scope("inter2att" + name, reuse = reuse): 
        logits = inter2logits(interactions, dim, dropout = dropout, sumMod = sumMod)
        if mask is not None:
            logits = expMask(logits, mask)
        attention = tf.nn.softmax(logits)    
    return attention

'''
Sums up features using attention distribution to get a weighted average over them. 
'''
def att2Smry(attention, features):
    return tf.reduce_sum(tf.expand_dims(attention, axis = -1) * features, axis = -2)

####################################### activations ########################################

'''
Performs a variant of ReLU based on config.relu
    PRM for PReLU
    ELU for ELU
    LKY for Leaky ReLU
    otherwise, standard ReLU
'''
def relu(inp, config):                  
    if config["relu"] == "PRM":
        with tf.variable_scope(None, default_name = "prelu"):
            alpha = tf.get_variable("alpha", shape = inp.get_shape()[-1], 
                initializer = tf.constant_initializer(0.25))
            pos = tf.nn.relu(inp)
            neg = - (alpha * tf.nn.relu(-inp))
            output = pos + neg
    elif config["relu"] == "ELU":
        output = tf.nn.elu(inp)
    elif config["relu"] == "SELU":
        output = tf.nn.selu(inp) 
    elif config["relu"] == "LKY":
        output = tf.maximum(inp, config["reluAlpha"] * inp)
    elif config["relu"] == "STD": # STD
        output = tf.nn.relu(inp)
    
    return output

activations = {
    "NON":      tf.identity,    
    "TANH":     tf.tanh,
    "SIGMOID":  tf.sigmoid,
    "RELU":     relu,
    "ELU":      tf.nn.elu
}

# Sample from Gumbel(0, 1)
def sampleGumbel(shape): 
    U = tf.random_uniform(shape, minval = 0, maxval = 1)
    return -tf.log(-tf.log(U + eps) + eps)

# Draw a sample from the Gumbel-Softmax distribution
def gumbelSoftmaxSample(logits, temperature): 
    y = logits + sampleGumbel(tf.shape(logits))
    return tf.nn.softmax(y / temperature)

def parametricDropout(name, train):
    var = tf.get_variable("varDp" + name, shape = (), initializer = tf.constant_initializer(2), 
        dtype = tf.float32)
    dropout = tf.cond(train, lambda: tf.sigmoid(var), lambda: 1.0)
    return dropout

###################################### sequence helpers ######################################

'''
Casts exponential mask over a sequence with sequence length.
Used to prepare logits before softmax.
'''
def expMask(seq, seqLength):
    maxLength = tf.shape(seq)[-1]
    mask = (tf.cast(tf.logical_not(tf.sequence_mask(seqLength, maxLength))), tf.float32) * (-inf)
    masked = seq + mask
    return masked

'''
Computes seq2seq loss between logits and target sequences, with given lengths.
'''
def seq2SeqLoss(logits, targets, lengths):
    mask = tf.sequence_mask(lengths, maxlen = tf.shape(targets)[1])
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.cast(mask, tf.float32))
    return loss

'''
Computes seq2seq loss between logits and target sequences, with given lengths.
    acc1: accuracy per symbol 
    acc2: accuracy per sequence
'''
def seq2seqAcc(preds, targets, lengths):
    mask = tf.sequence_mask(lengths, maxlen = tf.shape(targets)[1])
    corrects = tf.logical_and(tf.equal(preds, targets), mask)
    numCorrects = tf.reduce_sum(tf.cast(corrects, tf.int32), axis = 1)
    
    acc1 = tf.cast(numCorrects, tf.float32) / (tf.cast(lengths, tf.float32) + eps) # add small eps instead?
    acc1 = tf.reduce_mean(acc1)  
    
    acc2 = tf.cast(tf.equal(numCorrects, lengths), tf.float32)
    acc2 = tf.reduce_mean(acc2)      

    return acc1, acc2

def hingeLoss(labels, logits):
    maxLogit = tf.reduce_max(logits * (1 - labels), axis = 1, keepdims = True)  
    losses = tf.nn.relu((1 + maxLogit - logits) * labels)
    losses = tf.reduce_sum(losses, axis = 1) # reduce_max reduce sum will also work
    return losses
    #final_loss = tf.reduce_mean(tf.reduce_max(L, axis = 1))

########################################### linear ###########################################

'''
linear transformation.

Args:
    inp: input to transform
    inDim: input dimension
    outDim: output dimension
    dropout: dropout over input
    batchNorm: if not None, applies batch normalization to inputs
    addBias: True to add bias
    bias: initial bias value
    act: if not None, activation to use after linear transformation
    actLayer: if True and act is not None, applies another linear transformation on top of previous
    actDropout: dropout to apply in the optional second linear transformation
    retVars: if True, return parameters (weight and bias) 

Returns linear transformation result.
'''
# batchNorm = {"decay": float, "train": Tensor}
# actLayer: if activation is not non, stack another linear layer
# maybe change naming scheme such that if name = "" than use it as default_name (-->unique?)
def linear(inp, inDim, outDim, dropout = 1.0, 
    batchNorm = None, addBias = True, bias = 0.0,
    act = "NON", actLayer = True, actDropout = 1.0, 
    retVars = False, name = "", reuse = None):
    
    with tf.variable_scope("linearLayer" + name, reuse = reuse):        
        W = getWeight((inDim, outDim) if outDim > 1 else (inDim, ))
        b = getBias((outDim, ) if outDim > 1 else ()) + bias
        
        if batchNorm is not None:
            inp = tf.contrib.layers.batch_norm(inp, decay = batchNorm["decay"], 
                center = True, scale = True, is_training = batchNorm["train"], updates_collections = None)
            # tf.layers.batch_normalization, axis -1 ?

        inp = tf.nn.dropout(inp, dropout)                
        
        if outDim > 1:
            output = multiply(inp, W)
        else:
            output = tf.reduce_sum(inp * W, axis = -1)
        
        if addBias:
            output += b

        output = activations[act](output)

        # good?
        if act != "NON" and actLayer:
            output = linear(output, outDim, outDim, dropout = actDropout, batchNorm = batchNorm,  
                addBias = addBias, act = "NON", actLayer = False, 
                name = name + "_2", reuse = reuse)

    if retVars:
        return (output, (W, b))

    return output

'''
Computes Multi-layer feed-forward network.

Args:
    features: input features
    dims: list with dimensions of network. 
          First dimension is of the inputs, final is of the outputs.
    batchNorm: if not None, applies batchNorm
    dropout: dropout value to apply for each layer
    act: activation to apply between layers.
    NON, TANH, SIGMOID, RELU, ELU
'''
# no activation after last layer
# batchNorm = {"decay": float, "train": Tensor}
def FCLayer(features, dims, batchNorm = None, dropout = 1.0, act = "RELU"):
    layersNum = len(dims) - 1
    
    for i in range(layersNum):
        features = linear(features, dims[i], dims[i+1], name = "fc_%d" % i, 
            batchNorm = batchNorm, dropout = dropout)
        # not the last layer
        if i < layersNum - 1: 
            features = activations[act](features)
    
    return features   

######################################## rnns ########################################

'''
Creates an RNN cell.

Args:
    hdim: the hidden dimension of the RNN cell.
    
    reuse: whether the cell should reuse parameters or create new ones.
    
    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM, ProjLSTM

    act: the cell activation
    NON, TANH, SIGMOID, RELU, ELU

    projDim: if ProjLSTM, the dimension for the states projection

Returns the cell.
'''
def createCell(hDim, reuse, cellType = None, act = None, projDim = None, name=None):
    if cellType is None:
        cellType = "DQNA"

    activation = activations.get(act, None)

    cells = {
        "RNN": tf.nn.rnn_cell.BasicRNNCell,
        "GRU": tf.nn.rnn_cell.GRUCell,
        "DQNA": DQNALSTMCell,
        "LSTM": tf.nn.rnn_cell.BasicLSTMCell,
        "MiGRU": MiGRUCell,
        "MiLSTM": MiLSTMCell,
        "INFeRNN": tf.nn.rnn_cell.BasicLSTMCell #TODO
    }

    cell = cells[cellType](hDim, reuse = reuse, activation = activation, name = cellType)

    return cell

'''
Runs an forward RNN layer.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM, ProjLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize). 

Returns the outputs sequence and final RNN state.  
'''
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
def fwRNNLayer(inSeq, seqL, hDim, cellType = None, dropout = 1.0, varDp = None, 
    name = "", reuse = None): # proj = None
    
    with tf.variable_scope("rnnLayer" + name, reuse = reuse):
        batchSize = tf.shape(inSeq)[0]

        cell = createCell(hDim, reuse, cellType) # passing reuse isn't mandatory

        if varDp is not None:
            cell = tf.contrib.rnn.DropoutWrapper(cell, 
                state_keep_prob = varDp["stateDp"],
                input_keep_prob = varDp["inputDp"],
                variational_recurrent = True, input_size = varDp["inputSize"], dtype = tf.float32)
        else:           
            inSeq = tf.nn.dropout(inSeq, dropout)
        
        initialState = cell.zero_state(batchSize, tf.float32)

        outSeq, lastState = tf.nn.dynamic_rnn(cell, inSeq, 
            sequence_length = seqL, 
            initial_state = initialState,
            swap_memory = True)
            
        if isinstance(lastState, tf.nn.rnn_cell.LSTMStateTuple):
            lastState = lastState.h

        # if proj is not None:
        #     if proj["output"]:
        #         outSeq = linear(outSeq, cell.output_size, proj["dim"], act = proj["act"],  
        #             dropout = proj["dropout"], name = "projOutput")

        #     if proj["state"]:
        #         lastState = linear(lastState, cell.state_size, proj["dim"], act = proj["act"],  
        #             dropout = proj["dropout"], name = "projState")

    return outSeq, lastState

'''
Runs an bidirectional RNN layer.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize).   

Returns the outputs sequence and final RNN state.     
'''
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
def biRNNLayer(inSeq, seqL, hDim, cellType = None, dropout = 1.0, varDp = None, 
    name = "", reuse = None): # proj = None, 

    with tf.variable_scope("birnnLayer" + name, reuse = reuse):
        batchSize = tf.shape(inSeq)[0]

        with tf.variable_scope("fw"):
            cellFw = createCell(hDim, reuse, cellType)
        with tf.variable_scope("bw"):
            cellBw = createCell(hDim, reuse, cellType)
        
        if varDp is not None:
            cellFw = tf.contrib.rnn.DropoutWrapper(cellFw, 
                state_keep_prob = varDp["stateDp"],
                input_keep_prob = varDp["inputDp"],
                variational_recurrent = True, input_size = varDp["inputSize"], dtype = tf.float32)
            
            cellBw = tf.contrib.rnn.DropoutWrapper(cellBw, 
                state_keep_prob = varDp["stateDp"],
                input_keep_prob = varDp["inputDp"],
                variational_recurrent = True, input_size = varDp["inputSize"], dtype = tf.float32)            
        else:
            inSeq = tf.nn.dropout(inSeq, dropout)

        initialStateFw = cellFw.zero_state(batchSize, tf.float32)
        initialStateBw = cellBw.zero_state(batchSize, tf.float32)

        (outSeqFw, outSeqBw), (lastStateFw, lastStateBw) = tf.nn.bidirectional_dynamic_rnn(
            cellFw, cellBw, inSeq, 
            sequence_length = seqL, 
            initial_state_fw = initialStateFw, 
            initial_state_bw = initialStateBw,
            swap_memory = True)

        if isinstance(lastStateFw, tf.nn.rnn_cell.LSTMStateTuple):
            lastStateFw = lastStateFw.h # take c? 
            lastStateBw = lastStateBw.h  

        outSeq = tf.concat([outSeqFw, outSeqBw], axis = -1)
        lastState = tf.concat([lastStateFw, lastStateBw], axis = -1)

        # if proj is not None:
        #     if proj["output"]:
        #         outSeq = linear(outSeq, cellFw.output_size + cellFw.output_size, 
        #             proj["dim"], act = proj["act"], dropout = proj["dropout"], 
        #             name = "projOutput")

        #     if proj["state"]:
        #         lastState = linear(lastState, cellFw.state_size + cellFw.state_size, 
        #             proj["dim"], act = proj["act"], dropout = proj["dropout"], 
        #             name = "projState")

    return outSeq, lastState

# int(hDim / 2) for biRNN?
'''
Runs an RNN layer by calling biRNN or fwRNN.

Args:
    inSeq: the input sequence to run the RNN over.
    [batchSize, sequenceLength, inDim]
    
    seqL: the sequence matching lengths.
    [batchSize, 1]

    hDim: hidden dimension of the RNN.

    bi: true to run bidirectional rnn.

    cellType: the cell type 
    RNN, GRU, LSTM, MiGRU, MiLSTM

    dropout: value for dropout over input sequence

    varDp: if not None, state and input variational dropouts to apply.
    dimension of input has to be supported (inputSize).   

Returns the outputs sequence and final RNN state.     
'''
# proj = {"output": bool, "state": bool, "dim": int, "dropout": float, "act": str}
# varDp = {"stateDp": float, "inputDp": float, "inputSize": int}
def RNNLayer(inSeq, seqL, hDim, bi = None, cellType = None, dropout = 1.0, varDp = None, 
    name = "", reuse = None): # proj = None
    
    with tf.variable_scope("rnnLayer" + name, reuse = reuse):
        rnn = biRNNLayer if bi else fwRNNLayer
        
        if bi:
            hDim = int(hDim / 2)

    return rnn(inSeq, seqL, hDim, cellType = cellType, dropout = dropout, varDp = varDp) # , proj = proj

# tf counterpart?
def multigridRNNLayer(features, h, w, dim, name = "", reuse = None):
    with tf.variable_scope("multigridRNNLayer" + name, reuse = reuse):
        features = linear(features, dim, dim / 2, name = "i")

        output0 = gridRNNLayer(features, h, w, dim, right = True, down = True, name = "rd")
        output1 = gridRNNLayer(features, h, w, dim, right = True, down = False, name = "r")
        output2 = gridRNNLayer(features, h, w, dim, right = False, down = True, name = "d")
        output3 = gridRNNLayer(features, h, w, dim, right = False, down = False, name = "NON")

        output = tf.concat([output0, output1, output2, output3], axis = -1)
        output = linear(output, 2 * dim, dim, name = "o")

    return outputs

###################################### belief ##################################### 

'''
Belief sampling: sampling a belief vector based on some contextual information. Such
information is composed by previous beliefs and recurrent states
'''
def initBeliefEstimator(batch_size, dim_input, dim_output, hiddens, hidden_size, attention_size, estimator="LVM"):
    '''
       Args:
          observations: input at time step t-1
          prevHiddens: previous hidden states of the RNN
          prevBeliefs: previous beliefs held by the agent
          currHidden: current hidden code
    '''
    models = {
       "LVM": BeliefEstimator # Latent variable model
    }

    estimator_belief = models[estimator](
                            batch_size,
                            dim_input,
                            dim_output,
                            hiddens,
                            hidden_size,
                            attention_size)

    return estimator_belief
 
################################## policy network ################################# 

'''
Policy network: sampling of an action. Every action is a tuple composed of weights.
The weights are intended to modulate some gates -- forget and input gates if the RNN is
an LSTM, and reset gate only if the RNN is a GRU -- and the memory cell.
'''
def initPolicyEstimator(dim_input, dim_output, fc_layers=[100]):
    return PolicyEstimator(dim_input, dim_output, fc_layers)

################################# value network ################################### 

'''
Value network:
'''
def initValueEstimator(dim_input, dim_output, fc_layers=[100]):
    return ValueEstimator(dim_input, dim_output, fc_layers)

############################### variational dropout ###############################

'''
Generates a variational dropout mask for a given shape and a dropout 
probability value.
'''
def generateVarDpMask(shape, keepProb):
    randomTensor = tf.to_float(keepProb)
    randomTensor += tf.random_uniform(shape, minval = 0, maxval = 1)
    binaryTensor = tf.floor(randomTensor)
    mask = tf.to_float(binaryTensor)
    return mask

'''
Applies the a variational dropout over an input, given dropout mask 
and a dropout probability value. 
'''
def applyVarDpMask(inp, mask, keepProb):
    ret = (tf.div(inp, tf.to_float(keepProb))) * mask
    return ret   
