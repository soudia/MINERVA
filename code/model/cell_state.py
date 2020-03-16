import numpy as np
import tensorflow as tf
import collections

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

DQNACellTuple = collections.namedtuple("DQNACellTuple", ("control", "actions", "beliefs"))
DQNACellTuple.__new__.__defaults__ = ([],) * len(DQNACellTuple._fields)

class DQNACellState(tf.nn.rnn_cell.LSTMStateTuple):
    def __new__(cls, c, h, a=None, b=None):
        self = super(DQNACellState, cls).__new__(cls, c, h)
        self.a = a if a is not None else tf.zeros([FLAGS.batchSize, 4*100])
        self.b = b if b is not None else tf.zeros([FLAGS.batchSize, FLAGS.hiDim])
        self.cell_tuple = self._newCellTuple()
        return self

    def _newCellTuple(self, fromCellTuple=None):
        control = fromCellTuple.control if fromCellTuple else [] 
        actions = fromCellTuple.actions if fromCellTuple else []
        beliefs = fromCellTuple.beliefs if fromCellTuple else []

        control.append((self.c, self.h))
        if self.a is not None:
            actions.append(self.a)
        if self.b is not None:
            beliefs.append(self.b)

        return DQNACellTuple(control, actions, beliefs)

    def append(self, c, h, a, b):
        a = a if a is not None else tf.ones([FLAGS.batchSize, 4*100])
        b = b if b is not None else tf.random_normal(
                     shape=[FLAGS.batchSize, FLAGS.hiDim], stddev=.01)
        self.cell_tuple.actions.append(a)
        self.cell_tuple.beliefs.append(b)
        self.cell_tuple.control.append((c, h))
        return self

    def set_belief(self, b):
        if b is None:
            raise ValueError("belief is not valid!")
        self.cell_tuple.beliefs.append(b)
        return self

    def set_action(self, a):
        if a is None:
            raise ValueError("action is not valid!")
        self.cell_tuple.actions.append(a)
        return self

    def set_control(self, c, h):
        if c is None:
            raise ValueError("memory cell is not valid!")
        if h is None:
            raise ValueError("hidden state is not valid!")
        self.cell_tuple.control.append((c, h))
        return self

    def beliefs(self):
        return self.cell_tuple.beliefs

    def control(self):
        return self.cell_tuple.control

    def actions(self):
        return self.cell_tuple.actions

    def hiddens(self):
        all_hidden = list(zip(*self.cell_tuple.control))[1]
        return list(all_hidden)

    def states(self):
        return [DQNACellState(c, h) \
         for c, h in self.cell_tuple.control]

    def get_action(self, index):
        return self.cell_tuple.actions[index]

    def get_belief(self, index):
        return self.cell_tuple.beliefs[index]

    def get_cell(self, index):
        if not self.cell_tuple:
            raise ValueError("previous states not stored. Use `DQNACellState`")

        a = self.actions()[index]
        b = self.beliefs()[index]
        c, h = self.control()[index]

        return DQNACellState(c, h, a, b)