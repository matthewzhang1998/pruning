import tensorflow as tf
from model.networks.rnn_vanilla import *
from util.sparse_util import *

def get_model(name):
    name = name.lower()
    if name == 'mlp':
        raise NotImplementedError

    elif name == 'rnn':
        return RNNModel

    elif name == 'cnn':
        raise NotImplementedError

class Vanilla(object):
    def __init__(self, scope, params, input_size, num_classes, seed=1, init_path=None):
        self.params = params
        self.Snip = {}
        self.scope = scope
        with tf.variable_scope(scope+'/'):
            self.model = get_model(params.model_type)(
                params, input_size, num_classes, seed, init_path=init_path
            )

        self.Tensor = {}
        self.Info = {}
        self.Op = {}

    def run(self, features, use_last=True):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Predictions'] = self.model(features, use_last)
            return self.Tensor['Predictions']