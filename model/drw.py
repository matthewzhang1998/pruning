import tensorflow as tf
from model.networks.rnn_drw import *
from util.network_util import *
from util.sparse_util import *

def get_model(name):
    name = name.lower()
    if name == 'mlp':
        raise NotImplementedError

    elif name == 'rnn':
        return RNNModel

    elif name == 'cnn':
        raise NotImplementedError

class DRW(object):
    def __init__(self, scope, params, input_size, num_classes, seed=1, init_path=None):
        self.params = params
        self.Snip = {}
        self.scope = scope
        with tf.variable_scope(scope+'/'):
            self.model = get_model(params.model_type)(
                params, input_size, num_classes, seed, init_path
            )

        self.Tensor = {}
        self.Info = {}
        self.Op = {}

        self.Snip['Mask'] = self.model.get_mask()
        self.Snip['Theta'] = self.model.get_theta()

    def run(self, features):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Predictions'] = self.model(features)
            return self.Tensor['Predictions']