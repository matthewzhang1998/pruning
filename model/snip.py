import tensorflow as tf
from model.networks.rnn_snip import *
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

class Snip(object):
    def __init__(self, scope, params, input_size, num_classes, seed=1, init=None):
        self.params = params
        self.Snip = {}
        self.scope = scope
        with tf.variable_scope(scope+'/'):
            self.model = get_model(params.model_type)(
                params, input_size, num_classes, seed, init
            )

        self.Tensor = {}
        self.Info = {}
        self.Op = {}

        self.Snip['Dummy_Kernel'] = self.model.get_dummy_variables()
        self.Info['Type'] = self.model.Network['Type']
        self.Info['Params'] = self.model.Network['Params']

    def build_sparse(self, sparse_list, use_dense=False):
        with tf.variable_scope(self.scope+'/'):
            self.model.build_sparse(sparse_list, use_dense)
            self.initialize_op = self.model.initialize_op

    def run(self, features):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Predictions'] = self.model(features)
            return self.Tensor['Predictions']

    def snip(self, minibatch, loss_fnc):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Snip_Pred'] = self.model.snip(
                minibatch['Features']
            )

            self.Tensor['Loss'] = tf.reduce_mean(loss_fnc(
                self.Tensor['Snip_Pred'], minibatch['Labels']
            ))

            # one layer case
            self.Tensor['Grad'] = tf.gradients(
                self.Tensor['Loss'], self.Snip['Dummy_Kernel'])

            self.Tensor['Snip_Grad'] = []
            for ix, grad in enumerate(self.Tensor['Grad']):
                self.Tensor['Snip_Grad'].append(grad * self.Snip['Dummy_Kernel'][ix])

            return self.Tensor['Snip_Grad']