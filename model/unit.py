import tensorflow as tf
from model.networks.rnn_unit import *
from model.networks.mlp_unit import *
from util.sparse_util import *

def get_model(name):
    name = name.lower()
    if name == 'mlp':
        return MLPModel

    elif name == 'rnn':
        return RNNModel

    elif name == 'cnn':
        raise NotImplementedError

class Unit(object):
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
        self.Snip['Dummy_Roll'] = self.model.get_roll_variables()

        self.Tensor['Unit_Grad'] = [None for _ in self.Snip['Dummy_Kernel']]
        self.Info['Type'] = self.model.Network['Type']
        self.Info['Params'] = self.model.Network['Params']

    def build_sparse(self, sparse_list, ii, use_dense=False):
        with tf.variable_scope(self.scope+'/'):
            self.model.build_sparse(sparse_list, ii, use_dense)
            self.initialize_op = self.model.initialize_op

    def run(self, features, use_last=True):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Predictions'] = self.model(features, use_last)

            if 'Weights' in self.model.Tensor:
                self.Tensor['Weights'] = self.model.Tensor['Weights']

            return self.Tensor['Predictions']


    def unit(self, minibatch, loss_fnc, ii, use_last=True, loss_params={}):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Unit_Pred'] = self.model.unit(
                minibatch['Features'], ii, use_last
            )

            self.Tensor['Unit_Loss'] = tf.reduce_mean(loss_fnc(
                self.Tensor['Unit_Pred'], minibatch['Labels'],
                **loss_params
            ))

            # one layer case
            self.Tensor['Unit_Grad'][ii] = tf.gradients(
                self.Tensor['Unit_Loss'], self.Snip['Dummy_Kernel'][ii])

            if self.Info['Type'][ii] != 'embedding':
                self.Tensor['Unit_Grad'][ii] *= self.Snip['Dummy_Kernel'][ii]

            return self.Tensor['Unit_Grad'][ii]