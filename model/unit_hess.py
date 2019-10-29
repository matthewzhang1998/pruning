import tensorflow as tf
import numpy as np
from model.networks.rnn_unit_regress import *
from util.sparse_util import *

class Unit(object):
    def __init__(self, scope, params, input_size, num_classes, seed=1, init=None):
        self.params = params
        self.Snip = {}
        self.scope = scope
        with tf.variable_scope(scope+'/'):
            self.model = RNNModel(params, input_size, num_classes, seed, init)

        self.Tensor = {}
        self.Info = {}
        self.Op = {}

        self.Info['Type'] = self.model.Network['Type']
        self.Info['Params'] = self.model.Network['Params']

        self.fork_model = RNNModel(params, input_size, num_classes, seed, init)

    def run(self, features, use_last=True):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Predictions'] = self.model(features)

            return self.Tensor['Predictions']

    def set_embed_and_softmax(self, embed_mat, softmax_mat, use_dense=True):
        self.Tensor['Cache_Embed'] = embed_mat
        self.Tensor['Cache_Softmax'] = softmax_mat

    def set_variables(self):
        self.fork_model.init_model(scope='reg_unit',
            embed = self.Tensor['Cache_Embed'], softmax = self.Tensor['Cache_Softmax']
        )

    def get_placeholders(self):
        return self.fork_model.get_placeholders()

    def hessian_variables(self):
        return self.fork_model.hessian_variable()

    def run_fork(self, feature, return_rnn=False):
        output = self.fork_model(feature, return_rnn=return_rnn)
        return output

    def run_dummy(self, hidden, states, use_last=False):
        output = self.fork_model.run_dummy(hidden, states, use_last=use_last)
        return output