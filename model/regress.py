import tensorflow as tf
import numpy as np
from model.networks.rnn_regress import *
from util.sparse_util import *

class Regress(object):
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

        self.fork_model = [RNNModel(params, input_size, num_classes, seed, init)
            for _ in range(self.params.num_gpu)]

    def run(self, features, use_last=True):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Predictions'] = self.model(features)

            return self.Tensor['Predictions']

    def set_embed_and_softmax(self, embed_mat, softmax_mat, use_dense=True):
        self.Tensor['Cache_Embed'] = embed_mat
        self.Tensor['Cache_Softmax'] = softmax_mat

    def fork_model_lstm(self, weights, row_ind, col_ind, i, use_dense=False):
        # assume sparse for now

        if use_dense:
            self.fork_model[i].init_model(scope='reg{}'.format(i),
                embed = self.Tensor['Cache_Embed'], softmax = self.Tensor['Cache_Softmax'],
                lstm = weights.T, use_dense=True
            )

        else:
            self.fork_model[i].init_model(scope='reg{}'.format(i),
                embed = self.Tensor['Cache_Embed'], softmax = self.Tensor['Cache_Softmax'],
                lstm = (weights, np.array([row_ind, col_ind]).T))

    def hessian_variables(self, i):
        return self.fork_model[i].hessian_variable()

    def placeholders(self, i):
        return self.fork_model[i].placeholders()

    def run_fork(self, feature, i, return_rnn=False):
        output = self.fork_model[i](feature, return_rnn=return_rnn)
        return output

    def build_full_model(self, lstm_matrix, use_cache_embed=True):
        if use_cache_embed:
            self.model.init_model(embed = self.Tensor['Cache_Embed'],
            softmax = self.Tensor['Cache_Softmax'], lstm = lstm_matrix)

        else:
            pass