import tensorflow as tf
import numpy as np
from model.networks.mlp_regress import *
from util.sparse_util import *

class Regress(object):
    def __init__(self, scope, params, input_size, num_classes, seed=1, init=None):
        self.params = params
        self.Snip = {}
        self.scope = scope
        with tf.variable_scope(scope+'/'):
            self.model = MLPModel(params, input_size, num_classes, seed, init)

        self.Tensor = {}
        self.Info = {}
        self.Op = {}

        self.Info['Type'] = self.model.Network['Type']
        self.Info['Params'] = self.model.Network['Params']

        self.fork_model = [MLPModel(params, input_size, num_classes, seed, init)
            for _ in range(self.params.num_gpu)]

    def run(self, features, use_last=True):
        with tf.variable_scope(self.scope+'/'):
            self.Tensor['Predictions'] = self.model(features)

            return self.Tensor['Predictions']

    def fork_model_fnc(self, weights, i):
        # assume sparse for now
        self.fork_model[i].init_model(scope='reg{}'.format(i), weights=weights)

    def hessian_variables(self, i):
        return self.fork_model[i].hessian_variable()

    def run_fork(self, feature, i, return_hidden=False):
        output = self.fork_model[i](feature, return_hidden=return_hidden)
        return output