import tensorflow as tf
from util.random_util import *
from model.base import *

class Random(Pruner):
    def __init__(self, scope, params, input_size, num_classes):
        super(Random, self).__init__(scope, params, input_size, num_classes)

        self.Tensor['Mask'] = {}
        self.Bernoulli = Binary_Bernoulli(self.params.random_k)

        with tf.variable_scope(scope):
            for weight in self.weight_params:
                self.Tensor['Mask'][weight.name] = \
                    self.Bernoulli(tf.zeros_like(weight))

        self.Op['Sparse'] = \
            self.model.set_weighted_params(self.Tensor['Mask'], 'Sparse')

    def run(self, features):
        self.Tensor['predictions'] = self.model.weighted(features, 'Sparse')
        return self.Tensor['predictions']