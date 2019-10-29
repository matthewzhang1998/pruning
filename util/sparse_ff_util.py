from util.sparse_util import *
import tensorflow as tf
import numpy as np

class SparseDummyFullyConnected(object):
    def __init__(self, input_depth, hidden_size, scope,
                 activation_type, normalizer_type, seed=1,
                 num_unitwise=None, train=True, use_bias=True, num_shards=1,
                 **kwargs):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size
        self._num_unitwise = num_unitwise
        self._train = train

        self.use_bias = use_bias
        self.seed = seed

        self._dummy_kernel = tf.placeholder(
            shape=[input_depth, self._num_unitwise], dtype=tf.float32
        )
        self._dummy_bias = tf.zeros(
            shape=[self._num_unitwise], dtype=tf.float32
        )
        self.roll = tf.placeholder_with_default(
            tf.zeros([1], dtype=tf.int32), [1]
        )

        self.sample_inds = tf.sparse_placeholder(dtype=tf.float32)

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self.var = [self._dummy_kernel]
        self.placeholder = (self._dummy_kernel, self.roll, self.sample_inds)

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope):
            res = tf.matmul(flat_input, self._dummy_kernel)

            res_o = sparse_matmul([flat_input], self.sample_inds)

            res = tf.concat([res, res_o], axis=1)
            res = tf.roll(res, self.roll, [-1])

            if self._activation_type is not None:
                act_func = \
                    get_activation_func(self._activation_type)
                res = \
                    act_func(res, name='activation')

            if self._normalizer_type is not None:
                normalizer = get_normalizer(self._normalizer_type,
                    train=self._train)
                res = \
                    normalizer(res, 'normalizer')

        return tf.reshape(res,
            tf.concat([output_shape[:-1], tf.constant([self.hidden_size])], axis=0)
        )

class SparseFullyConnected(object):

    def __init__(self, input_depth, hidden_size, scope,
                 activation_type, normalizer_type, sparse_list,
                 train=True, use_bias=True, num_shards=1, **kwargs):

        self._scope = scope
        self.input_depth = input_depth
        self.use_bias = use_bias
        self.hidden_size = hidden_size

        with tf.variable_scope(self._scope):
            self.weight, var = get_sparse_sharded_weight_matrix([input_depth, hidden_size],
                sparse_list, out_type='sparse', dtype=tf.float32, name='', num_shards=num_shards)
            if use_bias:
                self._b = tf.Variable(tf.zeros(shape=[hidden_size], dtype=tf.float32))
        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type

        self.var = [var]

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope):
            res = sparse_matmul(flat_input, self.weight)
            if self.use_bias:
                res += self._b

            if self._activation_type is not None:
                act_func = \
                    get_activation_func(self._activation_type)
                res = \
                    act_func(res, name='activation')

            if self._normalizer_type is not None:
                normalizer = get_normalizer(self._normalizer_type,
                    train=self._train)
                res = \
                    normalizer(res, 'normalizer')

        return tf.reshape(res,
            tf.concat([output_shape[:-1], tf.constant([self.hidden_size])], axis=0)
        )
