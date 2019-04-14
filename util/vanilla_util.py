import tensorflow as tf

from util.network_util import *

class DenseFullyConnected(object):
    def __init__(self, input_depth, hidden_size, scope,
                 activation_type, normalizer_type, weight,
                 train=True):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size

        with tf.variable_scope(self._scope):
            if weight is not None:
                self.weight = tf.Variable(weight, dtype=tf.float32)
            else:
                self.weight = tf.get_variable('w', [input_depth, hidden_size], dtype=tf.float32)

            self.b = tf.Variable(tf.zeros(shape=[hidden_size], dtype=tf.float32))
        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        # self.initialize_op = tf.initialize_variables([self._b, self.weight])

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [self.input_depth]], axis=0)
        )
        with tf.variable_scope(self._scope):
            res = tf.matmul(flat_input, self.weight) + self.b

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

class DenseEmbedding(object):
    def __init__(self, input_depth, hidden_size, scope,
        weight, seed=1):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size
        self.seed = seed

        with tf.variable_scope(self._scope):
            if weight is not None:
                self.weight = tf.Variable(weight, dtype=tf.float32)
            else:
                self.weight = tf.get_variable("embedding", [input_depth, hidden_size], dtype=tf.float32)

        # self.initialize_op = tf.initialize_variables([self.weight])

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)

        with tf.variable_scope(self._scope):
            res = tf.nn.embedding_lookup(self.weight, input_vec)

        return tf.reshape(res,
            tf.concat([output_shape, tf.constant([self.hidden_size])], axis=0)
        )

class DenseRecurrentNetwork(object):
    """
    borrows tensorflow api for recurrent networks
    """

    def __init__(self, scope, input_depth, activation_type,
                 normalizer_type, recurrent_cell_type, weight,
                 train, hidden_size, seed=12345, dtype=tf.float32, reuse=None):
        self._scope = scope
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size
        with tf.variable_scope(scope):
            self._cell = tf.contrib.rnn.BasicLSTMCell(self._hidden_size)

    def __call__(self, input_tensor, hidden_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor, initial_state=hidden_states,
                dtype=tf.float32
            )
            if self._activation_type is not None:
                act_func = \
                    get_activation_func(self._activation_type)
                _rnn_outputs = \
                    act_func(_rnn_outputs, name='activation_0')

            if self._normalization_type is not None:
                normalizer = get_normalizer(self._normalization_type,
                                            train=self._train)
                _rnn_outputs = \
                    normalizer(_rnn_outputs, 'normalizer_0')
        return _rnn_outputs, _rnn_states