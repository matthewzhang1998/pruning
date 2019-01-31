import tensorflow as tf
import numpy as np
from util import norm_util
from util.rnn_util import *

'''
THIS CODE IS MOSTLY DEPRECATED AND ONLY USED BY THE ORIGINAL MNIST EXPERIMENT
PLEASE IGNORE
'''

def get_activation_func(activation_type):
    if activation_type == 'leaky_relu':
        activation_func = tf.nn.leaky_relu
    elif activation_type == 'tanh':
        activation_func = tf.nn.tanh
    elif activation_type == 'relu':
        activation_func = tf.nn.relu
    elif activation_type == 'elu':
        activation_func = tf.nn.elu
    elif activation_type == 'none':
        activation_func = tf.identity
    elif activation_type == 'sigmoid':
        activation_func = tf.sigmoid
    else:
        raise ValueError(
            "Unsupported activation type: {}!".format(activation_type)
        )
    return activation_func

def get_rnn_cell(rnn_cell_type):
    cell_args = {}
    if rnn_cell_type == 'basic':
        cell_type = tf.contrib.rnn.BasicRNNCell
        shape = 2
    elif rnn_cell_type == 'lstm':
        cell_type = tf.contrib.rnn.BasicLSTMCell
        cell_args['state_is_tuple'] = False
        shape = 4
    elif rnn_cell_type == 'gru':
        cell_type = tf.contrib.rnn.GRUCell
        shape = 3
    elif rnn_cell_type == 'norm_lstm':
        cell_type = tf.contrib.rnn.LayerNormBasicLSTMCell
        shape = 4
    elif rnn_cell_type == 'mask_basic':
        cell_type = BasicMaskRNNCell
        shape = 2
    elif rnn_cell_type == 'mask_lstm':
        cell_type = BasicMaskLSTMCell
        cell_args['state_is_tuple'] = False
        shape = 4
    else:
        raise ValueError("Unsupported cell type: {}".format(rnn_cell_type))
    return cell_type, cell_args, shape

def get_unitwise_rnn_cell(rnn_cell_type):
    cell_args = {}
    if rnn_cell_type == 'mask_basic':
        pass
        #cell_type = BasicUnitRNNCell
        shape = 2
    elif rnn_cell_type == 'mask_lstm':
        cell_type = BasicUnitLSTMCell
        cell_args['state_is_tuple'] = False
        shape = 4
    else:
        raise ValueError("Unsupported cell type: {}".format(rnn_cell_type))
    return cell_type, cell_args, shape

def get_normalizer(normalizer_type, train=True):
    if normalizer_type == 'batch':
        normalizer = norm_util.batch_norm_with_train if train else \
            norm_util.batch_norm_without_train

    elif normalizer_type == 'layer':
        normalizer = norm_util.layer_norm

    elif normalizer_type == 'none':
        normalizer = tf.identity

    else:
        raise ValueError(
            "Unsupported normalizer type: {}!".format(normalizer_type)
        )
    return normalizer


def normc_initializer(shape, seed=1234, stddev=1.0, dtype=tf.float32):
    npr = np.random.RandomState(seed)
    out = npr.randn(*shape).astype(np.float32)
    out *= stddev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)

def bradly_initializer(shape, alpha=0.1, seed=1234, dtype=tf.float32):
    npr = np.random.RandomState(seed)
    stddev = np.sqrt(2/(shape[0]+shape[1]))
    out = npr.normal(loc=alpha, scale=np.sqrt(stddev**2 - alpha**2), size=shape)\
        * (2*npr.binomial(1, 0.5, shape) - 1)
    print(out)
    return tf.constant(out, dtype=dtype)

def init(shape=None, name=None, init_method=None, dtype=tf.float32,
        init_para=None, seed=1234, trainable=True):
    """ @brief:
            Initialize weights

        @input:
            shape: list of int, shape of the weights
            init_method: string, indicates initialization method
            init_para: a dictionary,
            init_val: if it is not None, it should be a tensor

        @output:
            var: a TensorFlow Variable
    """

    if init_method is None or init_method == 'zero':
        initializer = tf.zeros_initializer(dtype=dtype)

    if init_method == "normc":
        var = normc_initializer(
            shape, stddev=init_para['stddev'],
            seed=seed, dtype=dtype
        )
        return tf.get_variable(initializer=var, name=name, trainable=trainable)

    elif init_method == "normal":
        initializer = tf.random_normal_initializer(
            mean=init_para["mean"], stddev=init_para["stddev"],
            seed=seed, dtype=dtype
        )

    elif init_method == "truncated_normal":
        initializer = tf.truncated_normal_initializer(
            mean=init_para["mean"], stddev=init_para["stddev"],
            seed=seed, dtype=dtype
        )

    elif init_method == "uniform":
        initializer = tf.random_uniform_initializer(
            minval=init_para["minval"], maxval=init_para["maxval"],
            seed=seed, dtype=dtype
        )

    elif init_method == "constant":
        initializer = tf.constant_initializer(
            value=init_para["val"], dtype=dtype
        )

    elif init_method == "xavier":
        initializer = tf.contrib.layers.xavier_initializer(
            uniform=init_para['uniform'], seed=seed, dtype=dtype
        )

    elif init_method == 'orthogonal':
        initializer = tf.orthogonal_initializer(
            gain=1.0, seed=seed, dtype=dtype
        )

    elif init_method == 'variance_scaling':
        initializer = tf.contrib.layers.variance_scaling_initializer(
            factor=init_para['factor'], uniform=init_para['uniform'],
            seed=seed, dtype=dtype
        )

    elif init_method == 'bradly':
        var = bradly_initializer(
            shape, alpha=init_para['alpha'],
            seed=seed, dtype=dtype
        )
        return lambda x: var

    else:
        raise ValueError("Unsupported initialization method {}".format(init_method))

    return initializer

def weight_variable(shape, name='w', init_method=None, dtype=tf.float32,
                    init_para=None, seed=1234, trainable=True):

    initializer = init(shape, name, init_method, dtype, init_para, seed, trainable)
    var = tf.get_variable(initializer=initializer(shape),
                          name=name, trainable=trainable)

    return var


class Linear(object):
    """
    Performs a simple weight multiplication without activations
    """

    def __init__(self, dims, scope, train,
                 normalizer_type, init_data,
                 dtype=tf.float32, reuse=None):

        self._scope = scope
        self.num_layer = len(dims) - 1  # the last one is the input dim
        self._w = [None] * self.num_layer
        self._b = [None] * self.num_layer
        self._train = train
        self._reuse = reuse

        self._normalizer_type = normalizer_type
        self._init_data = init_data
        self._last_dim = dims[-1]

        # initialize variables
        with tf.variable_scope(scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in, dim_out = dims[ii], dims[ii + 1]

                    self._w[ii] = weight_variable(
                        shape=[dim_in, dim_out], name='weights',
                        init_method=self._init_data[ii]['w_init_method'],
                        init_para=self._init_data[ii]['w_init_para'],
                        dtype=dtype, trainable=self._train
                    )

                    self._b[ii] = weight_variable(
                        shape=[dim_out], name='bias',
                        init_method=self._init_data[ii]['b_init_method'],
                        init_para=self._init_data[ii]['b_init_para'],
                        dtype=dtype, trainable=self._train
                    )

    def __call__(self, input_vec):
        self._h = [None] * self.num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
                                tf.concat([[-1], [output_shape[-1]]], axis=0)
                                )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    if ii == 0:
                        self._h[ii] = tf.matmul(flat_input, self._w[ii]) \
                                      + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._w[ii]) \
                            + self._b[ii]

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
                          tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
                          )

    def weights(self):
        weights = []
        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    weights.append(self._w[ii])
        return weights

class Recurrent_Network(object):
    """
    borrows tensorflow api for recurrent networks
    """

    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, seed=12345, dtype=tf.float32, reuse=None):
        self._scope = scope
        _cell_proto, _cell_kwargs = get_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size
        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs)

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

    def weights(self):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            pass

    def set_weights(self, mask):
        self._cell.set_weights()

class Recurrent_Network_with_mask(Recurrent_Network):
    """
    borrows tensorflow api for recurrent networks
    """
    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, input_depth, seed=12345,
                 dtype=tf.float32, reuse=None, init_data=None):
        self._scope = scope
        _cell_proto, _cell_kwargs, cell_shape = get_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size
        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs, input_depth=input_depth,
                init_data=init_data, seed=seed)

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

    def set_weights(self, mask):
        self._cell.set_weights()

    def weights(self):
        return [self._cell._kernel]

    def get_mask(self):
        return [self._cell._mask]

    def get_weighted_mask(self):
        return [self._cell._combined]

class Recurrent_Network_Unitwise(Recurrent_Network):
    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, input_depth, num_unitwise=None, seed=12345,
                 dtype=tf.float32, reuse=None, init_data=None):
        print(num_unitwise)
        self._scope = scope
        self._use_lstm = True if 'lstm' in recurrent_cell_type else False
        _cell_proto, _cell_kwargs, cell_shape = get_unitwise_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size

        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs, input_depth=input_depth,
                init_data=init_data, seed=seed, num_unitwise=num_unitwise)
    def __call__(self, input_tensor, hidden_states=None):
        print(input_tensor, hidden_states)
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

    def unitwise(self, input_tensor, hidden_states=None, distributions=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):

            input_shape = tf.shape(input_tensor)
            _batch_size = input_shape[0]
            _time_size = input_shape[1]
            _dummy_outputs = tf.zeros(
                [_batch_size, _time_size, 1]
            )

            def _cond(_input, _output_tensor, _i):
                return _i < self._hidden_size

            def _lambd(_input, _output_tensor, _i):
                _rnn_outputs, _ = dynamic_unitwise_rnn(
                    _i, self._cell, input_tensor, initial_state=hidden_states,
                    use_lstm=self._use_lstm, hidden_size=self._hidden_size
                )

                _output_tensor = tf.concat(
                    [_output_tensor, tf.expand_dims(_rnn_outputs[:,:,_i], 2)], axis=2
                )
                _i += 1
                return _input, _output_tensor, _i

            _, _rnn_outputs, _ = tf.while_loop(
                _cond, _lambd,
                [input_tensor, _dummy_outputs, tf.constant(0)],
                shape_invariants=[input_tensor.get_shape(),
                    tf.TensorShape((None, None, None)),
                    tf.TensorShape(())]
            )

            _rnn_outputs = _rnn_outputs[:,:,1:]

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
        return _rnn_outputs, None

    def single_unit(self, input_tensor, hidden_states=None, distributions=None):
        _, _rnn_outputs = dynamic_dummy_rnn(
            self._cell, input_tensor, initial_state=hidden_states,
            use_lstm=self._use_lstm, hidden_size=self._hidden_size
        )

        return _rnn_outputs

    def dummy_weights(self):
        return [self._cell._dummy_kernel, self._cell._dummy_bias]

    def biases(self):
        return [self._cell._bias]

    def weights(self):
        return [self._cell._kernel]

    def get_mask(self):
        return [self._cell._mask]

    def get_weighted_mask(self):
        return [self._cell._combined]

class Bidirectional_Recurrent_Network(object):
    """
    borrows tensorflow api for recurrent networks
    """

    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, dtype=tf.float32, reuse=None):
        self._scope = scope
        _cell_proto, _cell_kwargs = get_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size
        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs)

    def __call__(self, input_tensor, fw_states=None, bw_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.bidirectional_dynamic_rnn(
                self._cell, input_tensor, initial_state_fw=fw_states,
                initial_state_bw=bw_states, dtype=tf.float32
            )

            _fw, _bw = _rnn_outputs

            if self._activation_type[0] is not None:
                act_func = \
                    get_activation_func(self._activation_type[0])
                _fw = \
                    act_func(_fw, name='activation_0')
                _bw = \
                    act_func(_bw, name='activation_0')

            if self._normalization_type is not None:
                normalizer = get_normalizer(self._normalization_type[0],
                                            train=self._train)
                _fw = \
                    normalizer(_fw, 'normalizer_0')
                _bw = \
                    normalizer(_bw, 'normalizer_0')

        return (_fw, _bw), _rnn_states


class Dilated_Recurrent_Network(object):
    """
    borrows tensorflow api for recurrent networks
    """

    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, dtype=tf.float32,
                 dilation=1, reuse=None):
        self._scope = scope
        _cell_proto, _cell_kwargs = get_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._dilation = dilation
        self._hidden_size = hidden_size
        self._reuse = reuse
        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs)

    def __call__(self, input_tensor, hidden_states):

        _hidden_states = tf.expand_dims(hidden_states, 1)
        _dilated_hidden_states = tf.tile(
            _hidden_states, [1, self._dilation, 1]
        )
        _batch_size = tf.shape(input_tensor)[0]
        _dummy_outputs = tf.zeros(
            [_batch_size, 1, self._hidden_size]
        )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            # construct dilated rnn
            def _dilated_rnn(_input, _state, _output_tensor, _i):
                _output, _next_state = self._cell(_input[:, _i],
                                                  _state[:, _i % self._dilation])

                _state = tf.concat(
                    [_state, tf.expand_dims(_next_state, 1)], axis=1
                )
                _output_tensor = tf.concat(
                    [_output_tensor, tf.expand_dims(_output, 1)], axis=1
                )
                _i += 1

                return _input, _state, _output_tensor, _i

            def _condition(_input, _state, _output_tensor, _i):
                return _i < tf.shape(_input)[1]

            _, _rnn_state_arr, _rnn_output_arr, _ = tf.while_loop(
                _condition, _dilated_rnn,
                [input_tensor, _dilated_hidden_states,
                 _dummy_outputs, tf.constant(0)],
                shape_invariants=[input_tensor.get_shape(),
                                  tf.TensorShape((None, None, self._hidden_size)),
                                  tf.TensorShape((None, None, self._hidden_size)),
                                  tf.TensorShape(())]
            )

            _rnn_states = _rnn_state_arr[:, -1]
            _rnn_outputs = _rnn_output_arr[:, 1:]

            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor, initial_state=hidden_states
            )
            if self._activation_type[0] is not None:
                act_func = \
                    get_activation_func(self._activation_type[0])
                _rnn_outputs = \
                    act_func(_rnn_outputs, name='activation_0')

            if self._normalization_type is not None:
                normalizer = get_normalizer(self._normalization_type[0],
                                            train=self._train)
                _rnn_outputs = \
                    normalizer(_rnn_outputs, 'normalizer_0')
        return _rnn_outputs, _rnn_states


class MLP(object):
    """ Multi Layer Perceptron (MLP)
                Note: the number of layers is N

        Input:
                dims: a list of N+1 int, number of hidden units (last one is the
                output dimension)
                act_func: a list of N activation functions
                add_bias: a boolean, indicates whether adding bias or not
                scope: tf scope of the model

    """

    def __init__(self, dims, scope, train,
                 activation_type, normalizer_type, init_data, seed=1,
                 dtype=tf.float32, reuse=None):

        self._scope = scope
        self.num_layer = len(dims) - 1  # the last one is the output dim
        self.dims = dims
        self._w = [None] * self.num_layer
        self._b = [None] * self.num_layer
        self._train = train
        self._last_dim = dims[-1]
        self._reuse = reuse

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self._init_data = init_data

        # initialize variables
        with tf.variable_scope(scope, reuse=reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii)):
                    dim_in, dim_out = dims[ii], dims[ii + 1]

                    self._w[ii] = weight_variable(
                        shape=[dim_in, dim_out], name='permanent_weights',
                        init_method=self._init_data[ii]['w_init_method'],
                        init_para=self._init_data[ii]['w_init_para'],
                        dtype=dtype, trainable=self._train, seed=seed
                    )

                    self._b[ii] = weight_variable(
                        shape=[dim_out], name='bias',
                        init_method=self._init_data[ii]['b_init_method'],
                        init_para=self._init_data[ii]['b_init_para'],
                        dtype=dtype, trainable=self._train,
                        seed=seed
                    )

    def __call__(self, input_vec):
        self._h = [None] * self.num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
                                tf.concat([[-1], [output_shape[-1]]], axis=0)
                                )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):
                    if ii == 0:
                        self._h[ii] = tf.matmul(flat_input, self._w[ii]) \
                                      + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._w[ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
            tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
        )

    def get_mask(self):
        return self.mask

    def get_weighted_mask(self):
        return self._p

    def weights(self):
        weights = [self._w[ii] for ii in range(self.num_layer-1)]
        return weights

class MLPWithMask(MLP):
    """ Multi Layer Perceptron (MLP)
                Note: the number of layers is N

        Input:
                dims: a list of N+1 int, number of hidden units (last one is the
                output dimension)
                act_func: a list of N activation functions
                add_bias: a boolean, indicates whether adding bias or not
                scope: tf scope of the model

    """

    def __init__(self, *args, **kwargs):
        super(MLPWithMask, self).__init__(*args, **kwargs)
        self._p = []
        self.mask = []
        for ii in range(self.num_layer):
            self.mask.append(
                tf.placeholder(dtype=tf.float32,
                shape=self._w[ii].get_shape())
            )
            self._p.append(self._w[ii] * self.mask[ii])

    def __call__(self, input_vec):
        self._h = [None] * self.num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):

                    if ii == 0:

                        self._h[ii] = tf.matmul(flat_input, self._p[ii]) \
                            + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._p[ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])

                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))


                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
            tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
        )

    def weights(self):
        weights = [self._w[ii] for ii in range(self.num_layer)]
        return weights

    def get_mask(self):
        return self.mask

    def get_weighted_mask(self):
        return self._p

class MLPLayerWise(MLP):
    """ Multi Layer Perceptron (MLP)
                Note: the number of layers is N

        Input:
                dims: a list of N+1 int, number of hidden units (last one is the
                output dimension)
                act_func: a list of N activation functions
                add_bias: a boolean, indicates whether adding bias or not
                scope: tf scope of the model

    """

    def __init__(self, *args, **kwargs):
        super(MLPLayerWise, self).__init__(*args, **kwargs)
        self._p = []
        self.mask = []
        self.proxy_input = []
        self.proxy_output = []
        for ii in range(self.num_layer):
            self.mask.append(
                tf.placeholder(dtype=tf.float32,
                shape=self._w[ii].get_shape(),
                name=self._scope + 'mask' + str(ii))
            )
            self.proxy_input.append(
                tf.placeholder(dtype=tf.float32,
                shape=[None, self._w[ii].get_shape()[0]],
                name=self._scope + 'proxy' + str(ii))
            )
            self.proxy_output.append(
                tf.placeholder(dtype=tf.float32,
                shape=[None, self._w[ii].get_shape()[1]],
                name=self._scope + 'pred' + str(ii))
            )
            self._p.append(self._w[ii] * self.mask[ii])

    def __call__(self, input_vec):
        self._h = [None] * self.num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
                                tf.concat([[-1], [output_shape[-1]]], axis=0)
                                )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):
                    if ii == 0:
                        self._h[ii] = tf.matmul(flat_input, self._p[ii]) \
                                      + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._p[ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
            tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
        )

    def run_backward(self, i):
        h = [None] * self.num_layer

        output_shape = tf.shape(self.proxy_input[i])
        flat_input = tf.reshape(self.proxy_input[i],
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            with tf.variable_scope("layer_{}".format(i), reuse=tf.AUTO_REUSE):
                h[i] = tf.matmul(flat_input, self._p[i]) + self._b[i]

                if self._activation_type[i] is not None:
                    act_func = \
                        get_activation_func(self._activation_type[i])
                    h[i] = \
                        act_func(h[i], name='activation_' + str(i))

                if self._normalizer_type[i] is not None:
                    normalizer = get_normalizer(self._normalizer_type[i],
                                                train=self._train)
                    h[i] = \
                        normalizer(h[i], 'normalizer_' + str(i))

            for ii in range(i+1, self.num_layer):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):
                    if ii == 0:
                        h[ii] = tf.matmul(flat_input, self._p[ii]) \
                                      + self._b[ii]

                    else:
                        h[ii] = \
                            tf.matmul(h[ii - 1], self._p[ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        h[ii] = \
                            act_func(h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        h[ii] = \
                            normalizer(h[ii], 'normalizer_' + str(ii))

        return h[i], tf.reshape(h[-1],
            tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
        )

    def weights(self):
        weights = [self._w[ii] for ii in range(self.num_layer)]
        return weights

    def get_mask(self):
        return self.mask

    def get_weighted_mask(self):
        return self._p

    def get_proxy_input(self):
        return self.proxy_input

    def get_proxy_output(self):
        return self.proxy_output


class MLPCopy(object):
    """ Multi Layer Perceptron (MLP)
                Note: the number of layers is N

        Input:
                dims: a list of N+1 int, number of hidden units (last one is the
                output dimension)
                act_func: a list of N activation functions
                add_bias: a boolean, indicates whether adding bias or not
                scope: tf scope of the model

    """

    def __init__(self, net, scope):
        self._scope = scope
        self.num_layer = net.num_layer
        self._w = [tf.Variable(w.initialized_value()) for w in net._w]
        self._b = [tf.Variable(b.initialized_value()) for b in net._b]
        self._train = net._train
        self.dims= net.dims
        self._last_dim = net._last_dim
        self._reuse = net._reuse

        self._activation_type = net._activation_type
        self._normalizer_type = net._normalizer_type
        self._init_data = net._init_data

        self._p = []
        self.mask = []
        try:
            self.proxy_input = net.proxy_input
            self.proxy_output = net.proxy_output
        except:
            self.proxy_output = self.proxy_input = None

        for ii in range(self.num_layer):
            self.mask.append(
                tf.placeholder(dtype=tf.float32,
                    shape=self._w[ii].get_shape(),
                    name = self._scope+'mask'+str(ii))
            )
            self._p.append(self._w[ii] * self.mask[ii])

    def __call__(self, input_vec):
        self._h = [None] * self.num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):
                    if ii == 0:
                        self._h[ii] = tf.matmul(flat_input, self._p[ii]) \
                        + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._p[ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
            tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
        )

    def run_forward(self, i, x):
        h = [None] * self.num_layer

        output_shape = tf.shape(x[i])
        flat_input = tf.reshape(x[i],
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )
        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(i+1):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):
                    if ii == 0:
                        h[ii] = tf.matmul(flat_input, self._p[ii]) \
                                      + self._b[ii]
                    else:
                        h[ii] = \
                            tf.matmul(h[ii - 1], self._p[ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        h[ii] = \
                            act_func(h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        h[ii] = \
                            normalizer(h[ii], 'normalizer_' + str(ii))

            curr = tf.matmul(self.proxy_input[i], self._p[i]) \
                + self._b[i]

            if self._activation_type[i] is not None:
                act_func = \
                    get_activation_func(self._activation_type[i])
                curr = \
                    act_func(curr, name='activation_' + str(i))

            if self._normalizer_type[i] is not None:
                normalizer = get_normalizer(self._normalizer_type[i],
                                            train=self._train)
                curr = \
                    normalizer(curr, 'normalizer_' + str(i))

        return curr, h[i]

    def weights(self):
        weights = [self._w[ii] for ii in range(self.num_layer)]
        return weights

    def get_mask(self):
        return self.mask

    def get_weighted_mask(self):
        return self._p

    def get_proxy_input(self):
        return self.proxy_input

    def get_proxy_output(self):
        return self.proxy_output

class MLPTrainMask(MLP):
    def __init__(self, *args, **kwargs):
        super(MLPTrainMask, self).__init__(*args, **kwargs)
        self._p = []
        self.mask = []

        self._npr = np.random.RandomState(seed = 15694)
        self.train_mask = []

        for ii in range(self.num_layer):
            self.mask.append(
                tf.placeholder(dtype=tf.float32,
                shape=self._w[ii].get_shape())
            )
            self.train_mask.append(
                tf.sigmoid(tf.Variable(
                    self._npr.normal(-3, 5,
                        self._w[ii].get_shape().as_list()),
                    dtype=tf.float32,
                )))
            self._p.append(self._w[ii] * self.mask[ii])
        self._v = [w * mask for (w, mask) in zip(self._w, self.train_mask)]

    def __call__(self, input_vec):
        self._h = [None] * self.num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
                                tf.concat([[-1], [output_shape[-1]]], axis=0)
                                )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):
                    if ii == 0:
                        self._h[ii] = tf.matmul(flat_input, self._p[ii]) \
                                      + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._p[ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
            tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
        )

    def mask_train(self, input_vec):
        self._h = [None] * self.num_layer

        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope, reuse=self._reuse):
            for ii in range(self.num_layer):
                with tf.variable_scope("layer_{}".format(ii),
                                       reuse=tf.AUTO_REUSE):
                    if ii == 0:
                        self._h[ii] = tf.matmul(flat_input, self._v[ii]) \
                                      + self._b[ii]

                    else:
                        self._h[ii] = \
                            tf.matmul(self._h[ii - 1], self._v[ii]) \
                            + self._b[ii]

                    if self._activation_type[ii] is not None:
                        act_func = \
                            get_activation_func(self._activation_type[ii])
                        self._h[ii] = \
                            act_func(self._h[ii], name='activation_' + str(ii))

                    if self._normalizer_type[ii] is not None:
                        normalizer = get_normalizer(self._normalizer_type[ii],
                                                    train=self._train)
                        self._h[ii] = \
                            normalizer(self._h[ii], 'normalizer_' + str(ii))

        return tf.reshape(self._h[-1],
                          tf.concat([output_shape[:-1], [self._last_dim]], axis=0)
                          )

    def weights(self):
        return self._w

    def get_mask_train(self):
        return self.train_mask

    def get_weighted_mask_train(self):
        return self._v

    def get_mask(self):
        return self.mask

    def get_weighted_mask(self):
        return self._p
