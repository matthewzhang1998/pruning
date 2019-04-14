import math

import tensorflow as tf
import numpy as np

from util import norm_util

def split_indices(sparse_list, split, num_split, axis):
    vals, inds = sparse_list
    return_arr = []
    for i in range(num_split):
        slice = np.logical_and(inds[:,axis]<(i+1)*split, inds[:,axis]>i*split-1)
        return_arr.append((vals[slice], inds[slice,:]))

        return_arr[i][1][:,axis] -= i*split

    return return_arr

def get_activation_func(activation_type):
    if activation_type == 'leaky_relu':
        activation_func = tf.nn.leaky_relu
    elif activation_type == 'tanh':
        activation_func = tf.nn.tanh
    elif activation_type == 'relu':
        activation_func = tf.nn.relu
    elif activation_type == 'elu':
        activation_func = tf.nn.elu
    elif activation_type == 'none' or activation_type is None:
        activation_func = tf.identity
    elif activation_type == 'sigmoid':
        activation_func = tf.sigmoid
    else:
        raise ValueError(
            "Unsupported activation type: {}!".format(activation_type)
        )
    return activation_func

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

def get_initializer(shape, init_method, init_para, seed):
    npr = np.random.RandomState(seed)
    if init_method == 'normc':
        out = npr.randn(*shape).astype(np.float32)
        out *= init_para['stddev'] \
            / np.sqrt(np.square(out).sum(axis=0, keepdims=True))

    elif init_method == 'normal':
        out = npr.normal(loc=init_para['mean'], scale=init_para['stddev'],
            size=shape)

    elif init_method == 'xavier':
        if init_para['uniform']:
            out = npr.uniform(low=-np.sqrt(2/(shape[0]+shape[-1])),
                high=np.sqrt(2/(shape[0]+shape[-1])),
                size=shape)
        else:
            out = npr.normal(loc=0, scale=np.sqrt(2/(shape[0]+shape[-1])),
                size=shape)

    return out


def get_random_sparse_matrix(scope, shape, dtype=None, initializer=None, sparsity=0.99, npr=None, seed=None):
    seed = seed or 12345
    npr = npr or np.random.RandomState(seed)
    k_ix = int(np.prod(shape)*(1-sparsity))
    sparse_values = initializer((k_ix,))
    sparse_values = tf.Variable(sparse_values, dtype=dtype)

    sparse_indices = []
    for dim in shape:
        sparse_indices.append(npr.randint(dim, size=k_ix))

    sparse_indices = np.array(sparse_indices).T
    with tf.variable_scope(scope):
        sparse_matrix = tf.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape=shape)
    return sparse_matrix

def get_tensor(sparse_values):
    trainable_var = tf.Variable(sparse_values),
    return

def get_sparse_weight_matrix(shape, sparse_list, out_type='sparse', dtype=tf.float32, name=''):
    # Directly Initialize the sparse matrix
    values, indices = sparse_list
    sparse_values = tf.Variable(values, dtype=dtype)

    #if np.max(shape)<2**16:
        #dtype=tf.int16
    if np.max(shape)<2**32:
        dtype=tf.int32
    else:
        dtype=tf.int64

    sparse_indices = tf.convert_to_tensor(
        indices, name="indices", dtype=dtype
    )

    if out_type == "sparse":
        sparse_weight_matrix = CustomSparseShardedTensor(indices=sparse_indices,
            values=sparse_values, dense_shape=shape, dtype=dtype)
    elif out_type == "dense":
        sparse_weight_matrix = tf.sparse_to_dense(sparse_indices=sparse_indices, output_shape=shape,
                                                  sparse_values=sparse_values, validate_indices=False)
    else:
        raise ValueError("Unknown output type {}".format(out_type))

    return sparse_weight_matrix, sparse_values


def get_dense_weight_matrix(shape, sparse_list, dtype=tf.float32, name=''):
    # Directly Initialize the sparse matrix
    sparse_values, sparse_indices = sparse_list
    sparse_indices = sparse_indices.astype(np.int32)
    dense_arr = np.zeros(shape)
    dense_arr[sparse_indices[:,0], sparse_indices[:,1]] = sparse_values
    return tf.Variable(dense_arr, dtype=dtype)

def sparse_matmul(args, sparse_matrix, scope=None, use_sparse_mul=True):
    if not isinstance(args, list):
        args = [args]

    # Now the computation.

    if len(args) == 1:
        # res = math_ops.matmul(args[0], sparse_matrix,b_is_sparse=True)
        input = args[0]
    else:
        input = tf.concat(args, 1, )
    output_shape = tf.shape(input)
    input = tf.reshape(input,
        tf.concat([[-1], [output_shape[-1]]], axis=0)
    )
    input = tf.transpose(input, perm=[1,0])

    with tf.variable_scope(scope or "Linear"):
        if use_sparse_mul:
            res = tf.sparse_tensor_dense_matmul(sparse_matrix, input, adjoint_a = True)
            res = tf.transpose(res, perm=[1, 0])
        else:
            sparse_matrix = tf.transpose(sparse_matrix, perm=[1, 0])
            if len(args) == 1:
                input = tf.transpose(args[0], perm=[1, 0])
                res = tf.matmul(sparse_matrix, input)
            else:
                # res = math_ops.matmul(array_ops.concat(args, 1, ), sparse_matrix, b_is_sparse=True)
                input = tf.transpose(tf.concat(args, 1, ), perm=[1, 0])
                res = tf.matmul(sparse_matrix, input)
            res = tf.transpose(res, perm=[1, 0])

    res = tf.reshape(res,
        tf.concat([output_shape[:-1], [-1]], axis=0)
    )
    return res

def get_dummy_rnn_cell(rnn_cell_type):
    cell_args = {}
    if rnn_cell_type == 'basic':
        raise NotImplementedError
    elif rnn_cell_type == 'lstm':
        cell_type = SparseDummyLSTMCell
        cell_args['state_is_tuple'] = False

    return cell_type, cell_args

def get_sparse_rnn_cell(rnn_cell_type):
    cell_args = {}
    if rnn_cell_type == 'basic':
        raise NotImplementedError
    elif rnn_cell_type == 'lstm':
        cell_type = SparseLSTMCell
        cell_args['state_is_tuple'] = False

    return cell_type, cell_args

class SparseDummyLSTMCell(object):
    def __init__(self, num_units, input_depth, seed=None,
                 init_data=None, num_unitwise=None, state_is_tuple=False,
                 forget_bias=1.0, activation=None, dtype=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._seed = seed
        self.dtype = dtype
        if activation:
            self._activation = get_activation_func(activation)
        else:
            self._activation = tf.tanh

        self._input_size = input_depth
        self._num_unitwise = num_unitwise if num_unitwise is not None else 1

        self._dummy_kernel = tf.placeholder(
            shape=[input_depth + self._num_units, 4 * self._num_unitwise], dtype=tf.float32
        )
        self._dummy_bias = tf.zeros(
            shape=[4 * self._num_unitwise], dtype=tf.float32
        )
        self.roll = tf.placeholder_with_default(
            tf.zeros([1], dtype=tf.int32), [1]
        )

        self.output_size = num_units
        self.state_size = 2*num_units

    def zero_state(self, batch_size, dtype):
        return tf.zeros(
            tf.stack([batch_size, tf.constant(self.state_size)]),
            dtype=dtype)

    def __call__(self, inputs, state):
        num_proj = self._num_units

        c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
        m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i = tf.matmul(
            tf.concat([inputs, m_prev], 1),
            self._dummy_kernel[:, :self._num_unitwise]) \
            + self._dummy_bias[:self._num_unitwise]

        j = tf.matmul(
            tf.concat([inputs, m_prev], 1),
            self._dummy_kernel[:, self._num_unitwise:2 * self._num_unitwise]) \
            + self._dummy_bias[self._num_unitwise:2 * self._num_unitwise]

        f = tf.matmul(
            tf.concat([inputs, m_prev], 1),
            self._dummy_kernel[:, 2 * self._num_unitwise:3 * self._num_unitwise]) \
            + self._dummy_bias[2 * self._num_unitwise:3 * self._num_unitwise]

        o = tf.matmul(
            tf.concat([inputs, m_prev], 1),
            self._dummy_kernel[:, 3 * self._num_unitwise:]) \
            + self._dummy_bias[3 * self._num_unitwise:]

        # Diagonal connections
        batch_size = tf.shape(inputs)[0]
        random_shape = tf.stack([batch_size, tf.constant(self._num_units - self._num_unitwise)])

        stddev_f = tf.nn.moments(f, axes=[0,1])[1]
        stddev_i = tf.nn.moments(i, axes=[0,1])[1]
        stddev_j = tf.nn.moments(j, axes=[0,1])[1]
        stddev_o = tf.nn.moments(o,axes=[0,1])[1]

        random_f = tf.random.normal(random_shape, 0, stddev_f)
        random_i = tf.random.normal(random_shape, 0, stddev_i)
        random_j = tf.random.normal(random_shape, 0, stddev_j)
        random_o = tf.random.normal(random_shape, 0, stddev_o)

        i = tf.concat([i, random_i], axis=-1)
        j = tf.concat([j, random_j], axis=-1)
        o = tf.concat([o, random_o], axis=-1)
        f = tf.concat([f, random_f], axis=-1)

        i = tf.roll(i, self.roll, [-1])
        j = tf.roll(j, self.roll, [-1])
        o = tf.roll(o, self.roll, [-1])
        f = tf.roll(f, self.roll, [-1])

        c = (tf.sigmoid(f + self._forget_bias) * c_prev + tf.sigmoid(i) *
             self._activation(j))

        m = tf.sigmoid(o) * self._activation(c)

        new_state = tf.concat([c, m], 1)
        return m, new_state

class SparseDummyGRUCell(object):
    def __init__(self, params):
        pass

    def __call__(self, input):
        pass

class SparseLSTMCell(object):

    def __init__(self, num_units, sparse_list, forget_bias=1.0,
                 input_depth=None, state_is_tuple=False, activation='tanh',
                 use_sparse_mul=False, scope=None, seed=None, num_shards=1):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        self._forget_bias = forget_bias
        self._scope = scope or 'sparse_lstm'
        self._num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._activation = get_activation_func(activation)
        self._use_sparse_mul = use_sparse_mul

        i,j,f,o = split_indices(sparse_list, num_units, 4, axis=1)

        with tf.variable_scope(self._scope):
            with tf.device("/cpu:0"):
                # no need to shard sparse matrix
                self.wi, wi = get_sparse_sharded_weight_matrix(
                    [input_depth+num_units, num_units],
                    i, name='sparse_i', num_shards=1, dtype=tf.float32
                )

                self.wj, wj = get_sparse_sharded_weight_matrix(
                    [input_depth+num_units, num_units],
                    j, name='sparse_j', num_shards=1, dtype=tf.float32
                )

                self.wf, wf = get_sparse_sharded_weight_matrix(
                    [input_depth + num_units, num_units],
                    f, name='sparse_f', num_shards=1, dtype=tf.float32
                )

                self.wo, wo = get_sparse_sharded_weight_matrix(
                    [input_depth + num_units, num_units],
                    o, name='sparse_o', num_shards=1, dtype=tf.float32
                )

                initializer = tf.zeros_initializer
                self.bi = tf.get_variable(
                    name='bi', shape=[self._num_units], initializer=initializer
                )
                self.bj = tf.get_variable(
                    name='bj', shape=[self._num_units], initializer=initializer
                )
                self.bf = tf.get_variable(
                    name='bf', shape=[self._num_units], initializer=initializer
                )
                self.bo = tf.get_variable(
                    name='bo', shape=[self._num_units], initializer=initializer
                )

        self.var = [self.bi, self.bj, self.bf, self.bo, wi, wj, wf, wo]
        '''
        self.initialize_op = tf.initialize_variables([self.bi, self.bj, self.bf, self.bo, wi, wj, wf, wo])
        '''
        self.output_size = num_units
        self.state_size = 2 * num_units

    def zero_state(self, batch_size, dtype):
        return tf.zeros(
            tf.stack([batch_size, tf.constant(self.state_size)]),
            dtype=dtype)

    @property
    def sparsity(self):
        return self._sparsity

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                raise NotImplementedError
            else:
                c, h = tf.split(state, 2, 1)
            # concat = _linear([inputs, h], 4 * self._num_units, True)

            i = sparse_matmul([inputs, h], self.wi) + self.bi
            j = sparse_matmul([inputs, h], self.wj) + self.bj
            f = sparse_matmul([inputs, h], self.wf) + self.bf
            o = sparse_matmul([inputs, h], self.wo) + self.bo

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                raise NotImplementedError
            else:
                new_state = tf.concat([new_c, new_h], 1)
            return new_h, new_state

def dummy_step(cell, _input, _state, _output_tensor, _i):
    _output, _next_state = cell.call(_input[:, _i], _state[:, _i])

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


class SparseDummyRecurrentNetwork(object):
    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, input_depth, reuse=True,
                 num_unitwise=None, swap_memory=True, seed=12345, num_shards=1):
        self._scope = scope
        self._use_lstm = True if 'lstm' in recurrent_cell_type else False
        _cell_proto, _cell_kwargs = get_dummy_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size
        self.swap_memory = True

        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs, input_depth=input_depth,
                seed=seed, num_unitwise=num_unitwise)
            self.roll = self._cell.roll

    def __call__(self, input_tensor, hidden_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor,
                initial_state=hidden_states,
                dtype=tf.float32,
                swap_memory=True
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

    @property
    def weight(self):
        return self._cell._dummy_kernel

    def sample(self, input_vec):
        output_shape = tf.shape(input_vec)

        if self._use_lstm:
            random_states = tf.random.normal(
                shape=tf.concat([output_shape[:-1], tf.constant([2*self._hidden_size])], axis=0), stddev=0.3)

        else:
            random_states = tf.random.normal(
                shape=tf.concat([output_shape[:-1], tf.constant([self._hidden_size])], axis=0), stddev=0.3)

        random_output = tf.random.normal(
            shape=tf.concat([output_shape[:-1], tf.constant([self._hidden_size])], axis=0), stddev=0.3)

        if self._activation_type is not None:
            act_func = \
                get_activation_func(self._activation_type)
            random_output = \
                act_func(random_output, name='activation_0')

        if self._normalization_type is not None:
            normalizer = get_normalizer(self._normalization_type,
                                        train=self._train)
            random_output = \
                normalizer(random_output, 'normalizer_0')

        return random_output, random_states

class SparseRecurrentNetwork(object):
    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type, sparse_list,
                 train, hidden_size, input_depth, seed=12345,
                 dtype=tf.float32, swap_memory=True, reuse=None, num_shards=1):
        self._scope = scope
        self._use_lstm = True if 'lstm' in recurrent_cell_type else False
        _cell_proto, _cell_kwargs = get_sparse_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size

        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs,
                input_depth=input_depth, sparse_list=sparse_list, seed=seed,
                num_shards=num_shards)

        self.var = self._cell.var
        '''
        self.initialize_op = self._cell.initialize_op
        '''
        self.swap_memory = bool(swap_memory)

    def __call__(self, input_tensor, hidden_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor,
                initial_state=hidden_states,
                dtype=tf.float32,
                swap_memory=True
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

class SparseDummyFullyConnected(object):
    def __init__(self, input_depth, hidden_size, scope,
                 activation_type, normalizer_type, seed=1,
                 num_unitwise=None, train=True, use_bias=True, num_shards=1,
                 **kwargs):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size
        self.num_unitwise = num_unitwise

        self.use_bias = use_bias
        self.seed = seed

        with tf.variable_scope(self._scope):
            self.weight = tf.placeholder(shape=[input_depth, num_unitwise], dtype=tf.float32)
            if use_bias:
                self._b = tf.zeros(shape=[hidden_size], dtype=tf.float32)

            self.roll = tf.placeholder_with_default(tf.zeros([1], dtype=tf.int32), [1])
        print("Sparse:{}".format(self._scope))

        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope):
            res = tf.matmul(flat_input, self.weight)
            batch_size = tf.shape(flat_input)[0]
            random_shape = tf.stack([batch_size,
                tf.constant(self.hidden_size - self.num_unitwise)])

            stddev = tf.nn.moments(res, axes=[0,1])[0]

            random_res = tf.random.normal(random_shape, 0, stddev)
            res = tf.concat([res, random_res], axis=1)

            res = tf.roll(res, self.roll, [-1])
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

    def sample(self, input):
        output_shape = tf.shape(input)

        sample = tf.random.normal(stddev=0.1,
            shape=tf.concat([output_shape[:-1],
                tf.constant([self.hidden_size])], axis=0))

        if self._activation_type is not None:
            act_func = \
                get_activation_func(self._activation_type)
            sample = \
                act_func(sample, name='activation')

        if self._normalizer_type is not None:
            normalizer = get_normalizer(self._normalizer_type,
                                        train=self._train)
            sample = \
                normalizer(sample, 'normalizer')

        return sample

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
        print('Sparse:{}'.format(self._scope))

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        '''if use_bias:
            self.initialize_op = tf.initialize_variables([self._b, *var])
        else:
            self.initialize_op = tf.initialize_variables([*var])
        '''
        self.var = [self._b, *var]

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

class SparseDummyEmbedding(object):

    def __init__(self, input_depth, hidden_size, scope,
        seed=1):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size
        self.seed = seed
        with tf.variable_scope(self._scope):
            self.weight = tf.placeholder(shape=[input_depth, hidden_size], dtype=tf.float32)
        self.roll = None

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)

        with tf.variable_scope(self._scope):
            res = tf.nn.embedding_lookup(self.weight, input_vec)

        return tf.reshape(res,
            tf.concat([output_shape, tf.constant([self.hidden_size])], axis=0)
        )

    def sample(self, input):
        output_shape = tf.shape(input)

        embedding = tf.random.uniform([self.input_depth, self.hidden_size], -.1, .1, )

        return tf.nn.embedding_lookup(embedding, input)

class SparseEmbedding(object):
    def __init__(self, input_depth, hidden_size, scope,
        sparse_list, seed=1):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size
        self.seed = seed

        with tf.variable_scope(self._scope):
            self.weight, v = get_sparse_weight_matrix(
                [input_depth, hidden_size], sparse_list, out_type='dense')

        self.var = [v]
        '''
        self.initialize_op = tf.initialize_variables([v])
        '''

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)

        with tf.variable_scope(self._scope):
            res = tf.nn.embedding_lookup(self.weight, input_vec)

        return tf.reshape(res,
            tf.concat([output_shape, tf.constant([self.hidden_size])], axis=0)
        )

class DenseFullyConnected(object):

    def __init__(self, input_depth, hidden_size, scope,
                 activation_type, normalizer_type, weight,
                 train=True, num_shards=1, transpose_weight=None):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size

        with tf.variable_scope(self._scope):
            print("Num_Shards", num_shards)
            if num_shards > 1:
                weight = weight.T if transpose_weight else weight
                self.weight, v = get_concat_variable(weight, weight.shape,
                    dtype=tf.float32, num_shards=num_shards)

            else:
                self.weight = tf.Variable(weight, dtype=tf.float32)
                v = [self.weight]

            self.b = tf.Variable(tf.zeros(shape=[hidden_size], dtype=tf.float32))
        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        '''
        self.initialize_op = tf.initialize_variables([self.b, *v])
        '''
        self.var = [self.b, *v]

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
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
            self.weight, v = get_mask(weight)
        '''
        self.initialize_op = tf.initialize_variables([v])
        '''
        self.var = [v]

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)

        with tf.variable_scope(self._scope):
            res = tf.nn.embedding_lookup(self.weight, input_vec)

        return tf.reshape(res,
            tf.concat([output_shape, tf.constant([self.hidden_size])], axis=0)
        )

class DenseRecurrentNetwork(object):
    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type, weight,
                 train, hidden_size, input_depth, seed=12345,
                 dtype=tf.float32, reuse=None, num_shards=1):
        self._scope = scope
        self._use_lstm = True if 'lstm' in recurrent_cell_type else False
        _cell_proto, _cell_kwargs = get_dense_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size

        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs,
                input_depth=input_depth, init_matrix=weight, seed=seed,
                num_shards=num_shards)
        '''
        self.initialize_op = self._cell.initialize_op
        '''

        self.var = self._cell.var

    def __call__(self, input_tensor, hidden_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor,
                initial_state=hidden_states,
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

def get_dense_rnn_cell(rnn_cell_type):
    cell_args = {}
    if rnn_cell_type == 'basic':
        raise NotImplementedError
    elif rnn_cell_type == 'lstm':
        cell_type = DenseLSTMCell
        cell_args['state_is_tuple'] = False

    return cell_type, cell_args

class DenseLSTMCell(object):
    def __init__(self, num_units, init_matrix, forget_bias=1.0,
                 input_depth=None, state_is_tuple=False, activation='tanh',
                scope=None, seed=None, num_shards=1):
        """Initialize the basic LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        self._forget_bias = forget_bias
        self._scope = scope or 'sparse_lstm'
        self._num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._activation = get_activation_func(activation)

        with tf.variable_scope(self._scope):
            self.wi = tf.Variable(init_matrix[:,:num_units], dtype=tf.float32)

            self.wj = tf.Variable(init_matrix[:,num_units:2*num_units], dtype=tf.float32)

            self.wf = tf.Variable(init_matrix[:,2*num_units:3*num_units], dtype=tf.float32)

            self.wo = tf.Variable(init_matrix[:,3*num_units:4*num_units], dtype=tf.float32)

            self.bias = tf.get_variable(
                name='bias', shape=[4 * self._num_units], initializer=tf.zeros_initializer
            )

        self.var = [self.wi, self.wj, self.wf, self.wo, self.bias]

        '''
        self.initialize_op = tf.initialize_variables([self.bias, self.wi, self.wj, self.wf, self.wo])
        '''

        self.output_size = num_units
        self.state_size = 2*num_units

    def zero_state(self, batch_size, dtype):
        return tf.zeros(
            tf.stack([batch_size, tf.constant(self.state_size)]),
            dtype=dtype)

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                raise NotImplementedError
            else:
                c, h = tf.split(state, 2, 1)
            # concat = _linear([inputs, h], 4 * self._num_units, True)

            concat = tf.concat([inputs, h], axis=1)

            i = tf.matmul(concat, self.wi) + self.bias[:self._num_units]
            j = tf.matmul(concat, self.wj) + \
                self.bias[self._num_units:2*self._num_units]
            f = tf.matmul(concat, self.wf) + \
                self.bias[2*self._num_units:3*self._num_units]
            o = tf.matmul(concat, self.wo) + \
                self.bias[3*self._num_units:]

            new_c = (c * tf.sigmoid(f + self._forget_bias)) + (tf.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                raise NotImplementedError
            else:
                new_state = tf.concat([new_c, new_h], 1)
            return new_h, new_state

class SparseKNET(object):
    def __init__(self, activation_type, input_depth, hidden_size, scope,
                 sparse_var=None, **kwargs):
        with tf.variable_scope(scope):
            self.input_depth = input_depth
            self.hidden_size = hidden_size
            self.in_sqrt = int(np.sqrt(input_depth))
            part_in1 = part_in2 = self.in_sqrt
            part_out1 = part_out2 = hidden_size

            if sparse_var is None:
                self.w1 = tf.get_variable('w1', [part_in1, part_out1], dtype=tf.float32)
                self.w2 = tf.get_variable('w2', [part_in2, part_out2], dtype=tf.float32)

            else:
                self.w1 = tf.Variable(sparse_var[0], dtype=tf.float32)
                self.w2 = tf.Variable(sparse_var[1], dtype=tf.float32)

            self.b = tf.get_variable(
                'b', [hidden_size], dtype=tf.float32,
                initializer=tf.zeros_initializer
            )
            self.w = [self.w1, self.w2]

        self.var = [self.b, self.w1, self.w2]

        self._activation = get_activation_func(activation_type)

        '''
        self.initialize_op = tf.initialize_variables([self.b, self.w1, self.w2])
        '''
    def __call__(self, input_tensor):
        input_reshape = tf.reshape(input_tensor, [-1, self.in_sqrt, self.in_sqrt])
        reduce_l = tf.reduce_sum(input_reshape, 1)
        reduce_r = tf.reduce_sum(input_reshape, 2)

        res = tf.matmul(reduce_l, self.w1) + tf.matmul(reduce_r, self.w2) + self.b
        res = self._activation(res)

        return tf.reshape(res,
            tf.concat([tf.shape(input_tensor)[:-1], tf.constant([self.hidden_size])], axis=0)
        )

def get_mask(weight):
    mask = np.zeros_like(weight)
    mask_inds = np.nonzero(weight)
    
    mask[mask_inds] = 1

    mask = tf.constant(mask, dtype=tf.float32)
    weight = tf.Variable(weight, dtype=tf.float32, trainable=True)

    return mask * weight, weight

class CustomSparseShardedTensor(tf.SparseTensor):
    def __init__(self, indices, values, dense_shape, dtype=tf.int64, **kwargs):
        """Creates a `SparseTensor`.
        Args:
          indices: A 2-D int64 tensor of shape `[N, ndims]`.
          values: A 1-D tensor of any type and shape `[N]`.
          dense_shape: A 1-D int64 tensor of shape `[ndims]`.
        """

        print(dtype)
        with tf.name_scope(None, "SparseTensor",
                            [indices, values, dense_shape]):
            dense_shape = tf.convert_to_tensor(
                dense_shape, name="dense_shape", dtype=tf.int64)
        self._indices = indices
        self._values = values
        self._dense_shape = dense_shape

        indices_shape = indices.get_shape().with_rank(2)
        values_shape = values.get_shape().with_rank(1)
        dense_shape_shape = dense_shape.get_shape().with_rank(1)

        # Assert number of rows in indices match the number of elements in values.
        indices_shape[0].merge_with(values_shape[0])
        # Assert number of columns in indices matches the number of elements in
        # dense_shape.

        indices_shape[1].merge_with(dense_shape_shape[0])

# XXX(rafal): Code below copied from rnn_cell.py
def _get_sharded_variable(values, shape, dtype, num_shards, name='', constant=False, **kwargs):
    """Get a list of sharded variables with the given dtype."""
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                         (shape, num_shards))

    print(shape)
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    start = 0
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1

        if constant:
            shards.append(tf.constant(values[start:start + current_size], name=name + "_%d" % i,
                dtype=dtype, **kwargs))

        else:
            shards.append(tf.Variable(values[start:start+current_size], name=name + "_%d" % i,
                dtype=dtype, **kwargs))
        start = start + current_size
    return shards

def get_concat_variable(values, shape, dtype, num_shards, **kwargs):
    print(num_shards)
    """Get a sharded variable concatenated into one tensor."""
    _sharded_variable = _get_sharded_variable(values, shape, dtype, num_shards, **kwargs)
    if len(_sharded_variable) == 1:
        return _sharded_variable[0], [_sharded_variable[0]]

    return tf.concat(_sharded_variable, 0), _sharded_variable

def get_concat_sparse_variable(shape, sparse_list, dtype=tf.float32, num_shards=1, **kwargs):
    """Get a sharded variable concatenated into one tensor."""
    values, indices = sparse_list

    sparse_values, sparse_var = get_concat_variable(
        values, values.shape, dtype, num_shards, **kwargs)

    # if np.max(shape)<2**16:
    # dtype=tf.int16
    if np.max(shape) < 2 ** 32:
        dtype = tf.int32
    else:
        dtype = tf.int64

    sparse_indices, _ = get_concat_variable(
        indices, indices.shape, dtype, num_shards, constant=True,
        **kwargs
    )

    sparse_matrix = CustomSparseShardedTensor(indices=sparse_indices,
        values=sparse_values, dense_shape=shape, **kwargs)

    return sparse_matrix, sparse_var

def get_sparse_sharded_weight_matrix(*args, num_shards=1, **kwargs):
    # shape, sparse_list, out_type='sparse', dtype=tf.float32, name=''
    if num_shards == 1:
        return get_sparse_weight_matrix(*args, **kwargs)

    else:
        return get_concat_sparse_variable(*args, num_shards=num_shards, **kwargs)

def _compute_factor_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None,
                            num_factor=1,
                            seed=None):
    # weights cannot be list

    with tf.name_scope(name, "compute_sampled_logits",
                        [weights,biases, inputs, labels]):
        if labels.dtype != tf.int64:
            labels = tf.cast(labels, tf.int64)
        labels_flat = tf.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        if sampled_values is None:
            sampled_values = tf.nn.log_uniform_candidate_sampler(
                true_classes=labels,
                num_true=num_true,
                num_sampled=num_sampled,
                unique=True,
                range_max=num_classes,
                seed=seed)
        # NOTE: pylint cannot tell that 'sampled_values' is a sequence
        # pylint: disable=unpacking-non-sequence
        sampled, true_expected_count, sampled_expected_count = (
            tf.stop_gradient(s) for s in sampled_values)
        # pylint: enable=unpacking-non-sequence
        sampled = tf.cast(sampled, tf.int64)

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        all_ids = tf.concat([labels_flat, sampled], 0)

        # Retrieve the true weights and the logits of the sampled weights.

        # weights shape is [num_classes, dim]
        all_w = tf.transpose(tf.nn.embedding_lookup(
            tf.transpose(weights, [1,0]), all_ids,
            partition_strategy=partition_strategy), [1,0])
        # true_w shape is [batch_size * num_true, dim]
        true_w = tf.slice(all_w, [0, 0], tf.stack([-1, tf.shape(labels_flat)[0]]))

        # shape is [num_sample, dim]
        sampled_w = tf.slice(
            all_w, tf.stack([0, tf.shape(labels_flat)[0]]), [-1, -1])
        # inputs has shape [batch_size, factor * n_factor]
        # sampled_w has shape [num_sampled, factor]
        # Apply reduce(X*W'), which yields [batch_size, num_sampled]

        factor_inputs = tf.reshape(inputs, tf.stack([tf.shape(inputs)[0], num_factor, -1]))

        sampled_logits_mat = tf.einsum('ijl,lk->ijk', factor_inputs, sampled_w)
        sampled_logits = tf.reduce_mean(sampled_logits_mat, axis=1)

        # Retrieve the true and sampled biases, compute the true logits, and
        # add the biases to the true and sampled logits.
        all_b = tf.nn.embedding_lookup(
            biases, all_ids, partition_strategy=partition_strategy)
        # true_b is a [batch_size * num_true] tensor
        # sampled_b is a [num_sampled] float tensor
        true_b = tf.slice(all_b, [0], tf.shape(labels_flat))
        sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

        # inputs shape is [batch_size, factor, n_factor]
        # true_w shape is [batch_size, num_true, n_factor]
        # row_wise_dots is [batch_size, n_factor]

        dim = tf.shape(true_w)[0]
        new_true_w_shape = [-1, num_true, 1, dim]
        new_true_w = tf.tile(tf.reshape(true_w, new_true_w_shape), [1,1,num_factor,1])

        row_wise_dots = tf.reduce_mean(tf.expand_dims(factor_inputs, 1) \
            * tf.reshape(true_w, new_true_w_shape), axis=2)
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = tf.reshape(row_wise_dots, [-1, dim])
        true_logits = tf.reshape(tf.reduce_sum(dots_as_matrix, axis=-1), [-1, num_true])
        true_b = tf.reshape(true_b, [-1, num_true])
        true_logits += true_b
        sampled_logits += sampled_b

        if remove_accidental_hits:
            pass

        if subtract_log_q:
            # Subtract log of Q(l), prior probability that l appears in sampled.
            true_logits -= tf.log(true_expected_count)
            sampled_logits -= tf.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = tf.concat([true_logits, sampled_logits], 1)

        # true_logits is a float tensor, ones_like(true_logits) is a float
        # tensor of ones. We then divide by num_true to ensure the per-example
        # labels sum to 1.0, i.e. form a proper probability distribution.
        out_labels = tf.concat([
            tf.ones_like(true_logits) / num_true,
            tf.zeros_like(sampled_logits)
        ], 1)
    return out_logits, out_labels

def _sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
    # a matrix.  The gradient of _sum_rows(x) is more efficient than
    # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
    # we use _sum_rows(x) in the nce_loss() computation since the loss
    # is mostly used for training.
    cols = tf.shape(x)[1]
    ones_shape = tf.stack([cols, 1])
    ones = tf.ones(ones_shape, x.dtype)
    return tf.reshape(tf.matmul(x, ones), [-1])

