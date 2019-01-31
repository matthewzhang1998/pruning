import tensorflow as tf
import numpy as np
from util import norm_util

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
    print(sparse_values, sparse_indices.shape)
    with tf.variable_scope(scope):
        sparse_matrix = tf.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape=shape)
    return sparse_matrix

def get_tensor(sparse_values):
    trainable_var = tf.Variable(sparse_values),
    return
def get_sparse_weight_matrix(shape, sparse_list, out_type='sparse', dtype=tf.float32, name=''):
    # Directly Initialize the sparse matrix
    sparse_values, sparse_indices = sparse_list
    sparse_values = tf.Variable(sparse_values, dtype=dtype)

    if out_type == "sparse":
        sparse_weight_matrix = tf.SparseTensor(indices=sparse_indices,
            values=sparse_values, dense_shape=shape)
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
            sparse_matrix = tf.sparse_transpose(sparse_matrix, perm=[1, 0])
            res = tf.sparse_tensor_dense_matmul(sparse_matrix, input)
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
                 use_sparse_mul=False, scope=None, seed=None):
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

        with tf.variable_scope(self._scope):
            self._sparse_matrix, self._sparse_values = get_sparse_weight_matrix(
                [input_depth+num_units,4*num_units],
                sparse_list, name='sparse_linear'
            )

            initializer = tf.zeros_initializer
            self._bias = tf.get_variable(
                name='bias', shape=[4 * self._num_units], initializer=initializer
            )

        self.initialize_op = tf.initialize_variables([self._bias, self._sparse_values])

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

            concat = sparse_matmul([inputs, h], self._sparse_matrix) + self._bias
            i, j, f, o = tf.split(concat, 4, 1)

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

def _dynamic_rnn(cell, input, initial_state, hidden_size, use_lstm=True):
    _batch_size = tf.shape(input)[0]
    _dummy_outputs = tf.zeros(
        [_batch_size, 1, hidden_size]
    )

    if initial_state is None:
        if use_lstm:
            initial_state = tf.zeros(
                [_batch_size, 1, 2 * hidden_size]
            )
        else:
            initial_state = tf.zeros(
                [_batch_size, 1, hidden_size]
            )

    # specify first two
    _lambda = lambda a, b, c, d: dummy_step(cell, a, b, c, d)

    if use_lstm:
        _, _rnn_state_arr, _rnn_output_arr, _ = tf.while_loop(
            _condition, _lambda,
            [input, initial_state, _dummy_outputs, tf.constant(0)],
            shape_invariants=[input.get_shape(),
                tf.TensorShape((None, None, 2 * hidden_size)),
                tf.TensorShape((None, None, hidden_size)),
                tf.TensorShape(())]
        )

    else:
        _, _rnn_state_arr, _rnn_output_arr, _ = tf.while_loop(
            _condition, _lambda,
            [input, initial_state, _dummy_outputs, tf.constant(0)],
            shape_invariants=[input.get_shape(),
                tf.TensorShape((None, None, hidden_size)),
                tf.TensorShape((None, None, hidden_size)),
                tf.TensorShape(())]
        )

    _rnn_state_arr = _rnn_state_arr[:, 1:]
    _rnn_output_arr = _rnn_output_arr[:, 1:]
    return _rnn_output_arr, _rnn_state_arr

class SparseDummyRecurrentNetwork(object):
    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type,
                 train, hidden_size, input_depth, reuse=True,
                 num_unitwise=None, swap_memory=True, seed=12345):
        self._scope = scope
        self._use_lstm = True if 'lstm' in recurrent_cell_type else False
        _cell_proto, _cell_kwargs = get_dummy_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size
        self.swap_memory = swap_memory

        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs, input_depth=input_depth,
                seed=seed, num_unitwise=num_unitwise)
            self.roll = self._cell.roll

    def __call__(self, input_tensor, hidden_states=None):
        print(input_tensor, hidden_states)
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor,
                initial_state=hidden_states,
                dtype=tf.float32,
                swap_memory=self.swap_memory
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
                 dtype=tf.float32, swap_memory=True, reuse=None):
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
                input_depth=input_depth, sparse_list=sparse_list, seed=seed)

        self.initialize_op = self._cell.initialize_op
        self.swap_memory = swap_memory

    def __call__(self, input_tensor, hidden_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor,
                initial_state=hidden_states,
                dtype=tf.float32,
                swap_memory=self.swap_memory
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
                 num_unitwise=None, train=True, use_bias=True):

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
                 train=True, use_bias=True):

        self._scope = scope
        self.input_depth = input_depth
        self.use_bias = use_bias
        self.hidden_size = hidden_size

        with tf.variable_scope(self._scope):
            self.weight, self.var = get_sparse_weight_matrix([input_depth, hidden_size],
                sparse_list, out_type='sparse', dtype=tf.float32, name='')
            if use_bias:
                self._b = tf.Variable(tf.zeros(shape=[hidden_size], dtype=tf.float32))
        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        if use_bias:
            self.initialize_op = tf.initialize_variables([self._b, self.var])
        else:
            self.initialize_op = tf.initialize_variables([self.var])

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
            self.weight, self.var = get_sparse_weight_matrix(
                [input_depth, hidden_size], sparse_list, out_type='dense')

        self.initialize_op = tf.initialize_variables([self.var])

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
                 train=True):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size

        with tf.variable_scope(self._scope):
            self.weight, v = get_mask(weight)
            self._b = tf.Variable(tf.zeros(shape=[hidden_size], dtype=tf.float32))
        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self.initialize_op = tf.initialize_variables([self._b, v])

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope):
            res = tf.matmul(flat_input, self.weight) + self._b

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

        self.initialize_op = tf.initialize_variables([v])

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
                 dtype=tf.float32, reuse=None):
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
                input_depth=input_depth, init_matrix=weight, seed=seed)

        self.initialize_op = self._cell.initialize_op

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
                scope=None, seed=None):
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
            self.w, v = get_mask(init_matrix)

            self._bias = tf.get_variable(
                name='bias', shape=[4 * self._num_units], initializer=tf.zeros_initializer
            )

        self.initialize_op = tf.initialize_variables([self._bias, v])
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

            concat = tf.matmul(tf.concat([inputs, h], axis=1), self.w) + self._bias
            i, j, f, o = tf.split(concat, 4, 1)

            new_c = (c * tf.sigmoid(f + self._forget_bias)) + (tf.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                raise NotImplementedError
            else:
                new_state = tf.concat([new_c, new_h], 1)
            return new_h, new_state

def get_mask(weight):
    mask = np.zeros_like(weight)
    mask_inds = np.nonzero(weight)
    mask[mask_inds] = 1

    mask = tf.constant(mask, dtype=tf.float32)
    weight = tf.Variable(weight, dtype=tf.float32)

    return mask * weight, weight