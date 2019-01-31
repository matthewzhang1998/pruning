import tensorflow as tf
from util.network_util import *

class DRWFullyConnected(object):
    def __init__(self, input_depth, hidden_size, scope,
                 activation_type, normalizer_type, init_matrix,
                 train=True):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size

        with tf.variable_scope(self._scope):
            self.weight = self.weight = tf.Variable(init_matrix, dtype=tf.float32)
            self._b = tf.Variable(tf.zeros(shape=[hidden_size], dtype=tf.float32))
        self._train = train

        self._activation_type = activation_type
        self._normalizer_type = normalizer_type
        self.initialize_op = tf.initialize_variables([self._b, self.weight])

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)
        flat_input = tf.reshape(input_vec,
            tf.concat([[-1], [output_shape[-1]]], axis=0)
        )

        with tf.variable_scope(self._scope):
            res = tf.matmul(flat_input, self.weight)

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
            tf.concat([output_shape[:-1],
                tf.constant([self.hidden_size])], axis=0
            )
        )

class DRWEmbedding(object):
    def __init__(self, input_depth, hidden_size, scope,
        init_matrix, seed=1):

        self._scope = scope
        self.input_depth = input_depth
        self.hidden_size = hidden_size
        self.seed = seed

        with tf.variable_scope(self._scope):
            self.weight = tf.Variable(init_matrix, dtype=tf.float32)

        self.initialize_op = tf.initialize_variables([self.weight])

    def __call__(self, input_vec):
        output_shape = tf.shape(input_vec)

        with tf.variable_scope(self._scope):
            res = tf.nn.embedding_lookup(self.weight, input_vec)

        return tf.reshape(res,
            tf.concat([output_shape, tf.constant([self.hidden_size])], axis=0)
        )

class DRWRecurrentNetwork(object):
    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type, init_matrix,
                 train, hidden_size, input_depth, seed=12345,
                 dtype=tf.float32, reuse=None):
        self._scope = scope
        self._use_lstm = True if 'lstm' in recurrent_cell_type else False
        _cell_proto, _cell_kwargs = get_drw_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size

        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size, **_cell_kwargs,
                input_depth=input_depth, init_matrix=init_matrix, seed=seed)

    def __call__(self, input_tensor, hidden_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = _dynamic_rnn(
                self._cell, input_tensor,
                hidden_size=self._hidden_size, initial_state=hidden_states,
                use_lstm=self._use_lstm
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

    def mask(self):
        return self._cell.mask

    def theta(self):
        return self._cell.theta


def get_drw_rnn_cell(rnn_cell_type):
    cell_args = {}
    if rnn_cell_type == 'basic':
        raise NotImplementedError
    elif rnn_cell_type == 'lstm':
        cell_type = DRWLSTMCell
        cell_args['state_is_tuple'] = False

    return cell_type, cell_args

class DRWLSTMCell(object):

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

        sign = np.sign(init_matrix)
        val = np.abs(init_matrix)

        with tf.variable_scope(self._scope):
            self.theta = tf.Variable(val, dtype=tf.float32)
            self.sign = tf.constant(sign, dtype=tf.float32)
            self.mask = tf.placeholder(shape=sign.shape, dtype=tf.float32)

            self.w = self.theta*self.sign*self.mask

            self._bias = tf.get_variable(
                name='bias', shape=[4 * self._num_units], initializer=tf.zeros_initializer
            )

    @property
    def state_size(self):
        raise NotImplementedError

    @property
    def output_size(self):
        return self._num_units

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

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                raise NotImplementedError
            else:
                new_state = tf.concat([new_c, new_h], 1)
            return new_h, new_state



def dummy_step(cell, _input, _state, _output_tensor, _i):
    _output, _next_state = cell(_input[:, _i], _state[:, _i])

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