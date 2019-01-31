import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import *
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.deprecation import deprecated

from util.network_util import *

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def unitwise_step(hidden_ix, cell, _input, _state, _output_tensor, _i):
    _output, _next_state = cell.unitwise(hidden_ix, _input[:, _i], _state[:, _i])

    _state = tf.concat(
        [_state, tf.expand_dims(_next_state, 1)], axis=1
    )

    _output_tensor = tf.concat(
        [_output_tensor, tf.expand_dims(_output, 1)], axis=1
    )
    _i += 1

    return _input, _state, _output_tensor, _i

def dummy_step(cell, _input, _state, _output_tensor, _i):
    _output, _next_state = cell.dummy(_input[:, _i], _state[:, _i])

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

def dynamic_dummy_rnn(cell, input, initial_state, hidden_size, use_lstm=True):
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
    return _rnn_state_arr, _rnn_output_arr

def dynamic_unitwise_rnn(ix, cell, input, initial_state, hidden_size, use_lstm=True):

    _batch_size = tf.shape(input)[0]
    _dummy_outputs = tf.zeros(
        [_batch_size, 1, hidden_size]
    )

    if initial_state is None:
        if use_lstm:
            initial_state = tf.zeros(
                [_batch_size, 1, 2*hidden_size]
            )
        else:
            initial_state = tf.zeros(
                [_batch_size, 1, hidden_size]
            )

    # specify first two
    _lambda = lambda a, b, c, d: unitwise_step(ix, cell, a, b, c, d)

    if use_lstm:
        _, _rnn_state_arr, _rnn_output_arr, _ = tf.while_loop(
            _condition, _lambda,
            [input, initial_state, _dummy_outputs, tf.constant(0)],
            shape_invariants=[input.get_shape(),
              tf.TensorShape((None, None, 2*hidden_size)),
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

    _rnn_state_arr = _rnn_state_arr[:,1:]
    _rnn_output_arr = _rnn_output_arr[:, 1:]
    return _rnn_state_arr, _rnn_output_arr

def normc_initializer(shape, seed=1234, stddev=1.0, dtype=tf.float32):
    npr = np.random.RandomState(seed)
    out = npr.randn(*shape).astype(np.float32)
    out *= stddev / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    return tf.constant(out)


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
        initializer = tf.zeros_initializer(shape, dtype=dtype)

    if init_method == "normc":
        initializer = tf.random_normal_initializer(
            mean=0, stddev=init_para["stddev"]/shape[0],
            seed=seed, dtype=dtype
        )

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
        initializer = bradly_initializer(
            shape, alpha=init_para['alpha'],
            seed=seed, dtype=dtype
        )

    else:
        raise ValueError("Unsupported initialization method!")

    return initializer


class BasicMaskRNNCell(LayerRNNCell):
    """The most basic RNN cell.
    Note that this cell is not optimized for performance. Please use
    `tf.contrib.cudnn_rnn.CudnnRNNTanh` for better performance on GPU.
    Args:
      num_units: int, The number of units in the RNN cell.
      activation: Nonlinearity to use.  Default: `tanh`. It could also be string
        that is within Keras activation function names.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      name: String, the name of the layer. Layers with the same name will
        share weights, but to avoid mistakes we require reuse=True in such
        cases.
      dtype: Default dtype of the layer (default of `None` means use the type
        of the first input). Required when `build` is called before `call`.
      **kwargs: Dict, keyword named properties for common layer attributes, like
        `trainable` etc when constructing the cell from configs of get_config().
    """

    @deprecated(None, "This class is equivalent as tf.keras.layers.SimpleRNNCell,"
                      " and will be replaced by that in Tensorflow 2.0.")
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None,
                 input_depth=None,
                 initializer=None,
                 seed=12345,
                 **kwargs):
        super(BasicMaskRNNCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn("%s: Note that this cell is not optimized for performance. "
                         "Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better "
                         "performance on GPU.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

        self._mask = tf.placeholder(dtype=tf.float32,
                                    shape=[input_depth + self._num_units, self._num_units], name='mask')

        initializer = init([input_depth + self._num_units, self._num_units], 'kernel',
           init_data['w_init_method'], tf.float32, init_data['w_init_para'],
           seed=seed, trainable=True
        )
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=initializer
        )

        self._combined = self._kernel * self._mask

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._combined)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return output, output

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(BasicMaskRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf_export("nn.rnn_cell.LSTMCell")
class BasicMaskLSTMCell(LayerRNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.
    The default non-peephole implementation is based on:
      https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf
    Felix Gers, Jurgen Schmidhuber, and Fred Cummins.
    "Learning to forget: Continual prediction with LSTM." IET, 850-855, 1999.
    The peephole implementation is based on:
      https://research.google.com/pubs/archive/43905.pdf
    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.
    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    Note that this cell is not optimized for performance. Please use
    `tf.contrib.cudnn_rnn.CudnnLSTM` for better performance on GPU, or
    `tf.contrib.rnn.LSTMBlockCell` and `tf.contrib.rnn.LSTMBlockFusedCell` for
    better performance on CPU.
    """

    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None, num_unitwise=None,
                 num_unit_shards=None, num_proj_shards=None, init_data=None,
                 forget_bias=1.0, state_is_tuple=True, input_depth=None, seed=None,
                 activation=None, reuse=None, name=None, dtype=None, **kwargs):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          num_unit_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          num_proj_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`. It
            could also be string that is within Keras activation function names.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          dtype: Default dtype of the layer (default of `None` means use the type
            of the first input). Required when `build` is called before `call`.
          **kwargs: Dict, keyword named properties for common layer attributes, like
            `trainable` etc when constructing the cell from configs of get_config().
          When restoring from CudnnLSTM-trained checkpoints, use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(BasicMaskLSTMCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)
        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn("%s: Note that this cell is not optimized for performance. "
                         "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                         "performance on GPU.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializers.get(initializer)
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

        h_depth = self._num_units if self._num_proj is None else self._num_proj

        maybe_partitioner = (
            partitioned_variables.fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)

        initializer = init([input_depth + self._num_units, 4 * self._num_units], 'kernel',
           init_data['w_init_method'], tf.float32, init_data['w_init_para'],
           seed=seed, trainable=True
        )

        self._input_size = input_depth
        self._num_unitwise = num_unitwise if num_unitwise is not None else 1

        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units],
            initializer=initializer,
            partitioner=maybe_partitioner)

        self._mask = tf.placeholder(
            shape=[input_depth + h_depth, 4 * self._num_units],
            dtype=tf.float32
        )

        if self.dtype is None:
            initializer = init_ops.zeros_initializer
        else:
            initializer = init_ops.zeros_initializer(dtype=self.dtype)
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=initializer)

        self._dummy_kernel = tf.placeholder(
            shape=[input_depth + self._num_units, 4*self._num_unitwise], dtype=tf.float32
        )
        self._dummy_bias = tf.placeholder(
            shape=[4*self._num_unitwise], dtype=tf.float32
        )

        self._combined = self._kernel * self._mask

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                               initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                               initializer=self._initializer)

        self.built = True

    def call(self, inputs, state):
        """Run one step of LSTM.
        Args:
          inputs: input Tensor, must be 2-D, `[batch, input_size]`.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
        Returns:
          A tuple containing:
          - A `2-D, [batch, output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.
        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = math_ops.matmul(
            array_ops.concat([inputs, m_prev], 1), self._combined)
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes:
            raise NotImplementedError
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            raise NotImplementedError
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            raise NotImplementedError
            m = math_ops.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state

    def get_config(self):
        config = {
            "num_units": self._num_units,
            "use_peepholes": self._use_peepholes,
            "cell_clip": self._cell_clip,
            "initializer": initializers.serialize(self._initializer),
            "num_proj": self._num_proj,
            "proj_clip": self._proj_clip,
            "num_unit_shards": self._num_unit_shards,
            "num_proj_shards": self._num_proj_shards,
            "forget_bias": self._forget_bias,
            "state_is_tuple": self._state_is_tuple,
            "activation": activations.serialize(self._activation),
            "reuse": self._reuse,
        }
        base_config = super(BasicMaskLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BasicUnitLSTMCell(BasicMaskLSTMCell):
    def unitwise(self, ix, inputs, state):
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            raise NotImplementedError
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i = tf.tensordot(
            array_ops.concat([inputs, m_prev], 1),
            tf.expand_dims(self._combined[:,ix], -1), axes=1) \
            + self._bias[ix]

        j = tf.tensordot(
            array_ops.concat([inputs, m_prev], 1),
            tf.expand_dims(self._combined[:, ix+self._num_units], -1), axes=1) \
            + self._bias[ix+self._num_units]

        f = tf.tensordot(
            array_ops.concat([inputs, m_prev], 1),
            tf.expand_dims(self._combined[:, ix+2*self._num_units], -1), axes=1) \
            + self._bias[ix+2*self._num_units]

        o = tf.tensordot(
            array_ops.concat([inputs, m_prev], 1),
            tf.expand_dims(self._combined[:, ix+3*self._num_units], -1), axes=1) \
            + self._bias[ix+3*self._num_units]

        # Diagonal connections
        batch_size = tf.shape(inputs)[0]
        random_shape = tf.stack([batch_size, tf.constant(self._num_units-1)])

        random_f = tf.random.normal(random_shape, 0, 1)
        random_i = tf.random.normal(random_shape, 0, 1)
        random_j = tf.random.normal(random_shape, 0, 1)
        random_o = tf.random.normal(random_shape, 0, 1)

        i = tf.concat([random_i[:,:ix], i, random_i[:,ix:]], axis=-1)
        j = tf.concat([random_j[:,:ix], j, random_j[:,ix:]], axis=-1)
        o = tf.concat([random_o[:,:ix], o, random_o[:,ix:]], axis=-1)
        f = tf.concat([random_f[:,:ix], f, random_f[:,ix:]], axis=-1)

        if self._use_peepholes:
            raise NotImplementedError
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            raise NotImplementedError
        if self._use_peepholes:
            raise NotImplementedError
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            raise NotImplementedError

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state

    def dummy(self, inputs, state):
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            raise NotImplementedError
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i = tf.matmul(
            array_ops.concat([inputs, m_prev], 1),
            self._dummy_kernel[:,:self._num_unitwise]) \
            + self._dummy_bias[:self._num_unitwise]

        j = tf.matmul(
            array_ops.concat([inputs, m_prev], 1),
            self._dummy_kernel[:,self._num_unitwise:2*self._num_unitwise]) \
            + self._dummy_bias[self._num_unitwise:2*self._num_unitwise]

        f = tf.matmul(
            array_ops.concat([inputs, m_prev], 1),
            self._dummy_kernel[:,2*self._num_unitwise:3*self._num_unitwise]) \
            + self._dummy_bias[2*self._num_unitwise:3*self._num_unitwise]

        o = tf.matmul(
            array_ops.concat([inputs, m_prev], 1),
            self._dummy_kernel[:,3*self._num_unitwise:]) \
            + self._dummy_bias[3*self._num_unitwise:]

        # Diagonal connections
        batch_size = tf.shape(inputs)[0]
        random_shape = tf.stack([batch_size, tf.constant(self._num_units - self._num_unitwise)])

        random_f = tf.random.normal(random_shape, 0, 0.3)
        random_i = tf.random.normal(random_shape, 0, 0.3)
        random_j = tf.random.normal(random_shape, 0, 0.3)
        random_o = tf.random.normal(random_shape, 0, 0.3)

        i = tf.concat([i, random_i], axis=-1)
        j = tf.concat([j, random_j], axis=-1)
        o = tf.concat([o, random_o], axis=-1)
        f = tf.concat([f, random_f], axis=-1)

        if self._use_peepholes:
            raise NotImplementedError
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            raise NotImplementedError
        if self._use_peepholes:
            raise NotImplementedError
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            raise NotImplementedError

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state
