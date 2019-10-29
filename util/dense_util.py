import tensorflow as tf
import numpy as np

def get_dense_rnn_cell(rnn_cell_type):
    cell_args = {}
    if rnn_cell_type == 'basic':
        cell_type = tf.nn.rnn_cell.BasicRNNCell
    elif rnn_cell_type == 'peephole_lstm':
        cell_type = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
        cell_args['state_is_tuple'] = False
        cell_args['use_peephole'] = True

    elif rnn_cell_type == 'lstm':
        cell_type = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
        cell_args['state_is_tuple'] = False

    elif rnn_cell_type == 'gru':
        cell_type = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell

    return cell_type, cell_args

class DenseVRecurrentNetwork(object):
    def __init__(self, scope, activation_type,
                 normalizer_type, recurrent_cell_type, weight,
                 train, hidden_size, input_depth, seed=12345,
                 dtype=tf.float32, reuse=None, num_shards=1, **kwargs):
        self._scope = scope
        self._use_lstm = True if 'lstm' in recurrent_cell_type else False
        _cell_proto, _cell_kwargs = get_dense_rnn_cell(recurrent_cell_type)
        self._activation_type = activation_type
        self._normalization_type = normalizer_type
        self._train = train
        self._reuse = reuse
        self._hidden_size = hidden_size

        with tf.variable_scope(scope):
            self._cell = _cell_proto(hidden_size)
        '''
        self.initialize_op = self._cell.initialize_op
        '''

    def __call__(self, input_tensor, hidden_states=None):
        with tf.variable_scope(self._scope, reuse=self._reuse):
            _rnn_outputs, _rnn_states = tf.nn.dynamic_rnn(
                self._cell, input_tensor,
                initial_state=hidden_states,
                dtype=tf.float32
            )

        return _rnn_outputs, _rnn_states