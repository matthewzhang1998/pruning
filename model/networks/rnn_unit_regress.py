import tensorflow as tf
from model.networks.base_network import *
from util.sparse_util import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, output_size, seed, init,
            use_embedding=True, use_softmax=True):
        super(RNNModel, self).__init__(params)

        self.Network['Input_Size'] = []
        self.Network['Output_Size'] = []

        self.use_embedding = (self.params.dataset not in ['timit', 'seq_mnist'])

        self._input_size = input_size

        self.Network['Type'] = []
        self.Network['Params'] = []

        params = {'scope':'embed',
                  'hidden_size': self.params.embed_size,
                  'input_depth': input_size,
                  'trainable': self.params.train_embed}
        self.Network['Type'].append('embedding')
        self.Network['Params'].append(params)

        if self.use_embedding:
            input_size = self.params.embed_size

        for ii in range(len(self.params.rnn_r_hidden_seq)):
            if self.params.rnn_bidirectional:
                pass
            elif self.params.rnn_dilated:
                pass
            else:
                params = {'input_depth': input_size,
                          'hidden_size': self.params.rnn_r_hidden_seq[ii],
                          'activation_type': self.params.rnn_r_act_seq[ii],
                          'normalizer_type': self.params.rnn_r_norm_seq[ii],
                          'recurrent_cell_type': self.params.rnn_cell_type,
                          'train': True, 'scope': 'rnn' + str(ii),
                          'num_shards': self.params.num_shards,
                          'max_length': self.params.max_length,
                          'num_unitwise': self.params.num_unitwise_rnn,
                          }

            self.Network['Params'].append(params)
            self.Network['Type'].append('rnn')
            input_size = self.params.rnn_r_hidden_seq[ii]

        if use_softmax:
            if self.params.use_factor_softmax:
                assert input_size % self.params.num_factor == 0
                input_size = int(input_size/self.params.num_factor)
            params = {'hidden_size': output_size,
                      'input_depth': input_size,
                      'activation_type': None, 'normalizer_type': None,
                      'train': True, 'scope': 'softmax',
                      'num_shards': self.params.num_shards,
                      'transpose_weight': not (self.params.use_knet \
                        or self.params.use_factor_softmax)
                      }

            num_unitwise = self.params.embed_size
            self.Network['Type'].append('mlp')

            self.Network['Params'].append(params)

        self.initialize_op = []

        self.Network['Net'] = [None for _ in self.Network['Params']]

    def init_model(self, scope, embed, softmax, use_dense=False):
        with tf.variable_scope(scope):
            self.Network['Net'][0] = DenseEmbedding(
                **self.Network['Params'][0], weight=embed
            )
            self.Network['Net'][-1] = DenseFullyConnected(
                **self.Network['Params'][-1], weight=softmax
            )

            self.Network['Net'][1] = SparseDummyRecurrentNetwork(
               **self.Network['Params'][1],
               swap_memory=self.params.rnn_swap_memory
            )

    def __call__(self, input, return_rnn=False):
        use_last = (not self.params.use_sample_softmax)

        if self.use_embedding:
            output = self.Network['Net'][0](input)
        else:
            output = input
        print(return_rnn)

        if return_rnn:
            output, _, rnn_hidden = self.Network['Net'][1](output, return_rnn=return_rnn)
        else:
            output = self.Network['Net'][1](output)[0]

        output = tf.transpose(tf.squeeze(tf.stack(output)), [1,0,2])

        if self.params.dataset in ['timit', 'seq_mnist']:
            output = output[:,-1,:]

        if use_last:
            output = self.Network['Net'][-1](output)

        if return_rnn:
            return output, rnn_hidden
        else:
            return output

    def hessian_variable(self):
        return [self.Network['Net'][1].var]

    def get_placeholders(self):
        return self.Network['Net'][1].placeholder

    def run_dummy(self, hidden, states, use_last=False):
        if states is None:
            states = self.Network['Net'][1]._cell.zero_state(self.params.dummy_batch, tf.float32)

        outputs = []
        for i in range(self.params.unroll_dummy):
            hidden = tf.random.normal((self.params.dummy_batch, self.Network['Params'][1]['input_depth']),
                stddev=self.params.dummy_noise)
            output, states, prev, new = self.Network['Net'][1]._cell(hidden, states,
                return_prev=True)
            outputs.append(output)

        output = tf.transpose(tf.stack(outputs), [1,0,2])

        print("Output:", output)

        if self.params.dataset in ['timit', 'seq_mnist']:
            output = output[:,-1,:]

        if use_last and not self.params.use_sample_softmax:
            output = self.Network['Net'][-1](output)
            print(output)
        return new, prev, output