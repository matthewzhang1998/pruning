import tensorflow as tf
from model.networks.base_network import *
from util.sparse_util import *
from util.dense_util import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, output_size, seed, init,
            use_embedding=True, use_softmax=True):
        super(RNNModel, self).__init__(params)

        self.Network['Input_Size'] = []
        self.Network['Output_Size'] = []

        self._input_size = input_size

        self.use_embedding = (self.params.dataset not in ['timit', 'seq_mnist'])

        self.Network['Type'] = []
        self.Network['Params'] = []

        params = {'scope':'embed', 'hidden_size': self.params.embed_size,
                  'input_depth': input_size, 'trainable': self.params.train_embed}
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
                          'num_shards': self.params.num_shards
                          }

            self.Network['Params'].append(params)
            self.Network['Type'].append('rnn')
            input_size = self.params.rnn_r_hidden_seq[ii]

        for ii in range(len(self.params.rnn_l_hidden_seq)):
            act_type = \
                self.params.rnn_l_act_seq[ii]
            norm_type = \
                self.params.rnn_l_norm_seq[ii]

            num_unitwise = self.params.rnn_l_hidden_seq[ii]

            params = {'input_depth': input_size,
                'hidden_size': self.params.rnn_l_hidden_seq[ii],
                'activation_type': act_type, 'normalizer_type': norm_type,
                'train':True, 'scope':'mlp'+str(ii),
                'num_shards': self.params.num_shards
            }

            self.Network['Params'].append(params)

            input_size = self.params.rnn_l_hidden_seq[ii]
            self.Network['Type'].append('mlp')

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

    def init_model(self, scope, embed, softmax, lstm, use_dense=False):
        with tf.variable_scope(scope):
            if self.use_embedding:
                self.Network['Net'][0] = DenseEmbedding(
                    **self.Network['Params'][0], weight=embed
                )
            self.Network['Net'][-1] = DenseFullyConnected(
                **self.Network['Params'][-1], weight=softmax
            )

            self.Network['Net'][1] = DenseVRecurrentNetwork(
               **self.Network['Params'][1], weight=lstm
            )

    def __call__(self, input, return_rnn=False):

        use_last = (not self.params.use_sample_softmax)

        if self.use_embedding:
            output = self.Network['Net'][0](input)
        else:
            output = input

        if return_rnn:
            output, _, rnn_hidden = self.Network['Net'][1](output, return_rnn=True)


        else:
            output = self.Network['Net'][1](output)[0]

        if self.params.dataset in ['timit', 'seq_mnist']:
            output = output[:,-1,:]

        if use_last:
            output = self.Network['Net'][-1](output)

        if return_rnn:
            return output, rnn_hidden

        else:
            return output