import tensorflow as tf
from model.networks.base_network import *
from util.sparse_util import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, output_size, seed, init,
            use_embedding=True, use_softmax=True):
        super(RNNModel, self).__init__(params)

        self.Network['Input_Size'] = []
        self.Network['Output_Size'] = []

        self._input_size = input_size

        self.Network['Type'] = []
        self.Network['Params'] = []

        if use_embedding:
            params = {'scope':'embed', 'hidden_size': output_size, #self.params.embed_size,
                      'input_depth': input_size,}
            self.Network['Type'].append('embedding')
            self.Network['Params'].append(params)
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

            self.Network['Dummy'].append(SparseDummyFullyConnected(
                **params, seed=seed, num_unitwise=num_unitwise
            ))
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
            self.Network['Net'][0] = DenseEmbedding(
                **self.Network['Params'][0], weight=embed
            )
            self.Network['Net'][-1] = DenseFullyConnected(
                **self.Network['Params'][-1], weight=softmax
            )
            if not self.params.rnn_use_dense:
                #self.Network['Net'][1] = SparseRecurrentNetwork(
                #    **self.Network['Params'][1], sparse_list=lstm,
                #    swap_memory=self.params.rnn_swap_memory
                #)
                pass

            else:
                pass
                #self.Network['Net'][1] = DenseRecurrentNetwork(
                #    **self.Network['Params'][1], weight=lstm,
                #    swap_memory=self.params.rnn_swap_memory
                #)

    def __call__(self, input, use_last=False):
        output = self.Network['Net'][0](input)
        #output = self.Network['Net'][1](output)[0]
        if use_last:
            output = self.Network['Net'][-1](output)

        return output