import tensorflow as tf
from model.networks.base_network import *
from util.sparse_util import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, output_size, seed, init,
            use_embedding=True, use_softmax=True):
        super(RNNModel, self).__init__(params)

        self.Network['Dummy'] = []
        self.Network['Input_Size'] = []
        self.Network['Output_Size'] = []

        self._input_size = input_size

        self.Network['Type'] = []
        self.Network['Params'] = []

        if use_embedding:
            params = {'scope':'embed', 'hidden_size': self.params.embed_size,
                      'input_depth': input_size,}

            self.Network['Dummy'].append(SparseDummyEmbedding(
                **params, seed=seed))

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

                self.Network['Dummy'].append(SparseDummyRecurrentNetwork(
                    **params, num_unitwise=self.params.num_unitwise_rnn,
                ))
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
            self.Network['Dummy'].append(SparseDummyFullyConnected(
                **params, seed=seed, num_unitwise=num_unitwise
            ))
            self.Network['Type'].append('mlp')

            self.Network['Params'].append(params)

        self.initialize_op = []
        self.Tensor['Temp'] = [None for _ in self.Network['Dummy']]

    def build_sparse(self, sparse_var, ii, use_dense):
        if 'softmax' in self.Network['Params'][ii]['scope'] and self.params.use_knet:
            self.Network['Dummy'][ii] = SparseKNET(
                **self.Network['Params'][ii], sparse_list=sparse_var
            )

        else:
            if not use_dense:
                if self.Network['Type'][ii] == 'rnn':
                    self.Network['Dummy'][ii] = SparseRecurrentNetwork(
                        **self.Network['Params'][ii], sparse_list=sparse_var,
                        swap_memory=self.params.rnn_swap_memory
                    )

                elif self.Network['Type'][ii] == 'mlp':
                    self.Network['Dummy'][ii] = SparseFullyConnected(
                        **self.Network['Params'][ii], sparse_list=sparse_var
                    )

                elif self.Network['Type'][ii] == 'embedding':
                    self.Network['Dummy'][ii] = SparseEmbedding(
                        **self.Network['Params'][ii], sparse_list=sparse_var
                    )
            else:
                if self.Network['Type'][ii] == 'rnn':
                    self.Network['Dummy'][ii] = DenseRecurrentNetwork(
                        **self.Network['Params'][ii], weight=sparse_var
                    )

                elif self.Network['Type'][ii] == 'mlp':
                    self.Network['Dummy'][ii] = DenseFullyConnected(
                        **self.Network['Params'][ii], weight=sparse_var
                    )

                elif self.Network['Type'][ii] == 'embedding':
                    self.Network['Dummy'][ii] = DenseEmbedding(
                        **self.Network['Params'][ii], weight=sparse_var
                    )
        self.initialize_op = self.Network['Dummy'][ii].initialize_op

    def __call__(self, input, use_last=True):
        self.Tensor['Intermediate'] = [None for _ in self.Network['Dummy']]
        for i, network in enumerate(self.Network['Dummy']):
            if (not use_last) and i == len(self.Network['Dummy']) - 1:
                break
            if self.Network['Type'][i] == 'rnn':
                self.Tensor['Intermediate'][i] = network(input)[0] # get outputs, not hidden state

            elif self.Network['Type'][i] == 'embedding':
                with tf.device("/cpu:0"):
                    self.Tensor['Intermediate'][i] = network(input)

            else:
                with tf.device("/cpu:0"):
                    self.Tensor['Intermediate'][i] = network(input)

            input = self.Tensor['Intermediate'][i]

        if use_last:
            return self.Tensor['Intermediate'][-1]
        else:
            return self.Tensor['Intermediate'][-2]


    ### UNUSED FUNCTION ###
    def unit(self, input, ii, use_last=True):
        # gotta fix this for multilayer
        self.Tensor['Temp'][ii] = [None for _ in self.Network['Dummy']]
        for j in range(ii):
            if (not use_last) and j == len(self.Network['Dummy']) - 1:
                break

            if self.Network['Type'][j] == 'rnn':
                input = self.Tensor['Temp'][ii][j] = self.Network['Dummy'][j].sample(input)[0]

            else:
                input = self.Tensor['Temp'][ii][j] = self.Network['Dummy'][j].sample(input)

        if self.Network['Type'][ii] == 'rnn':
            input = self.Tensor['Temp'][ii][ii] = self.Network['Dummy'][ii](input)[0]

        else:
            input = self.Tensor['Temp'][ii][ii] = self.Network['Dummy'][ii](input)

        for k in range(ii+1,len(self.Network['Dummy'])):
            if (not use_last) and k == len(self.Network['Dummy']) - 1:
                break

            if self.Network['Type'][k] == 'rnn':
                input = self.Tensor['Temp'][ii][k] = self.Network['Dummy'][k](input)[0]

            else:
                input = self.Tensor['Temp'][ii][k] = self.Network['Dummy'][k](input)

        if use_last:
            return self.Tensor['Temp'][ii][-1]
        else:
            return self.Tensor['Temp'][ii][-2]

    def get_dummy_variables(self):
        dummy_weights = []
        for net in self.Network['Dummy']:
            dummy_weights.append(net.weight)
        return dummy_weights

    def get_roll_variables(self):
        dummy_roll = []
        for net in self.Network['Dummy']:
            dummy_roll.append(net.roll)
        return dummy_roll