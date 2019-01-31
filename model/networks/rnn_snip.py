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
                          }

                self.Network['Dummy'].append(SparseDummyRecurrentNetwork(
                    **params, num_unitwise=self.params.rnn_r_hidden_seq[ii],
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
            }

            self.Network['Dummy'].append(SparseDummyFullyConnected(
                **params, seed=seed, num_unitwise=num_unitwise
            ))
            self.Network['Params'].append(params)

            input_size = self.params.rnn_l_hidden_seq[ii]
            self.Network['Type'].append('mlp')

        final_mlp_size = self.params.embed_size if use_softmax else output_size
        params = {'input_depth': input_size,
              'hidden_size': final_mlp_size,
              'activation_type': None, 'normalizer_type': None,
              'train': True, 'scope': 'mlp' + str(ii),
        }

        num_unitwise = final_mlp_size

        self.Network['Dummy'].append(SparseDummyFullyConnected(
            **params, seed=seed, num_unitwise=num_unitwise
        ))
        self.Network['Type'].append('mlp')
        self.Network['Params'].append(params)

        if use_softmax:
            params = {'hidden_size': output_size,
                      'input_depth': self.params.embed_size,
                      'activation_type': None, 'normalizer_type': None,
                      'train': True, 'scope': 'softmax',
                      }

            self.Network['Dummy'].append(SparseDummyFullyConnected(
                **params, seed=seed, num_unitwise=output_size
            ))
            self.Network['Type'].append('mlp')

            self.Network['Params'].append(params)

        self.initialize_op = []

    def build_sparse(self, sparse_var, use_dense):
        for ix in range(len(sparse_var)):
            if not use_dense[ix]:
                if self.Network['Type'][ix] == 'rnn':
                    self.Network['Dummy'][ix] = SparseRecurrentNetwork(
                        **self.Network['Params'][ix], sparse_list=sparse_var[ix]
                    )

                elif self.Network['Type'][ix] == 'mlp':
                    self.Network['Dummy'][ix] = SparseFullyConnected(
                        **self.Network['Params'][ix], sparse_list=sparse_var[ix]
                    )

                elif self.Network['Type'][ix] == 'embedding':
                    self.Network['Dummy'][ix] = SparseEmbedding(
                        **self.Network['Params'][ix], sparse_list=sparse_var[ix]
                    )
            else:
                if self.Network['Type'][ix] == 'rnn':
                    self.Network['Dummy'][ix] = DenseRecurrentNetwork(
                        **self.Network['Params'][ix], weight=sparse_var[ix]
                    )

                elif self.Network['Type'][ix] == 'mlp':
                    self.Network['Dummy'][ix] = DenseFullyConnected(
                        **self.Network['Params'][ix], weight=sparse_var[ix]
                    )

                elif self.Network['Type'][ix] == 'embedding':
                    self.Network['Dummy'][ix] = DenseEmbedding(
                        **self.Network['Params'][ix], weight=sparse_var[ix]
                    )
            self.initialize_op.append(self.Network['Dummy'][ix].initialize_op)

    def __call__(self, input):
        self.Tensor['Intermediate'] = [None for _ in self.Network['Dummy']]
        for i, network in enumerate(self.Network['Dummy']):
            if self.Network['Type'][i] == 'rnn':
                self.Tensor['Intermediate'][i] = network(input)[0] # get outputs, not hidden state

            else:
                self.Tensor['Intermediate'][i] = network(input)

            input = self.Tensor['Intermediate'][i]
        return self.Tensor['Intermediate'][-1]

    def snip(self, input):
        # gotta fix this for multilayer
        self.Tensor['Temp'] = [None for _ in self.Network['Dummy']]
        for i in range(len(self.Network['Dummy'])):
            if self.Network['Type'][i] == 'rnn':
               input = self.Tensor['Temp'][i] = self.Network['Dummy'][i](input)[0]

            else:
                input = self.Tensor['Temp'][i] = self.Network['Dummy'][i](input)

        return self.Tensor['Temp'][-1]

    def get_dummy_variables(self):
        dummy_weights = []
        for net in self.Network['Dummy']:
            dummy_weights.append(net.weight)
        return dummy_weights
