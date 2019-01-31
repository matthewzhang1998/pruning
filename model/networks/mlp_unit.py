from model.networks.base_network import *
from util.sparse_util import *

class MLPModel(BaseModel):
    def __init__(self, params, input_size, output_size, seed, init,
            use_embedding=True, use_softmax=True):
        super(MLPModel, self).__init__(params)

        self.Network['Dummy'] = []
        self.Network['Input_Size'] = []
        self.Network['Output_Size'] = []

        self._input_size = input_size

        self.Network['Type'] = []
        self.Network['Params'] = []

        for ii in range(len(self.params.mlp_l_hidden_seq)):
            params = {'input_depth': input_size,
              'hidden_size': self.params.mlp_l_hidden_seq[ii],
              'activation_type': self.params.mlp_l_act_seq[ii],
              'normalizer_type': self.params.mlp_l_norm_seq[ii],
              'train': True, 'scope': 'mlp' + str(ii),
            }

            num_unitwise = min(self.params.num_unitwise_mlp, self.params.mlp_l_hidden_seq[ii])

            self.Network['Dummy'].append(SparseDummyFullyConnected(
                **params, num_unitwise=num_unitwise,
            ))
            self.Network['Params'].append(params)
            self.Network['Type'].append('mlp')
            input_size = self.params.mlp_l_hidden_seq[ii]

        params = {'input_depth': input_size,
              'hidden_size': output_size,
              'activation_type': None, 'normalizer_type': None,
              'train': True, 'scope': 'mlp' + str(ii+1),
        }

        num_unitwise = min(self.params.num_unitwise_mlp, output_size)

        self.Network['Dummy'].append(
            SparseDummyFullyConnected(
                **params, seed=seed, num_unitwise=num_unitwise,
            )
        )

        print(self.Network['Dummy'])

        self.Network['Type'].append('mlp')
        self.Network['Params'].append(params)

        self.initialize_op = []
        self.Tensor['Temp'] = [None for _ in self.Network['Dummy']]
        self.Tensor['Weights'] = [None for _ in self.Network['Dummy']]

    def build_sparse(self, sparse_var, ii, use_dense):
        if self.Network['Type'][ii] == 'rnn':
            self.Network['Dummy'][ii] = SparseRecurrentNetwork(
                **self.Network['Params'][ii], sparse_list=sparse_var
            )

        elif self.Network['Type'][ii] == 'mlp':
            self.Network['Dummy'][ii] = SparseFullyConnected(
                **self.Network['Params'][ii], sparse_list=sparse_var
            )

        elif self.Network['Type'][ii] == 'embedding':
            self.Network['Dummy'][ii] = SparseEmbedding(
                **self.Network['Params'][ii], sparse_list=sparse_var
            )
        self.initialize_op = self.Network['Dummy'][ii].initialize_op

    def __call__(self, input):
        self.Tensor['Intermediate'] = [None for _ in self.Network['Dummy']]
        for i, network in enumerate(self.Network['Dummy']):
            if self.Network['Type'][i] == 'rnn':
                self.Tensor['Intermediate'][i] = network(input)[0] # get outputs, not hidden state

            else:
                self.Tensor['Intermediate'][i] = network(input)

            self.Tensor['Weights'][i] = network.var

            input = self.Tensor['Intermediate'][i]
        return self.Tensor['Intermediate'][-1]

    def unit(self, input, ii):
        # gotta fix this for multilayer
        self.Tensor['Temp'][ii] = [None for _ in self.Network['Dummy']]
        for j in range(ii):
            if self.Network['Type'][j] == 'rnn':
                input = self.Tensor['Temp'][ii][j] = self.Network['Dummy'][j].sample(input)[0]

            else:
                input = self.Tensor['Temp'][ii][j] = self.Network['Dummy'][j].sample(input)

        if self.Network['Type'][ii] == 'rnn':
            input = self.Tensor['Temp'][ii][ii] = self.Network['Dummy'][ii](input)[0]

        else:
            input = self.Tensor['Temp'][ii][ii] = self.Network['Dummy'][ii](input)

        for k in range(ii+1,len(self.Network['Dummy'])):
            if self.Network['Type'][k] == 'rnn':
                input = self.Tensor['Temp'][ii][k] = self.Network['Dummy'][k](input)[0]

            else:
                input = self.Tensor['Temp'][ii][k] = self.Network['Dummy'][k](input)

        return self.Tensor['Temp'][ii][-1]

    def get_dummy_variables(self):
        dummy_weights = []
        for net in self.Network['Dummy']:
            dummy_weights.append(net.weight)
        return dummy_weights

    def get_roll_variables(self):
        dummy_roll = []
        for ix,net in enumerate(self.Network['Dummy']):
            if self.Network['Type'][ix] == 'mlp':
                dummy_roll.append(net.roll)

            else:
                dummy_roll.append(None)
        return dummy_roll
