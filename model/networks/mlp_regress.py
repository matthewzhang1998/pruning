import tensorflow as tf
from model.networks.base_network import *
from util.sparse_util import *
from util.sparse_ff_util import *

class MLPModel(BaseModel):
    def __init__(self, params, input_size, output_size, seed, init,
            use_embedding=True, use_softmax=True):
        super(MLPModel, self).__init__(params)

        self.Network['Input_Size'] = []
        self.Network['Output_Size'] = []

        self._input_size = input_size

        self.Network['Type'] = []
        self.Network['Params'] = []

        for ii in range(len(self.params.mlp_hidden_seq)):
            params = {'input_depth': input_size,
                      'hidden_size': self.params.mlp_hidden_seq[ii],
                      'activation_type': self.params.mlp_act_seq[ii],
                      'normalizer_type': self.params.mlp_norm_seq[ii],
                      'train': True, 'scope': 'mlp' + str(ii),
                      'num_shards': self.params.num_shards
                      }

            self.Network['Params'].append(params)
            self.Network['Type'].append('mlp')
            input_size = self.params.mlp_hidden_seq[ii]

        params = {'input_depth': input_size,
                  'hidden_size': output_size,
                  'activation_type': 'none',
                  'normalizer_type': 'none',
                  'train': True, 'scope': 'mlp' + str(ii+1),
                  'num_shards': self.params.num_shards
                  }

        self.Network['Params'].append(params)
        self.Network['Type'].append('mlp')

        self.initialize_op = []

        self.Network['Net'] = [None for _ in self.Network['Params']]
        self.Tensor['Output'] = [None for _ in self.Network['Params']] + [None]

    def init_model(self, scope, weights, use_dense=False):
        with tf.variable_scope(scope):
            for i in range(len(weights)):
                self.Network['Net'][i] = SparseFullyConnected(
                    **self.Network['Params'][i], sparse_list=weights[i]
                )

    def __call__(self, input, return_hidden=False):
        self.Tensor['Output'][0] = input

        for i in range(len(self.Network['Net'])):
            input = self.Tensor['Output'][i+1] = self.Network['Net'][i](input)

        if return_hidden:
            return input, self.Tensor['Output']
        else:
            return input

    def hessian_variable(self):
        var = []
        for i in range(len(self.Network['Net'])):
            var += self.Network['Net'][i].var
        return var