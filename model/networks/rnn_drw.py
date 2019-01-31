import os.path as osp
import numpy as np

from model.networks.base_network import *
from util.drw_util import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, output_size, seed, init_path):
        super(RNNModel, self).__init__(params)

        self.Network['DRW'] = []
        self.Network['Type'] = []
        self._input_size = input_size
        self.network_ix = 0
        params = {'scope':'embed', 'hidden_size': self.params.embed_size,
                  'input_depth': input_size,}

        load_mat = np.load(osp.join(init_path, '{}.npy'.format(self.network_ix)))

        self.Network['DRW'].append(DRWEmbedding(
            **params, seed=seed, init_matrix=load_mat))

        self.Network['Type'].append('embedding')
        self.network_ix += 1

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

                load_mat = np.load(osp.join(init_path, '{}.npy'.format(self.network_ix)))
                self.Network['DRW'].append(DRWRecurrentNetwork(
                    **params, init_matrix=load_mat
                ))
                self.network_ix += 1
                self.Network['Type'].append('rnn')
            input_size = self.params.rnn_r_hidden_seq[ii]

        for ii in range(len(self.params.rnn_l_hidden_seq)):
            act_type = \
                self.params.rnn_l_act_seq[ii]
            norm_type = \
                self.params.rnn_l_norm_seq[ii]

            params = {'input_depth': input_size,
                'hidden_size': self.params.rnn_l_hidden_seq[ii],
                'activation_type': act_type, 'normalizer_type': norm_type,
                'train':True, 'scope':'mlp'+str(ii),
            }
            load_mat = np.load(osp.join(init_path, '{}.npy'.format(self.network_ix)))

            self.Network['DRW'].append(DRWFullyConnected(
                **params, init_matrix=load_mat
            ))
            input_size = self.params.rnn_l_hidden_seq[ii]
            self.network_ix += 1

            self.Network['Type'].append('mlp')

        final_mlp_size = self.params.embed_size
        params = {'input_depth': input_size,
              'hidden_size': final_mlp_size,
              'activation_type': None, 'normalizer_type': None,
              'train': True, 'scope': 'mlp' + str(ii),
        }

        load_mat = np.load(osp.join(init_path, '{}.npy'.format(self.network_ix)))

        self.Network['DRW'].append(DRWFullyConnected(
            **params, init_matrix=load_mat
        ))
        self.network_ix += 1

        params = {'hidden_size': output_size,
                  'input_depth': self.params.embed_size,
                  'activation_type': None, 'normalizer_type': None,
                  'train': True, 'scope': 'softmax',
                  }

        self.Network['Type'].append('mlp')

        load_mat = np.load(osp.join(init_path, '{}.npy'.format(self.network_ix)))
        self.Network['DRW'].append(DRWFullyConnected(
            **params, init_matrix=load_mat
        ))
        self.network_ix += 1

        self.Network['Type'].append('mlp')

    def __call__(self, input):
        self.Tensor['Intermediate'] = [None for _ in self.Network['DRW']]
        for i, network in enumerate(self.Network['DRW']):
            if self.Network['Type'][i] == 'rnn':
                self.Tensor['Intermediate'][i] = network(input)[0] # get outputs, not hidden state

            else:
                self.Tensor['Intermediate'][i] = network(input)

            input = self.Tensor['Intermediate'][i]
        return self.Tensor['Intermediate'][-1]

    def get_mask(self):
        mask = []
        for ix, net in enumerate(self.Network['DRW']):
            if self.Network['Type'][ix] == 'rnn':
                mask.append(net.mask())
        return mask

    def get_theta(self):
        theta = []
        for ix, net in enumerate(self.Network['DRW']):
            if self.Network['Type'][ix] == 'rnn':
                theta.append(net.theta())
        return theta
