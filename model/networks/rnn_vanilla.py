import numpy as np
import os.path as osp

from model.networks.base_network import *
from util.vanilla_util import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, output_size, seed, init_path,
            use_embedding=True, use_softmax=True):
        super(RNNModel, self).__init__(params)
        self._npr = np.random.RandomState(seed)

        self.Network['Dummy'] = []
        self.Network['Type'] = []
        ix = 0

        self._input_size = input_size
        params = {'scope':'embed', 'hidden_size': self.params.embed_size,
                  'input_depth': input_size,}

        if init_path is not None:
            load_mat = np.load(osp.join(init_path, '{}.npy'.format(ix)))
        else:
            load_mat = None
        self.Network['Dummy'].append(DenseEmbedding(
            **params, weight=load_mat))
        self.Network['Type'].append('embedding')
        input_size = self.params.embed_size
        ix += 1

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

                if init_path is not None:
                    load_mat = np.load(osp.join(init_path, '{}.npy'.format(ix)))
                else:
                    load_mat = self._npr.normal(
                         size=(input_size + 2* self.params.rnn_r_hidden_seq[ii],
                         4*self.params.rnn_r_hidden_seq[ii])
                    )
                self.Network['Dummy'].append(DenseRecurrentNetwork(
                    **params, weight=load_mat
                ))
                self.Network['Type'].append('rnn')
                ix += 1
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
            if init_path is not None:
                load_mat = np.load(osp.join(init_path, '{}.npy'.format(ix)))
            else:
                load_mat = self._npr.uniform(-.1,.1, (input_size, self.params.rnn_l_hidden_seq[ii]))

            self.Network['Dummy'].append(DenseFullyConnected(
                **params, weight=load_mat
            ))

            self.Network['Type'].append('mlp')
            ix += 1

            input_size = self.params.rnn_l_hidden_seq[ii]

        # final_mlp_size = self.params.embed_size
        # params = {'input_depth': input_size,
        #       'hidden_size': final_mlp_size,
        #       'activation_type': None, 'normalizer_type': None,
        #       'train': True, 'scope': 'mlp' + str(ii),
        # }
        #
        # if init_path is not None:
        #     load_mat = np.load(osp.join(init_path, '{}.npy'.format(ix)))
        # else:
        #     load_mat = self._npr.uniform(-.1, .1, (input_size, final_mlp_size))
        # self.Network['Dummy'].append(DenseFullyConnected(
        #     **params, weight=load_mat
        # ))
        # self.Network['Type'].append('mlp')
        # ix += 1

        params = {'hidden_size': output_size,
                  'input_depth': input_size,
                  'activation_type': None, 'normalizer_type': None,
                  'train': True, 'scope': 'softmax',
                  }

        if init_path is not None:
            load_mat = np.load(osp.join(init_path, '{}.npy'.format(ix)))
        else:
            load_mat = None

        self.Network['Dummy'].append(DenseFullyConnected(
            **params, weight=load_mat
        ))
        self.Network['Type'].append('mlp')

    def __call__(self, input, use_last=True):
        self.Tensor['Intermediate'] = [None for _ in self.Network['Dummy']]
        for i, network in enumerate(self.Network['Dummy']):
            if not use_last and i == len(self.Network['Dummy']) - 1:
                break

            if self.Network['Type'][i] == 'rnn':
                self.Tensor['Intermediate'][i] = network(input)[0] # get outputs, not hidden state

            else:
                self.Tensor['Intermediate'][i] = network(input)

            input = self.Tensor['Intermediate'][i]

        if use_last:
            return self.Tensor['Intermediate'][-1]
        else:
            return self.Tensor['Intermediate'][-2]