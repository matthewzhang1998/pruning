from util.network_util import *
import tensorflow as tf
from model.networks.base_network import *

class RNNModel(BaseModel):
    def __init__(self, params, input_size, num_classes, seed, init):
        super(RNNModel, self).__init__(params)

        self.Network['Recurrent'] = []

        for ii in range(len(self.params.rnn_r_hidden_seq)):
            init_data = \
                {'w_init_method': 'normc', 'w_init_para': {'stddev':1.0}}

            if self.params.rnn_bidirectional:
                pass
            elif self.params.rnn_dilated:
                pass
            else:
                self.Network['Recurrent'].append(Recurrent_Network_with_mask(
                    scope='rnn'+str(ii), activation_type=self.params.rnn_r_act_seq[ii],
                    normalizer_type=self.params.rnn_r_norm_seq[ii],
                    recurrent_cell_type=self.params.rnn_cell_type,
                    train=True, hidden_size=self.params.rnn_r_hidden_seq[ii],
                    input_depth=input_size, init_data=init_data
                ))
                input_size = self.params.rnn_r_hidden_seq[ii]

        network_shape = [self.params.rnn_r_hidden_seq[-1]] + \
            self.params.rnn_l_hidden_seq + [num_classes]

        num_layer = len(network_shape) - 1
        act_type = \
            self.params.rnn_l_act_seq + [None]
        norm_type = \
            self.params.rnn_l_norm_seq + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'xavier', 'w_init_para': {'uniform': False},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        self.Network['Linear'] = MLPWithMask(
            dims=network_shape, scope='mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data, seed=seed
        )

    def __call__(self, input):
        self.Tensor['Intermediate'] = [None for _ in self.Network['Recurrent']]
        for i, network in enumerate(self.Network['Recurrent']):
            self.Tensor['Intermediate'][i] = network(input)[0]
            input = self.Tensor['Intermediate'][i]
        self.Tensor['Predictions'] = self.Network['Linear'](self.Tensor['Intermediate'][-1])
        return self.Tensor['Predictions']

    def weight_variables(self):
        weights = []
        for net in self.Network['Recurrent']:
            weights += net.weights()
        return weights + self.Network['Linear'].weights()

    def get_mask(self):
        mask = []
        for net in self.Network['Recurrent']:
            mask += net.get_mask()
        return mask + self.Network['Linear'].get_mask()

    def get_weighted_mask(self):
        weighted_mask = []
        for net in self.Network['Recurrent']:
            weighted_mask += net.get_weighted_mask()

        return weighted_mask + self.Network['Linear'].get_weighted_mask()




