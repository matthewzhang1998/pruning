import tensorflow as tf
from model.networks.base_network import *
from util.network_util import *

class MLPNormModel(BaseModel):
    def __init__(self, params, input_size, num_classes):
        super(MLPNormModel, self).__init__(params)

        network_shape = [input_size] + self.params.mlp_l_hidden_seq + [num_classes]
        num_layer = len(network_shape) - 1
        act_type = \
            self.params.mlp_l_act_seq + [None]
        norm_type = \
            self.params.mlp_l_norm_seq + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'xavier',
                 'w_init_para': {'uniform': False},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )

        self.Network['Linear'] = MLPLayerWise(
            dims=network_shape, scope='mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data
        )

    def __call__(self, input):
        self.Tensor['Predictions'] = self.Network['Linear'](input)

        return self.Tensor['Predictions']

    def layerwise(self):
        self.Tensor['Final'], self.Tensor['Current'] = [], []

        for i in range(self.Network['Linear'].num_layer):
            current, final = \
                self.Network['Linear'].run_backward(i)
            self.Tensor['Final'].append(final)
            self.Tensor['Current'].append(current)

        return self.Tensor['Current'], self.Tensor['Final']

    def weight_variables(self):
        return self.Network['Linear'].weights()

    def get_mask(self):
        return self.Network['Linear'].get_mask()

    def get_weighted_mask(self):
        return self.Network['Linear'].get_weighted_mask()

    def proxy_input(self):
        return self.Network['Linear'].get_proxy_input()

    def proxy_output(self):
        return self.Network['Linear'].get_proxy_output()

class BiMLPNormModel(BaseModel):
    def __init__(self, params, input_size, num_classes):
        super(BiMLPNormModel, self).__init__(params)

        network_shape = [input_size] + self.params.mlp_l_hidden_seq + [num_classes]
        num_layer = len(network_shape) - 1
        act_type = \
            self.params.mlp_l_act_seq + [None]
        norm_type = \
            [None] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'bradly',
                 'w_init_para': {'alpha':0.01},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )

        self.Network['Linear'] = MLPLayerWise(
            dims=network_shape, scope='mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data
        )

        self.Network['Fore'] = MLPCopy(self.Network['Linear'], 'fore')

    def __call__(self, input):
        self.Tensor['Back_Predictions'] = self.Network['Linear'](input)
        self.Tensor['Fore_Predictions'] = self.Network['Fore'](input)

        return self.Tensor['Back_Predictions'], \
            self.Tensor['Fore_Predictions']

    def layerwise_back(self):
        self.Tensor['Final'], self.Tensor['Current'] = [], []

        for i in range(self.Network['Linear'].num_layer):
            current, final = \
                self.Network['Linear'].run_backward(i)
            self.Tensor['Final'].append(final)
            self.Tensor['Current'].append(current)

        return self.Tensor['Current'], self.Tensor['Final']

    def layerwise_fore(self, x):
        self.Tensor['Final'], self.Tensor['Current'] = [], []

        for i in range(self.Network['Fore'].num_layer):
            current, final = \
                self.Network['Fore'].run_forward(i, x)
            self.Tensor['Final'].append(final)
            self.Tensor['Current'].append(current)

        return self.Tensor['Current'], self.Tensor['Final']

    def weight_variables(self):
        return {'Lisp': self.Network['Linear'].weights(),
            'Fore': self.Network['Fore'].weights()}

    def get_mask(self):
        return {'Lisp': self.Network['Linear'].get_mask(),
            'Fore': self.Network['Fore'].get_mask()}

    def get_weighted_mask(self):
        return {'Lisp': self.Network['Linear'].get_weighted_mask(),
            'Fore': self.Network['Fore'].get_weighted_mask()}

    def proxy_input(self):
        return self.Network['Linear'].get_proxy_input()

    def proxy_output(self):
        return self.Network['Linear'].get_proxy_output()


