import tensorflow as tf
from model.networks.base_network import *
from util.network_util import *

class MLPModel(BaseModel):
    def __init__(self, params, input_size, num_classes, seed, init=None):
        super(MLPModel, self).__init__(params)

        network_shape = [input_size] + self.params.mlp_l_hidden_seq + [num_classes]

        num_layer = len(network_shape) - 1
        act_type = \
            self.params.mlp_l_act_seq + [None]
        norm_type = \
            self.params.mlp_l_norm_seq + [None]

        if init is None:
            init_data = []
            for _ in range(num_layer):
                init_data.append(
                    {'w_init_method': 'xavier',
                     'w_init_para': {'uniform': False},
                     'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
                )
        else:
            init_data = init

        self.Network['Linear'] = MLPWithMask(
            dims=network_shape, scope='mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data, seed=seed
        )

    def __call__(self, input):
        self.Tensor['Predictions'] = self.Network['Linear'](input)

        return self.Tensor['Predictions']

    def weight_variables(self):
        return self.Network['Linear'].weights()

    def get_mask(self):
        return self.Network['Linear'].get_mask()

    def get_weighted_mask(self):
        return self.Network['Linear'].get_weighted_mask()

class MLPTrainMaskModel(BaseModel):
    def __init__(self, params, input_size, num_classes, seed, init=None):
        super(MLPTrainMaskModel, self).__init__(params)

        network_shape = [input_size] + self.params.mlp_l_hidden_seq + [num_classes]

        num_layer = len(network_shape) - 1
        act_type = \
            self.params.mlp_l_act_seq + [None]
        norm_type = \
            self.params.mlp_l_norm_seq + [None]
        if init is None:
            init_data = []
            for _ in range(num_layer):
                init_data.append(
                    {'w_init_method': 'xavier',
                     'w_init_para': {'uniform': False},
                     'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
                )
        else:
            init_data = init

        self.Network['Linear'] = MLPTrainMask(
            dims=network_shape, scope='mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data, seed=seed
        )

    def __call__(self, input):
        self.Tensor['Predictions'] = self.Network['Linear'](input)

        return self.Tensor['Predictions']

    def train_mask(self, input):
        self.Tensor['Train_Pred'] = self.Network['Linear'].mask_train(input)

        return self.Tensor['Train_Pred']

    def get_mask_train(self):
        return self.Network['Linear'].get_mask_train()

    def get_weighted_mask_train(self):
        return self.Network['Linear'].get_weighted_mask_train()

    def weight_variables(self):
        return self.Network['Linear'].weights()

    def get_mask(self):
        return self.Network['Linear'].get_mask()

    def get_weighted_mask(self):
        return self.Network['Linear'].get_weighted_mask()
