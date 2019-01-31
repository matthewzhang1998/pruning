import tensorflow as tf
from model.networks.base_network import *
from util.network_util import *

class MLPModel(BaseModel):
    def __init__(self, params, input_size, num_classes, seed):
        super(MLPModel, self).__init__(params)

        network_shape = [input_size] + self.params.mlp_l_hidden_seq + [num_classes]

        num_layer = len(network_shape) - 1
        act_type = \
            self.params.mlp_l_act_seq + [None]
        norm_type = \
            self.params.mlp_l_norm_seq + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )

        init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std

        self.Network['Linear'] = MLP(
            dims=network_shape, scope='mlp',
            activation_type=act_type, normalizer_type=norm_type,
            train=True, init_data=init_data,
            seed=seed
        )

    def set_weighted_params(self, weight_dict, scope):
        self.Mask[scope] = {}
        assigns = []

        for w in weight_dict:
            self.Mask[scope][w] = tf.Variable(tf.zeros_like(weight_dict[w]), False)
            assign_op = tf.assign(self.Mask[scope][w], weight_dict[w])
            assigns.append(assign_op)

        self.Network['Linear'].assign_masks(self.Mask[scope], scope)

        return assigns

    def weighted(self, input, scope):
        assert self.Mask[scope] is not None

        self.Tensor['Predictions'] = self.Network['Linear'](input, scope)

        return self.Tensor['Predictions']

    def weight_variables(self):
        return self.Network['Linear'].weights()
