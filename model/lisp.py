import tensorflow as tf
from model.base import *
from model.networks.mlp_norm import *

class Lisp(object):
    def __init__(self, scope, params, input_size, num_classes):
        self.params = params
        self.Snip = {}
        self.Tensor = {}
        self.Op = {}

        with tf.variable_scope(scope):
            self.model = get_model(params.model_type)(
                params, input_size, num_classes
            )

            self.Snip['Weight'] = self.model.weight_variables()
            self.Snip['Mask'] = self.model.get_mask()
            self.Snip['Comb'] = self.model.get_weighted_mask()
            self.Tensor['Proxy'] = self.model.proxy_input()
            self.Tensor['Predicted'] = self.model.proxy_output()

    def prune_backward(self, labels, loss_fnc1, loss_fnc2):
        '''

        :param minibatch: Batch of samples
        :param loss_fnc1: Loss function for global snip
        :param loss_fnc2: Loss function for local approx.
        :return:
        '''
        self.Tensor['Layer_Output'], self.Tensor['Final_Output'] = \
            self.model.layerwise()

        self.Tensor['Back_Grad'] = []
        for ix, param in enumerate(self.Snip[self.params.grad_param]):
            loss = loss_fnc1(
                self.Tensor['Final_Output'][ix], labels
            )

            if ix != len(self.Snip['Weight']) - 1:
                loss += loss_fnc2(
                    self.Tensor['Layer_Output'][ix],
                    self.Tensor['Predicted'][ix]
                )
            else:
                loss += loss_fnc2(
                    tf.nn.softmax(self.Tensor['Layer_Output'][ix]),
                    self.Tensor['Predicted'][ix]
                )

            grad = tf.gradients(loss, param)
            self.Tensor['Back_Grad'].append(grad)
        return self.Tensor['Back_Grad']

    def run(self, features):
        self.Tensor['Back_Predictions'] = \
            self.model(features)
        return self.Tensor['Back_Predictions']

    def prune(self, minibatch, loss_fnc):
        self.Tensor['Lisp_Pred'] = self.model(
            minibatch['Features']
        )

        self.Tensor['Lisp_Loss'] = tf.reduce_mean(loss_fnc(
            self.Tensor['Lisp_Pred'], minibatch['Labels']
        ))

        self.Tensor['Lisp_True_Grad'] = tf.gradients(
            self.Tensor['Lisp_Loss'], self.Snip[self.params.grad_param])
        ratio = tf.constant(1-self.params.prune_k, tf.float32)

def get_model(name):
    name = name.lower()
    if name == 'mlp':
        return MLPNormModel

    elif name == 'rnn':
        return RNNLayerNormModel

    elif name == 'cnn':
        return CNNLayerNormModel