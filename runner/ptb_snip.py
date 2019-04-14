import os.path as osp

import numpy as np
import tensorflow as tf
from model.mask import *
from runner.base_runner import *
from util.optimizer_util import *
from model.snip import *
from util.logger_util import *
from util.sparse_util import *
import scipy.misc
from collections import defaultdict

from tensorflow.contrib import slim

from data.load_pen import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

class SnipRunner(BaseRunner):
    def _build_snip(self):
        with tf.variable_scope(self.scope):
            self.Model['Snip'] = Snip('snip', self.params,
                self.vocab_size, self.vocab_size)
            self.Model['L2'] = Snip('l2', self.params,
                self.vocab_size, self.vocab_size)

            self.start_ix = 0

            self.Placeholder['Input_Feature'] = tf.placeholder(
                shape=[None, None], dtype=tf.int32,
            )

            self.Placeholder['Learning_Rate'] = tf.placeholder(
                tf.float32, []
            )

            self.Placeholder['Input_Label'] = tf.placeholder(
                tf.int32, [None, None]
            )

            self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                [None, None, self.vocab_size])

            self.Tensor['Proto_Minibatch'] = {
                'Features': self.Placeholder['Input_Feature'],
                'Labels': self.Placeholder['Input_Label']
            }

            self.Tensor['Loss_Function'] = \
                Seq2SeqLoss

            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

            self.Model['Snip'].snip(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function']
            )

            self.Tensor['Snip_Grad'] = self.Model['Snip'].Tensor['Snip_Grad']

            self.Placeholder['Snip_Kernel'] = self.Model['Snip'].Snip['Dummy_Kernel']

            self.Tensor['Variable_Initializer'] = {}

    def _preprocess(self):
        self.Sess.run(tf.global_variables_initializer())
        features, labels = self._get_batch()
        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels
        }
        weights = []
        for ix, kernel in enumerate(self.Placeholder['Snip_Kernel']):
            feed_dict[kernel] = weight = np.load(osp.join('../weights/rnn/{}.npy'.format(ix)))
            weights.append(weight)

        grads = self.Sess.run(self.Tensor['Snip_Grad'], feed_dict)
        k = len(self.params.rnn_r_hidden_seq)
        dense = [True for _ in range(k)]
        snip_weights = self.prune_together(weights[1:k+1], grads[1:k+1], self.params.snip_k, dense)
        snip_weights = [weights[0]] + snip_weights + weights[k+1:]
        use_dense = [True for _ in grads]

        l2_weights = self.prune_together(weights[1:k+1], weights[1:k+1], self.params.l2_k, dense)
        l2_weights = [weights[0]] + l2_weights + weights[k+1:]
        self._build_networks(snip_weights, l2_weights, use_dense)
        self._build_summary()
        self.Sess.run(self.Tensor['Variable_Initializer'])
        self.Sess.run(tf.variables_initializer(self.Output['Optimizer'].variables()))

    def _build_networks(self, snip_list, l2_list, use_dense=None):
        self.Model['Snip'].build_sparse(snip_list, use_dense=use_dense)
        self.Model['L2'].build_sparse(l2_list, use_dense=use_dense)

        self.Tensor['Variable_Initializer'] = {
            'L2': self.Model['L2'].initialize_op,
            'Snip': self.Model['Snip'].initialize_op,
        }

        self.Output['L2_Pred'] = self.Model['L2'].run(
            self.Placeholder['Input_Feature']
        )

        self.Output['Snip_Pred'] = self.Model['Snip'].run(
            self.Placeholder['Input_Feature']
        )

        self.Output['L2_Loss'] = tf.reduce_mean(
           self.Tensor['Loss_Function'](
               self.Output['L2_Pred'], self.Placeholder['Input_Label']
           )
        )

        self.Output['Snip_Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Output['Snip_Pred'], self.Placeholder['Input_Label']
            )
        )
        self.Output['L2_Train'] = \
           self.Output['Optimizer'].minimize(self.Output['L2_Loss'])
        self.Output['Snip_Train'] = \
            self.Output['Optimizer'].minimize(self.Output['Snip_Loss'])