import os.path as osp

import numpy as np
import tensorflow as tf
from model.drw import *
from runner.base_runner import *
from util.optimizer_util import *
from util.logger_util import *
from model.snip import *
from util.sparse_util import *
import scipy.misc
from collections import defaultdict

from tensorflow.contrib import slim

from data.load_pen import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

class DRWRunner(BaseRunner):
    def _build_snip(self):
        with tf.variable_scope(self.scope):
            self.Model['DRW'] = DRW('drw', self.params,
                self.vocab_size, self.vocab_size, init_path='../weights/rnn')

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

            self.Tensor['DRW_Theta'] = self.Model['DRW'].Snip['Theta']
            self.Placeholder['DRW_Mask'] = self.Model['DRW'].Snip['Mask']
            self.Placeholder['DRW_Newval'] = [
                tf.placeholder(dtype=tf.float32, shape=weight.get_shape())
                for weight in self.Tensor['DRW_Theta']
            ]
            self.Tensor['DRW_Assign'] = [
                tf.assign(theta, new_val) for (theta, new_val)
                in zip(self.Tensor['DRW_Theta'], self.Placeholder['DRW_Newval'])
            ]

            self.Output['DRW_Pred'] = self.Model['DRW'].run(
                self.Placeholder['Input_Feature']
            )

            self.Output['DRW_Loss'] = tf.reduce_mean(
               self.Tensor['Loss_Function'](
                   self.Output['DRW_Pred'], self.Placeholder['Input_Label']
               )
            )

            self.Output['DRW_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['DRW_Loss'])

    def _preprocess(self):
        self.Sess.run(tf.global_variables_initializer())
        weights = self.Sess.run(self.Tensor['DRW_Theta'])
        masks = []
        for weight in weights:
            rand_num= int((1-self.params.drw_k)*weight.size)
            rand_ix = np.unravel_index(
                self._npr.choice(np.arange(weight.size), size=(rand_num,)),
                weight.shape
            )
            mask = np.zeros_like(weight)
            mask[rand_ix] = 1
            masks.append(mask)

        self.Mask['DRW'] = {
            self.Placeholder['DRW_Mask'][ix]: masks[ix]
            for ix in range(len(masks))
        }

    def train(self, i, features, labels):
        self.drw_weight()
        super(DRWRunner, self).__train__()

    def drw_weight(self):
        weights = self.Sess.run(self.Tensor['DRW_Theta'])

        new_weights = []
        new_masks = []

        for ix,weight in enumerate(weights):
            weight -= self.learning_rate * self.params.weight_decay
            weight += np.sqrt(2*self.learning_rate*self.params.drw_temperature) * \
                self._npr.normal(loc=0, scale=1, size=weight.shape)

            weight = np.clip(weight, 0, None)
            new_weights.append(weight)
            new_mask = self.Mask['DRW'][self.Placeholder['DRW_Mask'][ix]]
            new_mask[np.where(weight == 0)] = 0

            rand_num = int((1 - self.params.drw_k) * weight.size)
            real_num = np.count_nonzero(weight)

            zero_num = weight.size - real_num
            zero_inds = np.where(weight == 0)
            if rand_num > real_num:
                new_x = self._npr.choice(zero_num, size=(rand_num - real_num,))

                new_inds = [ind[new_x] for ind in zero_inds]
                new_mask[new_inds] = 1

            new_masks.append(new_mask)

        feed_dict = {
            self.Placeholder['DRW_Newval'][ix]: new_weights[ix]
            for ix in range(len(new_weights))
        }

        self.Sess.run(
            self.Tensor['DRW_Assign'], feed_dict
        )

        self.Mask['DRW'] = {
            self.Placeholder['DRW_Mask'][ix]: new_masks[ix]
            for ix in range(len(new_masks))
        }