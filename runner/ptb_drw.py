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

class PTBRunner(BaseRunner):
    def __init__(self, scope, params):
        super(PTBRunner, self).__init__(scope, params)

        self._npr = np.random.RandomState(params.seed)
        self.Mask = {}
        self.Data = Dataset(self.params, "../data/ptb/data")
        self.vocab_size = self.Data.vocab_size
        self._build_snip()
        self._build_summary()
        self._preprocess()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Writer['DRW'] = \
            FileWriter(self.Dir+'/drw', self.Sess.graph)

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

    def _build_summary(self):
        self.Output['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Placeholder['Input_Logits'],
                self.Placeholder['Input_Label']
            )
        )

        self.Placeholder['Val_Error'] = tf.placeholder(
            dtype=tf.float32, shape=[]
        )
        self.Placeholder['Val_Loss'] = tf.placeholder(
            dtype=tf.float32, shape=[]
        )

        self.Output['Error'] = tf.exp(self.Output['Loss'])

        self.val_res = {
            'Val_Error': self.Output['Error'],
            'Val_Loss': self.Output['Loss']
        }

        self.val_placeholder = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': self.Placeholder['Val_Loss']
        }

        self.Summary['Train_Error'] = tf.summary.scalar(
            'Train_Error', self.Output['Error']
        )
        self.Summary['Val_Error'] = tf.summary.scalar(
            'Val_Error', self.Placeholder['Val_Error']
        )

        self.Summary['Train_Loss'] = tf.summary.scalar(
            'Train_Loss', tf.log(self.Output['Loss'])
        )
        self.Summary['Val_Loss'] = tf.summary.scalar(
            'Val_Loss', tf.log(self.Placeholder['Val_Loss'])
        )

        #self.Summary['Weight'] = {}
        #for key in ['Random', 'Unit']:
        #    self.Summary['Weight'][key] = [
        #        tf.summary.histogram(weight.name,
        #            tf.boolean_mask(weight, tf.not_equal(weight, ZERO_32)))
        #        for weight in self.Model.Sparse
        #    ]

        self.Output['Pred'] = {
            'DRW': self.Output['DRW_Pred'],
        }

        self.train_summary = {
            'Train_Error': self.Output['Error'],
            'Train_Loss': self.Output['Loss']
        }
        self.val_summary = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': self.Placeholder['Val_Loss']
        }
        self.train_op = [
            self.Output['DRW_Train']
        ]

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
        # self.Dataset.train.next_batch(self.params.batch_size)
        # print(features, labels)

        feed_dict = {
            **self.Mask['DRW'],
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        pred, *_ = self.Sess.run(
            [self.Output['Pred']] + self.train_op,
            feed_dict
        )

        for key in pred:
            summary = self.Sess.run(
                self.train_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )

            self.Writer[key].add_summary(summary, i)

        self.drw_weight()

        self.learning_rate = self.decay_lr(i, self.learning_rate)
        return features, labels

    def val(self, i):
        start = 0
        summary = {'DRW': defaultdict(list)}

        for k in range(self.params.val_size):
            end = start + self.params.batch_size

            b_feat, b_lab = self._get_batch('val')
            # self.Dataset.test.images, self.Dataset.test.labels

            feed_dict = {
                **self.Mask['DRW'],
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab,
            }
            pred = self.Sess.run(
                [self.Output['Pred']], feed_dict)

            pred = pred[0]
            for key in pred:
                b_summary = self.Sess.run(
                    self.val_res,
                    {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
                )

                for summ in b_summary:
                    summary[key][summ].append(b_summary[summ])

        for key in summary:
            for summ in summary[key]:
                summary[key][summ] = np.mean(summary[key][summ])

            write_summary = self.Sess.run(
                self.val_summary,
                {self.val_placeholder[summ]: summary[key][summ]
                 for summ in summary[key]}
            )
            self.Writer[key].add_summary(write_summary, i)

    def run(self):
        #slim.model_analyzer.analyze_vars(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), print_info=True)

        for e in range(self.params.num_steps):
            features, labels = self._get_batch()
            self.train(e, features, labels)
            if e % self.params.val_steps == 0:
                self.val(e)

    def decay_lr(self, i, learning_rate):
        if self.params.decay_scheme == 'exponential':
            if (i+1) % self.params.decay_iter == 0:
                learning_rate *= self.params.decay_rate

        elif self.params.decay_scheme == 'none':
            pass

        return learning_rate

    def _get_batch(self, type='train'):
        return self.Data.get_batch(type)

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