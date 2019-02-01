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

class PTBRunner(BaseRunner):
    def __init__(self, scope, params):
        super(PTBRunner, self).__init__(scope, params)

        self._npr = np.random.RandomState(params.seed)
        self.Mask = {}
        self.Data = Dataset(params, "../data/ptb/data")
        self.vocab_size = self.Data.vocab_size
        self._build_snip()

        self._preprocess()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Writer['Snip'] = \
            FileWriter(self.Dir+'/snip', self.Sess.graph)
        self.Writer['L2'] = \
            FileWriter(self.Dir+'/l2', self.Sess.graph)

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
            'Train_Loss', self.Output['Loss']
        )
        self.Summary['Val_Loss'] = tf.summary.scalar(
            'Val_Loss', self.Placeholder['Val_Loss']
        )

        #self.Summary['Weight'] = {}
        #for key in ['Random', 'Unit']:
        #    self.Summary['Weight'][key] = [
        #        tf.summary.histogram(weight.name,
        #            tf.boolean_mask(weight, tf.not_equal(weight, ZERO_32)))
        #        for weight in self.Model.Sparse
        #    ]

        self.Output['Pred'] = {
            'Snip': self.Output['Snip_Pred'],
            'L2': self.Output['L2_Pred']
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
            self.Output['Snip_Train'],
            self.Output['L2_Train']
        ]

    def train(self, i, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)
        # print(features, labels)

        print(i)

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        pred, *_ = self.Sess.run(
            [self.Output['Pred']] + self.train_op,
            feed_dict
        )
        #self.Writer['Unit'].add_run_metadata(self.Sess.rmd, 'train' + str(i))
        for key in pred:
            summary = self.Sess.run(
                self.train_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )

            self.Writer[key].add_summary(summary, i)

        self.learning_rate = self.decay_lr(i, self.learning_rate)
        return features, labels

    def val(self, i):
        start = 0
        summary = {'L2': defaultdict(list),'Snip': defaultdict(list)}

        for k in range(self.params.val_size):
            end = start + self.params.batch_size

            b_feat, b_lab = self._get_batch('val')
            # self.Dataset.test.images, self.Dataset.test.labels

            feed_dict = {
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

    def prune_together(self, weights, grads, sparse, return_dense_arr):
        total_shape, total_size, flat_grad = [], [], []
        for ix in range(len(grads)):
            grad = grads[ix]
            total_shape.append(grad.shape)
            total_size.append(grad.size)
            flat_grad.append(grad.flatten())

        flat_size = 1
        for size in total_size:
            flat_size += size

        flat_grad = np.concatenate(flat_grad, axis=0)

        k = int((1 - sparse) * flat_size)

        zeros = np.zeros_like(flat_grad)

        ind = np.argpartition(np.abs(flat_grad), -k, axis=None)[-k:]

        zeros[ind] = 1
        start = 0

        return_arr = []
        for ix, (shape, size) in enumerate(zip(total_shape, total_size)):
            end = start + size
            mask = np.reshape(zeros[start:end], shape)
            if not return_dense_arr[ix]:
                inds = np.nonzero(mask)
                w = weights[ix][inds]
                inds = np.vstack(inds).T
                return_arr.append([w, inds])

            else:
                return_arr.append(np.reshape(mask, shape)*weights[ix])

            start = end
        return return_arr