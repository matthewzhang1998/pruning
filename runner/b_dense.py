import os.path as osp
import sys

import numpy as np
import tensorflow as tf
from runner.base_runner import *
from util.optimizer_util import *
from model.dense import *
from model.unit_hess import *

import scipy.misc
from util.logger_util import *
from util.initializer_util import *
from collections import defaultdict

import matplotlib.pyplot as plt

from data.load_pen import *

_EPS = 0.1

class DenseRunner(LanguageRunner):
    def _build_snip(self):
        self.params.num_unitwise_rnn = self.params.rnn_r_hidden_seq[-1]

        if self.params.dataset in ['timit', 'seq_mnist']:
            self.Model['Regress'] = Dense('regress', self.params,
                self.vocab_size, self.output_size, self.params.seed)

        else:
            self.Model['Regress'] = Dense('regress', self.params,
                self.vocab_size, self.vocab_size, self.params.seed)

        self.start_ix = 0

        self._preprocess()
        sys.stdout.flush()

    def _build_base_graph(self):
        with tf.variable_scope(self.scope):
            if self.params.dataset in ['timit', 'seq_mnist']:
                self.Placeholder['Input_Feature'] = tf.placeholder(
                    shape=[None, self.vocab_size, self.vocab_size], dtype=tf.float32,
                )

            else:
                self.Placeholder['Input_Feature'] = tf.placeholder(
                    shape=[None, self.params.max_length], dtype=tf.int32,
                )

            self.Placeholder['Learning_Rate'] = tf.placeholder(
                tf.float32, []
            )

            if self.params.dataset in ['timit', 'seq_mnist']:
                self.Placeholder['Input_Label'] = tf.placeholder(
                    tf.int32, [None, self.output_size]
                )

                self.Tensor['Loss_Function'] = SoftmaxCE

            elif not self.params.use_sample_softmax:
                self.Placeholder['Input_Label'] = tf.placeholder(
                    tf.int32, [None, self.params.max_length]
                )

                self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                                                                  [None, self.params.max_length, self.vocab_size])

                self.Tensor['Loss_Function'] = \
                    Seq2SeqLoss

            else:
                self.Placeholder['Input_Label'] = tf.placeholder(
                    tf.int32, [None, self.params.max_length]
                )

                self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                                                                  [None, self.params.max_length,
                                                                   self.params.rnn_r_hidden_seq[-1]])

                if self.params.use_factor_softmax:
                    self.Tensor['Loss_Function'] = \
                        SSSLossWithFactorization

                else:
                    self.Tensor['Loss_Function'] = \
                        SampleSoftmaxSequenceLoss

            self.Tensor['Proto_Minibatch'] = {
                'Features': self.Placeholder['Input_Feature'],
                'Labels': self.Placeholder['Input_Label']
            }

            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

    def _preprocess(self):
        import itertools
        self.eval_ix = 0
        self.Tensor['Data'] = list(itertools.islice(self._get_batch(),
            max(self.params.eval_iter, self.params.test_iter)))

        info = self.Model['Regress'].Info['Params']
        type = self.Model['Regress'].Info['Type']

        #### EMBEDDING ####
        nh = info[0]['hidden_size']
        ni = info[0]['input_depth']

        self.embed_matrix = get_init(self.params.rnn_init_type)(
                (ni, nh), self._npr, self.params.rnn_init_scale
            )

        #### SOFTMAX ####
        nh = info[-1]['hidden_size']
        ni = info[-1]['input_depth']

        self.softmax_matrix = get_init(self.params.rnn_init_type)(
                (ni, nh), self._npr, self.params.rnn_init_scale
            )

        self.Model['Regress'].set_embed_and_softmax(
            self.embed_matrix, self.softmax_matrix, use_dense=True
        )

        #### LSTM ####
        nh = self.nh = info[1]['hidden_size']
        ni = self.ni = info[1]['input_depth']

        nu = self.params.num_unitwise_rnn
        assert (nh == nu)

        nb = self.params.batch_size
        self.sess = tf.Session()

        nj = 1
        if self.params.rnn_cell_type == 'gru':
            no = 3 * nu
            ng = 3
            ns = self.ns = 1

        elif self.params.rnn_cell_type == 'lstm':
            no = 4 * nu
            ng = 4
            ns = self.ns = 2

        elif self.params.rnn_cell_type == 'peephole_lstm':
            no = 4 * nu
            ng = 4
            ns = self.ns = 2
            nj = 2

        elif self.params.rnn_cell_type == 'basic':
            no = 1 * nu
            ng = 1
            ns = self.ns = 1

        else:
            pass
        self._build_base_graph()

        self._fork_lstm()

        self.Sess = tf.Session()

    def _fork_lstm(self):
        self.Model['Regress'].fork_model_lstm(0)

        if self.params.use_sample_softmax:
            weight = self.Model['Regress'].fork_model[0].Network['Net'][-1].weight
            bias = self.Model['Regress'].fork_model[0].Network['Net'][-1].b

            self.loss_params = {
                'weight': weight, 'bias': bias,
                'num_sample': self.params.num_sample,
                'vocab_size': self.vocab_size
            }

        self.Tensor['Pred'] = \
            self.Model['Regress'].run_fork(self.Placeholder['Input_Feature'], 0)

        self.Output['Pred'] = \
            {'Regress': self.Tensor['Pred']}

        self.Tensor['Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Tensor['Pred'], self.Placeholder['Input_Label'],
                    **self.loss_params
                )
            )

        self.Tensor['Train'] = self.Output['Optimizer'].minimize(
            tf.reduce_sum(self.Tensor['Loss'])
        )

        self._build_summary()

    def _build_summary(self):
        self.train_op = [self.Tensor['Train']]

        self.Output['Loss'] = self.Tensor['Loss']

        if self.params.dataset == 'seq_mnist':

            self.Output['Round'] = \
                tf.argmax(self.Tensor['Pred'], 1)

            self.Output['Error'] = 1 - tf.reduce_mean(
                tf.cast(tf.equal(
                    self.Output['Round'],
                    tf.argmax(self.Placeholder['Input_Label'], 1)
                ), tf.float32)
            )

        else:
            self.Output['Error'] = tf.exp(self.Output['Loss'])

        self.train_res = {
            'Train_Error': self.Output['Error'],
            'Train_Loss': self.Output['Loss']
        }

        self.val_res = {
            'Val_Error': self.Output['Error'],
            'Val_Loss': self.Output['Loss']
        }

        self.Placeholder['Val_Error'] = tf.placeholder(
            dtype=tf.float32, shape=[]
        )
        self.Placeholder['Val_Loss'] = tf.placeholder(
            dtype=tf.float32, shape=[]
        )

        self.Placeholder['Train_Error'] = tf.placeholder(
            dtype=tf.float32, shape=[]
        )
        self.Placeholder['Train_Loss'] = tf.placeholder(
            dtype=tf.float32, shape=[]
        )

        self.train_placeholder = {
            'Train_Error': self.Placeholder['Train_Error'],
            'Train_Loss': self.Placeholder['Train_Loss']
        }

        self.val_placeholder = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': self.Placeholder['Val_Loss']
        }

        self.train_summary = {
            'Train_Error': self.Placeholder['Train_Error'],
            'Train_Loss': self.Placeholder['Train_Loss']
        }
        self.val_summary = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': self.Placeholder['Val_Loss']
        }

