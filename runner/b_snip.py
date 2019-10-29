import os.path as osp
import sys

import numpy as np
import tensorflow as tf
from runner.base_runner import *
from util.optimizer_util import *
from model.regress import *
from model.unit_hess import *

import scipy.misc
from util.logger_util import *
from util.initializer_util import *
from collections import defaultdict

import matplotlib.pyplot as plt

from data.load_pen import *

_EPS = 0.1

class SnipRunner(LanguageRunner):
    def _build_snip(self):
        self.params.num_unitwise_rnn = self.params.rnn_r_hidden_seq[-1]

        self.Model['Unit'] = Unit('unit', self.params,
            self.vocab_size, self.vocab_size, self.params.seed)
        self.Model['Regress'] = Regress('regress', self.params,
            self.vocab_size, self.vocab_size, self.params.seed)

        self.start_ix = 0

        self._preprocess()
        sys.stdout.flush()

        self.Model.pop('Unit', None)

    def _build_base_graph(self):
        with tf.variable_scope(self.scope):

            self.Placeholder['Input_Feature'] = tf.placeholder(
                shape=[None, self.params.max_length], dtype=tf.int32,
            )

            self.Placeholder['Learning_Rate'] = tf.placeholder(
                tf.float32, []
            )

            self.Placeholder['Input_Label'] = tf.placeholder(
                tf.int32, [None, self.params.max_length]
            )

            if not self.params.use_sample_softmax:
                self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                    [None, self.params.max_length, self.vocab_size])

                self.Tensor['Loss_Function'] = \
                    Seq2SeqLoss

            else:
                self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                    [None, self.params.max_length, self.params.rnn_r_hidden_seq[-1]])

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

        self.Model['Unit'].set_embed_and_softmax(
            self.embed_matrix, self.softmax_matrix, use_dense=True
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

        h_ix = t_ix = int((1 - self.params.prune_k) * (ni + nj * nh) * nh * ng)

        self.weights = np.zeros(t_ix)
        self.row_ind = np.zeros(t_ix, dtype=np.int32)
        self.col_ind = np.zeros(t_ix, dtype=np.int32)

        ix = 0
        self._build_base_graph()
        self._fork_unit_lstm()
        self.sess.run(tf.global_variables_initializer())

        weights = get_init(self.params.rnn_pre_init_type)(
            (ni + nj * nh, no), self._npr, self.params.rnn_pre_init_scale
        )
        #norm_vec = [2.5, 9.7, 1.4, 3.1]
        #for i in range(4):
        #    weights[:,i*nu:(i+1)*nu] /= norm_vec[i]

        grads = np.zeros_like(weights)

        shape = (ni + nj * nh, 0)
        indices = np.zeros((0, 2))
        values = np.zeros((0,))

        if self.params.rnn_cell_type == 'gru':
            placeholder1 = tf.SparseTensorValue(
                indices=indices, values=values, dense_shape=shape)

            placeholder2 = tf.SparseTensorValue(
                indices=indices, values=values, dense_shape=shape)

        else:
            placeholder1 = tf.SparseTensorValue(
                indices=indices, values=values, dense_shape=shape)

        labels = self._npr.randint(0, self.vocab_size,
            size=(self.params.dummy_batch, self.params.max_length))

        if self.params.rnn_cell_type == 'gru':
            feed_dict = {
                self.Placeholder['Unit_Kernel']: weights,
                self.Placeholder['Input_Feature']: labels,
                self.Placeholder['Input_Label']: labels,
                self.Placeholder['Unit_Roll']: [0],
                self.Placeholder['Sample_Hidden'][0]: placeholder1,
                self.Placeholder['Sample_Hidden'][1]: placeholder2
            }
        else:
            feed_dict = {
                self.Placeholder['Unit_Kernel']: weights,
                self.Placeholder['Input_Feature']: labels,
                self.Placeholder['Input_Label']: labels,
                self.Placeholder['Unit_Roll']: [0],
                self.Placeholder['Sample_Hidden']: placeholder1
            }

        scores = self.sess.run(
            [self.Tensor['Dummy_Hess']], feed_dict
        )

        dummy = np.reshape(scores, grads.shape)/(self.params.eval_iter)

        for i in range(self.params.eval_iter):
            features, labels = self.Tensor['Data'][i]

            features = np.clip(features, None, self.params.vocab_size - 1)
            labels = np.clip(labels, None, self.params.vocab_size - 1)

            if self.params.rnn_cell_type == 'gru':
                feed_dict = {
                    self.Placeholder['Unit_Kernel']: weights,
                    self.Placeholder['Input_Feature']: features,
                    self.Placeholder['Input_Label']: labels,
                    self.Placeholder['Unit_Roll']: [0],
                    self.Placeholder['Sample_Hidden'][0]: placeholder1,
                    self.Placeholder['Sample_Hidden'][1]: placeholder2
                }
            else:
                feed_dict = {
                    self.Placeholder['Unit_Kernel']: weights,
                    self.Placeholder['Input_Feature']: features,
                    self.Placeholder['Input_Label']: labels,
                    self.Placeholder['Unit_Roll']: [0],
                    self.Placeholder['Sample_Hidden']: placeholder1
                }


            scores = self.sess.run(
                [self.Tensor['Unit_Hess']], feed_dict
            )

            grads += np.reshape(scores, grads.shape)/(self.params.eval_iter)# /dummy

        plt.imshow(np.log(np.abs(weights)), cmap=plt.get_cmap('binary'))
        plt.savefig('{}/w{}.png'.format(self.Dir, 0))

        grads = np.reshape(grads, weights.shape)

        sys.stdout.flush()

        plt.imshow(-np.log(np.abs(grads)), cmap=plt.get_cmap('binary'))
        plt.savefig('{}/grad{}.png'.format(self.Dir, 0))

        inds = top_row, top_col = np.unravel_index(
            np.argpartition(grads, t_ix, axis=None)[:t_ix],
            (ni + nj * nh, ng * nu)
        )

        self.weights = weights[inds]
        self.row_ind = top_row.astype(np.int32)
        self.col_ind = top_col.astype(np.int32)

        self.row_ind = np.clip(self.row_ind, 0, ni+nj*nh-1)
        self.col_ind = np.clip(self.col_ind, 0, ng*nh-1)

        self.plot(self.weights, self.row_ind, self.col_ind, [ni+nj * nh,ng*nh], 0)

        self.sess.close()
        tf.reset_default_graph()

        self._build_base_graph()

        #self.weights = get_init(self.params.rnn_init_type)(
        #    (t_ix), self._npr, self.params.rnn_init_scale
        #)

        self._fork_lstm(self.weights, self.row_ind, self.col_ind)

        self.Sess = tf.Session()

    def _fork_unit_lstm(self):
        self.Model['Unit'].set_variables()

        self.Placeholder['Unit_Kernel'], self.Placeholder['Unit_Roll'], \
            self.Placeholder['Sample_Hidden'] = \
            self.Model['Unit'].get_placeholders()

        if self.params.use_sample_softmax:
            weight = self.Model['Unit'].fork_model.Network['Net'][-1].weight
            bias = self.Model['Unit'].fork_model.Network['Net'][-1].b

            self.loss_params = {
                'weight': weight, 'bias': bias,
                'num_sample': self.params.num_sample,
                'vocab_size': self.vocab_size
            }
        self.Tensor['Unit_Out'] = \
            self.Model['Unit'].run_fork(self.Placeholder['Input_Feature'])

        _, _, self.Tensor['Dummy_Out'] = \
            self.Model['Unit'].run_dummy(None, None, use_last=True)

        self.Tensor['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Tensor['Unit_Out'], self.Placeholder['Input_Label'],
                **self.loss_params
            )
        )
        print(self.Tensor['Dummy_Out'], self.Tensor['Unit_Out'])

        self.Tensor['Dummy_Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Tensor['Dummy_Out'], self.Placeholder['Input_Label'],
                **self.loss_params
            )
        )

        grad_dummy = self.Model['Unit'].hessian_variables()

        grad_d = tf.gradients(self.Tensor['Dummy_Loss'], grad_dummy)
        flat_grad_d = tf.concat([tf.reshape(grad * var, [-1]) for (grad, var)
                                 in zip(grad_d, grad_dummy)], axis=0)

        self.Tensor['Dummy_Hess'] = flat_grad_d

        grad_var = self.Model['Unit'].hessian_variables()

        grad_i = tf.gradients(self.Tensor['Loss'], grad_var)
        flat_grad_i = tf.concat([tf.reshape(grad*var, [-1]) for (grad, var)
            in zip(grad_i, grad_var)], axis=0)
        print(flat_grad_i, grad_i, grad_var)

        self.Tensor['Unit_Hess'] = flat_grad_i

    def _fork_lstm(self, weights, row_ind, col_ind):
        self.Model['Regress'].fork_model_lstm(weights, row_ind, col_ind, 0)

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

    def plot(self, weights, row, col, shape, ix):
        full_lstm = np.zeros(shape)
        full_lstm[(row,col)] = weights

        plt.imshow(full_lstm, cmap=plt.get_cmap('binary'))
        plt.savefig('{}/{}.png'.format(self.Dir, ix))