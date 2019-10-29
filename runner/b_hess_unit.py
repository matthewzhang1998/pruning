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

def discrete_gaussian(scale, npr, size):
    vector = npr.normal(scale=scale, size=size).astype(np.int32)
    return vector

ZERO_32 = tf.constant(0.0, dtype=tf.float32)
EPS = tf.constant(1e-8, dtype=tf.float32)

class UnitRunner(LanguageRunner):
    def _build_snip(self):
        self.Model['Unit'] = Unit('unit', self.params,
            self.vocab_size, self.vocab_size, self.params.seed)
        self.Model['Regress'] = Regress('regress', self.params,
            self.vocab_size, self.vocab_size, self.params.seed)

        self.start_ix = 0

        self.Writer = {
            'Regress': FileWriter(self.Dir, None)
        }
        self.BatchWriter = {
            'Regress': FileWriter(self.Dir, None)
        }
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

            self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                [None, self.params.max_length, self.params.rnn_r_hidden_seq[-1]])

            if self.params.use_knet:
                self.Tensor['Loss_Function'] = \
                    Seq2SeqLoss

            else:
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
        nh = info[1]['hidden_size']
        ni = info[1]['input_depth']

        nu = self.params.num_unitwise_rnn
        h_ix = int((1 - self.params.prune_k) * (ni + nh) * nh / (nh // nu + 1)) * 4
        t_ix = h_ix * (nh // nu + 1)

        self.weights = np.zeros(t_ix)
        self.row_ind = np.zeros(t_ix, dtype=np.int32)
        self.col_ind = np.zeros(t_ix, dtype=np.int32)

        nb = self.params.batch_size


        self.sample_weight = get_init(self.params.rnn_init_type)(
            (t_ix-h_ix), self._npr, self.params.rnn_act_init_scale
        )

        self.sample_ind = np.array(
            np.unravel_index(
                np.random.choice(np.arange((ni + nh) * 4 * (nh-nu)), size=(t_ix-h_ix), replace=False),
            (ni + nh, 4 * (nh-nu)))
        )

        self.sess = tf.Session()
        self._build_base_graph()
        self._fork_unit_lstm()
        self.sess.run(tf.global_variables_initializer())

        for j in range(nh // nu + 1):

            weights = get_init(self.params.rnn_init_type)(
                (ni + nh, 4 * nu), self._npr, self.params.rnn_init_scale
            )
            grads = np.zeros_like(weights)

            for i in range(self.params.eval_iter):

                if j == nh // nu:
                    x = (nh - j*nu)
                    y = j*nu
                    rem = np.concatenate((np.arange(nu-x,nu), np.arange(2*nu-x,2*nu),
                                    np.arange(3*nu-x,3*nu), np.arange(4*nu-x,4*nu))).astype(np.int32)
                    weights[:,rem] = 0

                    ix = np.where(np.divmod(self.col_ind[:j*h_ix], nh)[1] < nh-nu)
                    values = self.weights[ix]

                    q, r = np.divmod(self.col_ind[ix], nh)
                    col = q*(nh-nu)+r
                    indices = np.array([self.row_ind[ix], col], dtype=np.int32).T
                    # must clip to prevent errors
                    shape = (ni + nh, 4 * (nh-nu))

                else:
                    ix = np.where(np.divmod(np.squeeze(self.sample_ind[1,:]), nh-nu)[1] >= j*nu)
                    values = np.concatenate([self.weights[:j*h_ix], self.sample_weight[ix]], axis=0)

                    row_ind = np.concatenate([self.row_ind[:j*h_ix],
                        np.squeeze(self.sample_ind[0,ix])], axis=0)

                    # fancy math for rolling true values (placed at the end of each gate)
                    q, r = np.divmod(self.col_ind[:j*h_ix], nh)

                    local_ind = r + (nh-(j+1)*nu) + q*(nh-nu)

                    # fancy math for rolling sampled values (placed at the start of each gate)
                    q, r = np.divmod(np.squeeze(self.sample_ind[1,:]), nh-nu)

                    sample_ind = r[ix] - j*nu + q[ix]*(nh-nu)

                    col_ind = np.concatenate([local_ind, sample_ind])

                    indices = np.array([row_ind, col_ind], dtype=np.int32).T

                    print(indices.shape, values.shape)
                    # must clip to prevent errors
                    shape = (ni + nh, 4 * (nh-nu))

                features, labels = self.Tensor['Data'][i]

                features = np.clip(features, None, self.vocab_size - 1)
                labels = np.clip(labels, None, self.vocab_size - 1)

                feed_dict = {
                    self.Placeholder['Unit_Kernel']: weights,
                    self.Placeholder['Input_Feature']: features,
                    self.Placeholder['Input_Label']: labels,
                    self.Placeholder['Unit_Roll']: [j*nu],
                    self.Placeholder['Sample_Index']: tf.SparseTensorValue(indices, values, shape)
                }

                scores = self.sess.run(
                    [self.Tensor['Unit_Hess']], feed_dict
                )

                grads += np.reshape(scores, grads.shape)/(self.params.eval_iter)

            plt.imshow(np.log(np.abs(weights)), cmap=plt.get_cmap('binary'))
            plt.savefig('{}/w{}.png'.format(self.Dir, j))

            print(weights)
            grads = np.reshape(grads, weights.shape)

            for k in range(4):
                #pass
                if self.params.uniform_by_gate:
                    grads[:,k*nu:(k+1)*nu] /= np.mean(np.abs(grads[:,k*nu:(k+1)*nu]))

                else:
                    pass

                #top_row, top_col = np.unravel_index(
                #    np.argpartition(grads[:,k*nu:(k+1)*nu], h_ix//4, axis=None)[:h_ix//4],
                #    (ni + nh, nu)
                #)
                # ind = (top_row, top_col + k * nu)
                # top_col = top_col + k * nh + j * nu
                # self.weights[j*h_ix+k*h_ix//4:j*h_ix+(k+1)*h_ix//4] = weights[ind]
                # self.row_ind[j*h_ix+k*h_ix//4:j*h_ix+(k+1)*h_ix//4] = top_row
                # self.col_ind[j*h_ix+k*h_ix//4:j*h_ix+(k+1)*h_ix//4] = top_col

            if self.params.uniform_by_input:
                grads[:ni] /= np.mean(np.abs(grads[:ni]))
                grads[ni:] /= np.mean(np.abs(grads[ni:]))

            plt.imshow(-np.abs(grads), cmap=plt.get_cmap('binary'))
            plt.savefig('{}/grad{}.png'.format(self.Dir, j))

            inds = top_row, top_col = np.unravel_index(
                np.argpartition(-grads, h_ix, axis=None)[:h_ix],
                (ni + nh, 4 * nu)
            )

            arg, rem = np.divmod(top_col, nu)
            top_col = rem + arg * nh + j * nu

            print(np.amax(rem))

            self.weights[j * h_ix:(j + 1) * h_ix] = weights[inds]
            self.row_ind[j * h_ix:(j + 1) * h_ix] = top_row.astype(np.int32)
            self.col_ind[j * h_ix:(j + 1) * h_ix] = top_col.astype(np.int32)

        self.row_ind = np.clip(self.row_ind, 0, ni+nh-1)
        self.col_ind = np.clip(self.col_ind, 0, 4*nh-1)

        self.plot(self.weights, self.row_ind, self.col_ind, [ni+nh,4*nh], 0)

        self.sess.close()
        tf.reset_default_graph()

        self._build_base_graph()
        self._fork_lstm(self.weights, self.row_ind, self.col_ind)

        self.Sess = tf.Session()

    def _fork_unit_lstm(self):
        self.Model['Unit'].set_variables()

        self.Placeholder['Unit_Kernel'], self.Placeholder['Unit_Roll'], \
            self.Placeholder['Sample_Index'] = \
            self.Model['Unit'].get_placeholders()

        if not self.params.use_knet:
            weight = self.Model['Unit'].fork_model.Network['Net'][-1].weight
            bias = self.Model['Unit'].fork_model.Network['Net'][-1].b

            self.loss_params = {
                'weight': weight, 'bias': bias,
                'num_sample': self.params.num_sample,
                'vocab_size': self.vocab_size
            }

        self.Tensor['Unit_Out'] = \
            self.Model['Unit'].run_fork(self.Placeholder['Input_Feature'])

        self.Tensor['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Tensor['Unit_Out'], self.Placeholder['Input_Label'],
                **self.loss_params
            )
        )

        if self.params.use_hessian_criteria:
            grad_var = self.Model['Unit'].hessian_variables()
            flat_grad_var = tf.concat([tf.reshape(var, [-1]) for var in grad_var], axis=0)

            grad_i = tf.gradients(self.Tensor['Loss'], grad_var)
            flat_grad_i = tf.concat([tf.reshape(grad, [-1]) for grad in grad_i], axis=0)

            grad_stop = tf.stop_gradient(flat_grad_i)
            grad_dot = tf.reduce_sum(flat_grad_i * grad_stop)
            hess_grad = tf.gradients(grad_dot, grad_var)
            flat_hess_grad = tf.concat([tf.reshape(hess, [-1]) for hess in hess_grad], axis=0)

            self.Tensor['Unit_Hess'] = -flat_grad_var * flat_hess_grad

    def _fork_lstm(self, weights, row_ind, col_ind):
        self.Model['Regress'].fork_model_lstm(weights, row_ind, col_ind, 0)

        if not self.params.use_knet:
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

        self.Output['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Placeholder['Input_Logits'],
                self.Placeholder['Input_Label'],
                **self.loss_params
            )
        )

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

