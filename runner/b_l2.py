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

class L2Runner(LanguageRunner):
    def _build_snip(self):
        self.params.num_unitwise_rnn = self.params.rnn_r_hidden_seq[-1]

        if self.params.dataset in ['timit', 'seq_mnist']:
            self.Model['Regress'] = Regress('regress', self.params,
                self.vocab_size, self.output_size, self.params.seed)

        else:
            self.Model['Regress'] = Regress('regress', self.params,
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

        self.weights = get_init(self.params.rnn_pre_init_type)(
            (ni + nj * nh, no), self._npr, self.params.rnn_pre_init_scale
        )
        self.mask = np.ones_like(self.weights)
        self.col_ind = self.row_ind = None

        self.plot(self.weights, self.row_ind, self.col_ind, [ni+nj * nh,ng*nh], 0)

        self._build_base_graph()

        self._fork_lstm(self.weights, self.row_ind, self.col_ind)

        self.Sess = tf.Session()

    def _fork_lstm(self, weights, row_ind, col_ind):
        self.Model['Regress'].fork_model_lstm(weights, row_ind, col_ind, 0, use_dense=True)

        if self.params.use_sample_softmax:
            weight = self.Model['Regress'].fork_model[0].Network['Net'][-1].weight
            bias = self.Model['Regress'].fork_model[0].Network['Net'][-1].b

            self.loss_params = {
                'weight': weight, 'bias': bias,
                'num_sample': self.params.num_sample,
                'vocab_size': self.vocab_size
            }

        self.Tensor['RNN_Weight'] = self.Model['Regress'].hessian_variables(0)
        self.Placeholder['RNN_Mask'] = self.Model['Regress'].placeholders(0)

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

    def plot(self, weights, row, col, shape, ix):
        full_lstm = np.zeros(shape)
        full_lstm[(row,col)] = weights

        plt.imshow(full_lstm, cmap=plt.get_cmap('binary'))
        plt.savefig('{}/{}.png'.format(self.Dir, ix))

    def train(self, i):
        summary = {key: defaultdict(list) for key in self.Writer}
        print(self.Dir)
        sys.stdout.flush()

        for (b_feat, b_lab) in self._get_batch('train'):

            b_feat = np.array(b_feat)
            b_lab = np.array(b_lab)
            if np.any(np.isnan(b_feat)) or np.any(np.isnan(b_lab)):
                assert False
            self.batch_ix['train'] += 1

            if self.batch_ix['train'] % self.params.val_steps == 0:
                self.val(self.batch_ix['train'])

            if (self.batch_ix['train'] + 1) % self.params.prune_iter == 0:
                self.recompute_mask((self.batch_ix['train']+1)// self.params.prune_iter)

            feed_dict = {
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab,
                self.Placeholder['Learning_Rate']: self.learning_rate,
                self.Placeholder['RNN_Mask']: self.mask.T
            }

            if self.params.get_jacobian and ((self.batch_ix['train'] - 1)
                % self.params.plot_jacobian_iter) == 0:
                jacob = self.Sess.run(self.Tensor['Jacobian'], feed_dict)
                self.plot_jacobian(jacob, self.batch_ix['train'] - 1)

            pred = self.Sess.run(
                [self.Output['Pred']]+self.train_op, feed_dict)

            pred = pred[0]
            for key in pred:
                b_summary = self.Sess.run(
                    self.train_res,
                    {**feed_dict}
                )

                if self.params.log_memory:
                    self.GraphWriter[key].add_run_metadata(self.Sess.rmd,
                        'step{}'.format(self.batch_ix['train']))

                    self.GraphWriter[key].flush()

                self.BatchWriter[key].add_summary(b_summary, self.batch_ix['train'])

                for summ in b_summary:
                    summary[key][summ].append(b_summary[summ])

            if self.batch_ix['train'] % self.params.log_steps == 0:
                for key in summary:
                    for summ in summary[key]:
                        summary[key][summ] = np.mean(summary[key][summ])

                    write_summary = self.Sess.run(
                        self.train_summary,
                        {self.train_placeholder[summ]: summary[key][summ]
                         for summ in summary[key]}
                    )
                    self.Writer[key].add_summary(write_summary, self.batch_ix['train'])

                summary = {key: defaultdict(list) for key in self.Writer}
        self.learning_rate = self.decay_lr(self.batch_ix['train'], self.learning_rate)

    def val(self, i):
        print(self.Dir)
        sys.stdout.flush()
        start = 0
        summary = {key: defaultdict(list) for key in self.Writer}

        for k,(b_feat, b_lab) in enumerate(self._get_batch('val')):
            if k > self.params.val_iter:
                break

            self.batch_ix['val'] += 1
            feed_dict = {
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab,
                self.Placeholder['RNN_Mask']: self.mask.T,
            }
            pred = self.Sess.run(
                [self.Output['Pred']], feed_dict)

            pred = pred[0]
            for key in pred:
                b_summary = self.Sess.run(
                    self.val_res,
                    {**feed_dict}
                )
                self.BatchWriter[key].add_summary(b_summary, self.batch_ix['val'])

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

    def recompute_mask(self, k):
        if k >= len(self.params.prune_iter_k_seq):
            return

        else:
            ratio_c = self.params.prune_iter_k_seq[k]
            weights = self.Sess.run([self.Tensor['RNN_Weight']])[0][0].T * self.mask

            new_mask = np.zeros_like(weights)

            num_parameters = int(np.prod(weights.shape) * ratio_c)
            sorted = np.unravel_index(
                np.argpartition(-np.abs(weights), num_parameters, axis=None)[:num_parameters],
                weights.shape
            )
            new_mask[sorted] = 1
            self.mask = new_mask
