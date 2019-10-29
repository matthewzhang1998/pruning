import os.path as osp
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from runner.base_runner import *
from util.optimizer_util import *
from model.regress import *
from model.unit_hess import *

import scipy.misc
from util.logger_util import *
from util.initializer_util import *
from collections import defaultdict

from data.load_pen import *

def norm_comp(ev):
    return np.square(np.abs(ev))

def jacobian_tensorflow(y, x):
    jacobian_matrix = []
    ny = y.get_shape().as_list()[-1]
    print(x)
    for i in range(ny):
        # We iterate over the M elements of the output vector
        grad_func = tf.gradients(y[0,i], x)
        print(i)
        grad_func = grad_func[0][0,:]
        jacobian_matrix.append(grad_func)
    jacobian_matrix = tf.stack(jacobian_matrix)
    return jacobian_matrix


def discrete_gaussian(scale, npr, size):
    vector = npr.normal(scale=scale, size=size).astype(np.int32)
    return vector

ZERO_32 = tf.constant(0.0, dtype=tf.float32)
EPS = tf.constant(1e-8, dtype=tf.float32)

class SVRunner(LanguageRunner):
    def _build_snip(self):
        if self.params.dataset in ['timit', 'seq_mnist']:
            self.Model['Unit'] = Unit('unit', self.params,
                self.vocab_size, self.output_size, self.params.seed)
            self.Model['Regress'] = Regress('regress', self.params,
                self.vocab_size, self.output_size, self.params.seed)

        else:
            self.Model['Unit'] = Unit('unit', self.params,
                self.vocab_size, self.vocab_size, self.params.seed)
            self.Model['Regress'] = Regress('regress', self.params,
                self.vocab_size, self.vocab_size, self.params.seed)


        self.start_ix = 0

        self.Writer = {
            'Unit': FileWriter(self.Dir, None)
        }
        self.BatchWriter = {
            'Unit': FileWriter(self.Dir, None)
        }
        self._preprocess()
        sys.stdout.flush()

        self.Model.pop('Unit', None)

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
        ni = info[1]['input_depth']

        nu = self.params.num_unitwise_rnn

        nj = 2 if self.params.rnn_cell_type == 'peephole_lstm' else 1
        if self.params.rnn_cell_type == 'gru':
            ng = 3
        elif self.params.rnn_cell_type in ['lstm','peephole_lstm']:
            ng = 4
        elif self.params.rnn_cell_type == 'basic':
            ng = 1

        h_ix = int((1 - self.params.prune_k) * (ni + nj * nh) * ng * nh / (nh // nu + 1))
        t_ix = h_ix * (nh // nu + 1)

        n = np.random.choice(np.arange((ni + nj * nh) * ng * nh), size=t_ix, replace=False)
        self.row_ind, self.col_ind = np.unravel_index(n, (ni + nj * nh, ng * nh))
        self.weights = get_init(self.params.rnn_pre_init_type)(
            (ni+nj * nh, ng * nh), self._npr, self.params.rnn_init_scale, arch=self.params.rnn_cell_type,
        )
        self.weights = self.weights[(self.row_ind,self.col_ind)]

        tf.reset_default_graph()

        self._build_base_graph()
        self._fork_lstm(self.weights, self.row_ind, self.col_ind)

        self.Sess = tf.Session()

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

        if self.params.get_jacobian:
            self.Tensor['Pred'], self.Tensor['State'] = \
                self.Model['Regress'].run_fork(self.Placeholder['Input_Feature'], 0, return_rnn=True)
            self.Tensor['Jacobian'] = \
                jacobian_tensorflow(
                    self.Tensor['State'][-1],
                    self.Tensor['State'][-2]
                )

        else:
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