import os.path as osp
import sys

import numpy as np
import tensorflow as tf
from runner.base_runner import *
from util.optimizer_util import *
from model.regress import *
import scipy.misc
from util.logger_util import *
from util.initializer_util import *
from collections import defaultdict

from data.load_pen import *

def sort_bin(b):
    # useful for removing symmetries (can still end up with
    b = np.sum(b, axis=0)

    arg_sort = np.argsort(b)
    return arg_sort

def sort_sparse_lstm(sparse_lstm_indices, sparse_lstm_values, ni, nh):
    sparse_lstm_binary = np.zeros([ni+nh,4*nh])
    sparse_lstm = np.zeros([ni+nh,4*nh])
    sparse_lstm_binary[sparse_lstm_indices] = 1
    sparse_lstm[sparse_lstm_indices] = sparse_lstm_values
    # ni+nh, 4*nh
    rotation_indices_column = sort_bin(sparse_lstm_binary[ni:,:nh])
    rotation_indices_row = np.concat([np.arange(nh), rotation_indices_column + nh])
    for i in range(4):
        sparse_lstm[rotation_indices_row][:,rotation_indices_column+i*nh]
    indices = np.argwhere(sparse_lstm > 0)
    vals = sparse_lstm[indices]

    return indices, vals

ZERO_32 = tf.constant(0.0, dtype=tf.float32)
EPS = tf.constant(1e-8, dtype=tf.float32)

class RegressionRunner(BillionRunner):
    def _build_snip(self):
        self.Model['Regress'] = Regress('regress', self.params,
            self.vocab_size, self.params.rnn_r_hidden_seq[-1], self.params.seed)

        self.start_ix = 0

        self.Writer = {
            key: FileWriter(self.Dir + '/epoch/' + key, None) for key in self.Model
        }
        self._preprocess()
        sys.stdout.flush()

    def _build_base_graph(self):
        with tf.variable_scope(self.scope):

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
        self.Tensor['Data'] = list(itertools.islice(self._get_batch(), self.params.eval_iter))

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
        nh = info[1]['hidden_size']
        ni = info[1]['input_depth']
        nu = self.params.num_unitwise_rnn
        h_ix = int((1 - self.params.prune_k) * (ni + nh) * 4 * nh / (nh // nu + 1))
        t_ix = h_ix * (nh // nu + 1)

        n = np.random.choice(np.arange((ni + nh) * 4 * nh), size=t_ix, replace=False)
        self.row_ind, self.col_ind = np.unravel_index(n, (ni + nh, 4 * nh))

        self.Tensor['Outputs'] = [None for _ in range(self.params.num_gpu)]
        self.Tensor['Initializer'] = [None for _ in range(self.params.num_gpu)]
        self.Tensor['Loss'] = [None for _ in range(self.params.num_gpu)]
        self.Tensor['Train'] = [None for _ in range(self.params.num_gpu)]

        num_per_iter = (self.params.num_generate // self.params.num_gpu) * self.params.num_gpu

        self.Tensor['Row_Eps'] = [None for _ in range(num_per_iter)]
        self.Tensor['Col_Eps'] = [None for _ in range(num_per_iter)]
        self.Tensor['Score'] = [None for _ in range(num_per_iter)]

        for k in range(self.params.num_iterate):
            for i in range(self.params.num_generate // self.params.num_gpu):
                self._build_base_graph()
                for j in range(self.params.num_gpu):
                    _npr = np.random.RandomState(self.params.seed + 16777 * i + 8522 * k + 4753 * j)
                    noise_vector_row = _npr.normal(scale=self.params.rand_eps, size=t_ix)
                    noise_vector_col = _npr.normal(scale=self.params.rand_eps, size=t_ix)

                    self.Tensor['Row_Eps'][i*self.params.num_gpu + j] = noise_vector_row
                    self.Tensor['Col_Eps'][i*self.params.num_gpu + j] = noise_vector_col

                    weights = get_init(self.params.rnn_init_type)(
                        (t_ix,), self._npr, self.params.rnn_init_scale
                    )

                    row_ind = np.clip(self.row_ind + noise_vector_row, 0, ni+nh-1)
                    col_ind = np.clip(self.col_ind + noise_vector_col, 0, 4*nh-1)

                    flt_row, int_row = np.modf(row_ind)
                    flt_col, int_col = np.modf(col_ind)
                    row_ind = int_row + _npr.binomial(1, flt_row)
                    col_ind = int_col + _npr.binomial(1, flt_col)

                    self._fork_random_lstm(weights, row_ind, col_ind, j)

                tf.set_random_seed(self.params.seed)
                #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
                #config = tf.ConfigProto(allow_soft_placement=True)
                #self.opt = tf.RunOptions(report_tensor_allocations_upon_oom=True)
                self.sess = tf.Session()#config=config)

                train_loss = self._train_parallel()
                for j in range(self.params.num_gpu):
                    self.Tensor['Score'][i*self.params.num_gpu + j] = train_loss[j]

                tf.reset_default_graph()
                self.sess.close()
                self.eval_ix += self.params.num_gpu

            sorted_score = np.argsort(-self.Tensor['Score'])
            inv_range = 1/np.arange(num_per_iter)[sorted_score]
            row_add = np.sum(inv_range * np.array(self.Tensor['Row_Eps']), axis=0)
            col_add = np.sum(inv_range * np.array(self.Tensor['Col_Eps']), axis=0)
            self.row_ind = np.clip(self.row_ind + row_add, 0, ni+nh)
            self.col_add = np.clip(self.col_ind + col_add, 0, 4*nh)

    def _fork_random_lstm(self, weights, row_ind, col_ind, i):
        self.Model['Regress'].fork_model_lstm(weights, row_ind, col_ind, i)

        if not self.params.use_knet:
            weight = self.Model['Regress'].fork_model[i].Network['Net'][-1].weight
            bias = self.Model['Regress'].fork_model[i].Network['Net'][-1].b

            self.loss_params = {
                'weight': weight, 'bias': bias,
                'num_sample': self.params.num_sample,
                'vocab_size': self.vocab_size
            }

        self.Tensor['Outputs'][i] = self.Model['Regress'].run_fork(self.Placeholder['Input_Feature'], i)

        self.Tensor['Loss'][i] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Tensor['Outputs'][i], self.Placeholder['Input_Label'],
                    **self.loss_params
                )
            )

    def _train_parallel(self):
        self.Tensor['Train'] = self.Output['Optimizer'].minimize(
            tf.reduce_sum(self.Tensor['Loss'])
        )
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.params.eval_iter-1):
            b_feat, b_lab = self.Tensor['Data'][i]
            feed_dict = {
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab,
                self.Placeholder['Learning_Rate']: self.params.learning_rate
            }
            self.sess.run([self.Tensor['Train']], feed_dict)#, options=self.opt)

        b_feat, b_lab = self.Tensor['Data'][-1]
        feed_dict = {
            self.Placeholder['Input_Feature']: b_feat,
            self.Placeholder['Input_Label']: b_lab,
        }

        final_loss = self.sess.run(self.Tensor['Loss'], feed_dict)
        for i in range(self.params.num_gpu):
            self.Writer['Regress'].add_summary({"RegressFinal": final_loss[i]}, self.eval_ix + i)

        return final_loss