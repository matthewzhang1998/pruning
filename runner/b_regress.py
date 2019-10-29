import os.path as osp
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from runner.base_runner import *
from util.optimizer_util import *
from model.regress import *
import scipy.misc
from util.logger_util import *
from util.initializer_util import *
from collections import defaultdict

from data.load_pen import *

def discrete_gaussian(scale, npr, size):
    vector = npr.normal(scale=scale, size=size).astype(np.int32)
    return vector

class Adam(object):
    def __init__(self, shape, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08, dtype=np.float32):
        self.stepsize, self.beta1, self.beta2, self.epsilon = stepsize, beta1, beta2, epsilon
        self.t = 0
        self.m = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)

    def step(self, g):
        self.t += 1
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        return - a * self.m / (np.sqrt(self.v) + self.epsilon)

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
            self.vocab_size, self.vocab_size, self.params.seed)

        self.start_ix = 0

        self.Writer = {
            key: FileWriter(self.Dir, None) for key in self.Model
        }
        self._preprocess()
        sys.stdout.flush()

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
                [None, self.params.max_length, self.vocab_size])

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

        self.Model['Regress'].set_embed_and_softmax(
            self.embed_matrix, self.softmax_matrix, use_dense=True
        )

        #### LSTM ####
        nh = info[1]['hidden_size']
        ni = info[1]['input_depth']

        print(ni,nh)
        nu = self.params.num_unitwise_rnn
        h_ix = int((1 - self.params.prune_k) * (ni + nh) * nh / (nh // nu + 1)) * 4
        t_ix = h_ix * (nh // nu + 1)

        n = np.random.choice(np.arange((ni + nh) * 4 * nh), size=t_ix, replace=False)
        self.row_ind, self.col_ind = np.unravel_index(n, (ni + nh, 4 * nh))
        self.weights = get_init(self.params.rnn_init_type)(
            (t_ix,), self._npr, self.params.rnn_init_scale
        )

        self.Tensor['Outputs'] = [None for _ in range(self.params.num_gpu)]
        self.Tensor['Initializer'] = [None for _ in range(self.params.num_gpu)]
        self.Tensor['Loss'] = [None for _ in range(self.params.num_gpu)]
        self.Tensor['Train'] = [None for _ in range(self.params.num_gpu)]

        num_per_iter = (self.params.num_generate // self.params.num_gpu) * self.params.num_gpu

        self.Tensor['Row_Eps'] = [None for _ in range(num_per_iter)]
        self.Tensor['Col_Eps'] = [None for _ in range(num_per_iter)]
        self.Tensor['Val_Eps'] = [None for _ in range(num_per_iter)]
        self.Tensor['Score'] = [None for _ in range(num_per_iter)]

        self.Tensor['Rank'] = [None for _ in range(num_per_iter)]

        self.previous_score = np.ones((t_ix,)) * np.inf
        self.col_adam = Adam(t_ix, self.params.evolution_lr)
        self.row_adam = Adam(t_ix, self.params.evolution_lr)

        for k in range(self.params.num_iterate):
            if k % self.params.val_steps == 0:

                self._build_base_graph()
                self.sess = tf.Session()
                tf.set_random_seed(self.params.seed)
                self._fork_random_lstm(self.weights, self.row_ind, self.col_ind, 0)
                if self.params.meta_opt_method == 'convex':
                    self.previous_score = self._eval_hess([0])[0]
                else:
                    self.previous_score = np.sum(self._eval_hess([0])[0])

                print(self.previous_score)

                self.train_first(iters=self.params.test_iter-1)

                self.plot(self.weights, self.row_ind, self.col_ind, (ni+nh, 4*nh), k)

                tf.reset_default_graph()
                self.sess.close()

            for i in range(self.params.num_generate // self.params.num_gpu):
                tf.set_random_seed(self.params.seed)
                self._build_base_graph()
                for j in range(self.params.num_gpu):
                    _npr = np.random.RandomState(self.params.seed + 16777 * i + 8522 * k + 4753 * j)

                    if self.params.noise_type == 'normal':
                        noise_vector_row = _npr.normal(scale=self.params.rand_eps, size=t_ix)
                        noise_vector_col = _npr.normal(scale=self.params.rand_eps, size=t_ix)
                        #noise_vector_weights = _npr.normal(scale=self.params.weight_eps, size=t_ix)

                        self.Tensor['Row_Eps'][i*self.params.num_gpu + j] = noise_vector_row
                        self.Tensor['Col_Eps'][i*self.params.num_gpu + j] = noise_vector_col
                        #self.Tensor['Val_Eps'][i*self.params.num_gpu + j] = noise_vector_weights

                        weights = self.weights

                        flt_row, int_row = np.modf(noise_vector_row + self.row_ind)
                        flt_col, int_col = np.modf(noise_vector_col + self.col_ind)
                        row_ind = int_row + _npr.binomial(1, flt_row)
                        col_ind = int_col + _npr.binomial(1, flt_col)

                        row_ind = np.clip(row_ind, 0, ni+nh-1)
                        col_ind = np.clip(col_ind, 0, 4*nh-1)

                    elif self.params.noise_type == 'discrete':
                        noise_vector_col = discrete_gaussian(self.params.rand_eps, _npr, t_ix)
                        noise_vector_row = discrete_gaussian(self.params.rand_eps, _npr, t_ix)

                        self.Tensor['Row_Eps'][i * self.params.num_gpu + j] = noise_vector_row
                        self.Tensor['Col_Eps'][i * self.params.num_gpu + j] = noise_vector_col
                        row_ind = np.clip(self.row_ind + noise_vector_row, 0, ni+nh-1)
                        col_ind = np.clip(self.col_ind + noise_vector_col, 0, 4*nh-1)
                        weights = self.Tensor['Val_Eps'][i * self.params.num_gpu + j] = \
                            get_init(self.params.rnn_init_type)(
                                (t_ix,), self._npr, self.params.rnn_init_scale
                            )

                    elif self.params.noise_type == 'replace':
                        n = _npr.choice(np.arange((ni + nh) * 4 * nh), size=t_ix, replace=False)
                        row_ind, col_ind = np.unravel_index(n, (ni + nh, 4 * nh))
                        self.Tensor['Row_Eps'][i * self.params.num_gpu + j] = row_ind
                        self.Tensor['Col_Eps'][i * self.params.num_gpu + j] = col_ind
                        #weights = self.weights
                        weights = self.Tensor['Val_Eps'][i * self.params.num_gpu + j] = \
                            get_init(self.params.rnn_init_type)(
                                (t_ix,), self._npr, self.params.rnn_init_scale
                            )

                    self._fork_random_lstm(weights, row_ind, col_ind, j)

                #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
                #config = tf.ConfigProto(allow_soft_placement=True)
                #self.opt = tf.RunOptions(report_tensor_allocations_upon_oom=True)
                self.sess = tf.Session()#config=config)

                if self.params.use_hessian_criteria:
                    train_loss = self._eval_hess()
                else:
                    train_loss = self._train_parallel()

                for j in range(self.params.num_gpu):
                    self.Tensor['Rank'][i*self.params.num_gpu + j] = train_loss[j]

                tf.reset_default_graph()
                self.sess.close()
                self.eval_ix += self.params.num_gpu

            #mean = np.mean(self.Tensor['Score'])
            #std = np.std(self.Tensor['Score'])

            if self.params.meta_opt_method == 'ranked':
                score = (1/np.arange(num_per_iter))[np.argsort(self.Tensor['Rank'])][:,np.newaxis] # don't take negative weights
                row_add = np.sum(score * np.array(self.Tensor['Row_Eps']), axis=0)
                col_add = np.sum(score * np.array(self.Tensor['Col_Eps']), axis=0)

                row_add = self.row_adam.step(row_add)
                col_add = self.col_adam.step(col_add)
                #val_add = np.sum(score * np.array(self.Tensor['Val_Eps']), axis=0)
                self.row_ind = np.clip(self.row_ind + row_add, 0, ni+nh-1)
                self.col_ind = np.clip(self.col_ind + col_add, 0, 4*nh-1)

                #self.weights += val_add

            elif self.params.meta_opt_method == 'max':
                rank = np.sum(self.Tensor['Rank'], axis=-1)
                ix = np.argmax(-rank)
                score = rank[ix]
                if score < self.previous_score:
                    self.row_ind = np.clip(self.row_ind + self.Tensor['Row_Eps'][ix], 0, ni+nh-1)
                    self.col_ind = np.clip(self.col_ind + self.Tensor['Col_Eps'][ix], 0, 4*nh-1)
                    self.weights = self.Tensor['Val_Eps'][ix]

            elif self.params.meta_opt_method == 'convex':
                scores = np.reshape(np.array(self.Tensor['Rank'] + [self.previous_score]), [-1])
                row_inds = np.reshape(np.array(self.Tensor['Row_Eps'] + [self.row_ind]), [-1])
                col_inds = np.reshape(np.array(self.Tensor['Col_Eps'] + [self.col_ind]), [-1])
                weights = np.reshape(np.array(self.Tensor['Val_Eps'] + [self.weights]), [-1])

                if self.params.uniform_by_gate:
                    for i in range(4):
                        gate_inds = np.where((col_inds < (i+1)*nh)&(col_inds >= i*nh))
                        ix = np.argpartition(scores[gate_inds], t_ix//4)[:t_ix//4]
                        self.row_ind[i*t_ix//4:(i+1)*t_ix//4] = row_inds[ix]
                        self.col_ind[i*t_ix//4:(i+1)*t_ix//4] = col_inds[ix]
                        self.weights[i*t_ix//4:(i+1)*t_ix//4] = weights[ix]

                else:
                    ix = np.argpartition(scores[gate_inds], t_ix)[:t_ix]
                    self.row_ind = row_inds[ix]
                    self.col_ind = col_inds[ix]
                    self.weights = weights[ix]

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

        if self.params.use_hessian_criteria:
            grad_var = self.Model['Regress'].hessian_variables(i)
            flat_grad_var = tf.concat([tf.reshape(var, [-1]) for var in grad_var], axis=0)

            grad_i = tf.gradients(self.Tensor['Loss'][i], grad_var)
            flat_grad_i = tf.concat([tf.reshape(grad, [-1]) for grad in grad_i], axis=0)

            grad_stop = tf.stop_gradient(flat_grad_i)
            grad_dot = tf.reduce_sum(flat_grad_i * grad_stop)
            hess_grad = tf.gradients(grad_dot, grad_var)
            flat_hess_grad = tf.concat([tf.reshape(hess, [-1]) for hess in hess_grad], axis=0)

            self.Tensor['Score'][i] = -flat_grad_var * flat_hess_grad

    def _eval_hess(self, ix=None):
        self.sess.run(tf.global_variables_initializer())

        final_score = []
        for i in range(self.params.eval_iter - 1):
            b_feat, b_lab = self.Tensor['Data'][i]

            b_feat = np.clip(b_feat, None, self.params.vocab_size - 1)
            b_lab = np.clip(b_lab, None, self.params.vocab_size - 1)
            feed_dict = {
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab
            }
            sys.stdout.flush()

            objective = [self.Tensor['Score'][i] for i in ix] if ix is not None else self.Tensor['Score']
            final_score.append(self.sess.run(objective, feed_dict))
        final_score = np.mean(np.array(final_score), axis=0)

        if ix is None:
            for i in range(self.params.num_gpu):
                self.Writer['Regress'].add_summary({"HessFinal": np.sum(final_score[i])}, self.eval_ix + i)

        return final_score

    def _train_parallel(self):
        self.Tensor['Train'] = self.Output['Optimizer'].minimize(
            tf.reduce_sum(self.Tensor['Loss'])
        )
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.params.eval_iter-1):
            b_feat, b_lab = self.Tensor['Data'][i]

            b_feat = np.clip(b_feat, None, self.params.vocab_size - 1)
            b_lab = np.clip(b_lab, None, self.params.vocab_size - 1)
            feed_dict = {
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab,
                self.Placeholder['Learning_Rate']: self.params.learning_rate
            }
            sys.stdout.flush()
            print(i)
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

    def train_first(self, iters):
        self.Tensor['Train'] = self.Output['Optimizer'].minimize(
            tf.reduce_sum(self.Tensor['Loss'][0])
        )
        self.sess.run(tf.global_variables_initializer())

        for i in range(iters):
            b_feat, b_lab = self.Tensor['Data'][i]

            b_feat = np.clip(b_feat, None, self.params.vocab_size - 1)
            b_lab = np.clip(b_lab, None, self.params.vocab_size - 1)
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

        final_loss = self.sess.run(self.Tensor['Loss'][0], feed_dict)
        self.Writer['Regress'].add_summary({"RegressFinal": final_loss}, self.eval_ix)

        return final_loss

    def plot(self, weights, row, col, shape, ix):
        full_lstm = np.zeros(shape)
        full_lstm[row,col] = weights

        plt.imshow(full_lstm, cmap=plt.get_cmap('binary'))
        plt.savefig('{}/{}.png'.format(self.Dir, ix))

