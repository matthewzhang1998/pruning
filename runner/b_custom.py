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

class CustomRunner(LanguageRunner):
    def _build_snip(self):
        self.params.num_unitwise_rnn = self.params.rnn_r_hidden_seq[-1]

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
        ni = self.ni = info[1]['input_depth']

        nu = self.params.num_unitwise_rnn
        assert (nh == nu)

        nb = self.params.batch_size
        self.sess = tf.Session()

        if self.params.rnn_cell_type == 'gru':
            no = 3 * nu
            ng = 3
            ns = self.ns = 1

        elif self.params.rnn_cell_type in ['lstm', 'peephole_lstm']:
            no = 4 * nu
            ng = 4
            ns = self.ns = 2

        elif self.params.rnn_cell_type == 'basic':
            no = 1 * nu
            ng = 1
            ns = self.ns = 1

        else:
            pass

        h_ix = t_ix = int((1 - self.params.prune_k) * (ni + nh) * nh * ng)

        self.weights = np.zeros(t_ix)
        self.row_ind = np.zeros(t_ix, dtype=np.int32)
        self.col_ind = np.zeros(t_ix, dtype=np.int32)

        ix = 0
        self._build_base_graph()
        self._fork_unit_lstm()
        self.sess.run(tf.global_variables_initializer())

        weights = get_init(self.params.rnn_pre_init_type)(
            (ni + nh, no), self._npr, self.params.rnn_pre_init_scale
        )
        #norm_vec = [2.5, 9.7, 1.4, 3.1]
        #for i in range(4):
        #    weights[:,i*nu:(i+1)*nu] /= norm_vec[i]

        grads = np.zeros_like(weights)

        shape = (ni + nh, 0)
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

        dummy_size = (self.params.dummy_batch, ni)
        state_size = (self.params.dummy_batch, ns*nh)

        if self.params.rnn_cell_type == 'gru':
            feed_dict = {
                self.Placeholder['Unit_Kernel']: weights,
                self.Placeholder['Dummy_Inputs']: \
                    np.ones(dummy_size) + \
                    self._npr.normal(size=dummy_size, loc=0, scale=_EPS),
                self.Placeholder['Dummy_States']: \
                    np.ones(state_size) + \
                    self._npr.normal(size=state_size, loc=0, scale=_EPS),
                self.Placeholder['Unit_Roll']: [0],
                self.Placeholder['Sample_Hidden'][0]: placeholder1,
                self.Placeholder['Sample_Hidden'][1]: placeholder2
            }

        else:
            feed_dict = {
                    self.Placeholder['Unit_Kernel']: weights,
                    self.Placeholder['Dummy_Inputs']: \
                        np.ones(dummy_size) + \
                            self._npr.normal(size=dummy_size, loc=0, scale=_EPS),
                    self.Placeholder['Dummy_States']: \
                        np.ones(state_size) + \
                            self._npr.normal(size=state_size, loc=0, scale=_EPS),
                    self.Placeholder['Unit_Roll']: [0],
                    self.Placeholder['Sample_Hidden']: placeholder1
            }

        dummy_score, out, stat = self.sess.run([self.Tensor['Dummy_Score'],
            self.Tensor['Dummy_Outputs'], self.Tensor['Dummy_States']], feed_dict)


        plt.imshow(out, cmap=plt.get_cmap('binary'))
        plt.savefig('{}/ds{}.png'.format(self.Dir, 0))

        plt.imshow(stat, cmap=plt.get_cmap('binary'))
        plt.savefig('{}/dst{}.png'.format(self.Dir, 0))

        dummy_score = np.reshape(dummy_score, weights.shape)

        plt.imshow(-dummy_score, cmap=plt.get_cmap('binary'))
        plt.savefig('{}/d{}.png'.format(self.Dir, 0))

        jacobian = np.zeros((nh,nh))

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

            if self.params.plot_jacobian_pre:
                jacobian =self.sess.run([self.Tensor['Jacobian']], feed_dict)[0]

            if self.params.prune_criteria in ['jacobian', 'jacobian_easy']:
                for t in range(self.params.jacobian_horizon):
                    scores = self.sess.run(
                        [self.Tensor['Unit_Hess'][t]], feed_dict
                    )
                    grads += np.reshape(scores, grads.shape)/(self.params.eval_iter)/dummy_score

            else:
                scores = self.sess.run(
                    [self.Tensor['Unit_Hess']], feed_dict
                )

                grads += np.reshape(scores, grads.shape)/(self.params.eval_iter)/dummy_score

            plt.imshow(np.log(np.abs(weights)), cmap=plt.get_cmap('binary'))
            plt.savefig('{}/w{}.png'.format(self.Dir, 0))

            grads = np.reshape(grads, weights.shape)

            sys.stdout.flush()

            if self.params.uniform_by_gate:
                for k in range(ng):
                    grads[:,k*nh:(k+1)*nh] /= np.mean(np.abs(grads[:,k*nh:(k+1)*nh]))

            if self.params.uniform_by_input:
                grads[:ni] /= np.mean(np.abs(grads[:ni]))
                grads[ni:] /= np.mean(np.abs(grads[ni:]))

            plt.imshow(-np.log(np.abs(grads)), cmap=plt.get_cmap('binary'))
            plt.savefig('{}/grad{}.png'.format(self.Dir, 0))

            inds = top_row, top_col = np.unravel_index(
                np.argpartition(grads, t_ix, axis=None)[:t_ix],
                (ni + nh, ng * nu)
            )

            self.weights = weights[inds]
            self.row_ind = top_row.astype(np.int32)
            self.col_ind = top_col.astype(np.int32)

        if self.params.plot_jacobian_pre:
            self.plot_jacobian(jacobian, '_pre')

        self.row_ind = np.clip(self.row_ind, 0, ni+nh-1)
        self.col_ind = np.clip(self.col_ind, 0, ng*nh-1)

        self.plot(self.weights, self.row_ind, self.col_ind, [ni+nh,ng*nh], 0)

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

        if not self.params.use_knet:
            weight = self.Model['Unit'].fork_model.Network['Net'][-1].weight
            bias = self.Model['Unit'].fork_model.Network['Net'][-1].b

            self.loss_params = {
                'weight': weight, 'bias': bias,
                'num_sample': self.params.num_sample,
                'vocab_size': self.vocab_size
            }

        self.Tensor['Unit_Out'], self.Tensor['Unit_Hidden'] = \
            self.Model['Unit'].run_fork(
                self.Placeholder['Input_Feature'], return_rnn = True
            )
        self.Tensor['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Tensor['Unit_Out'], self.Placeholder['Input_Label'],
                **self.loss_params
            )
        )

        if self.params.plot_jacobian_pre:
            self.Tensor['Jacobian'] = \
                jacobian_tensorflow(
                    self.Tensor['Unit_Hidden'][-1],
                    self.Tensor['Unit_Hidden'][-2]
                )

        if self.params.prune_criteria == 'snip':
            grad_var = self.Model['Unit'].hessian_variables()

            self.Placeholder['Dummy_Inputs'] = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, self.ni])

            self.Placeholder['Dummy_States'] = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, self.ns * self.nh])

            self.Tensor['Dummy_Outputs'], self.Tensor['Dummy_States'], _ = \
                self.Model['Unit'].run_dummy(self.Placeholder['Dummy_Inputs'],
                                             self.Placeholder['Dummy_States'])

            if self.params.dummy_objective == 'jacobian':
                self.Tensor['Dummy_Score'] = tf.abs(
                    self.simple_jacobian_objective(
                        self.Tensor['Dummy_Outputs'], self.Tensor['Dummy_States'],
                        grad_var
                    )
                )
            elif self.params.dummy_objective == 'theta':
                self.Tensor['Dummy_Score'] = tf.abs(
                    grad_var[0] * tf.gradients(self.Tensor['Dummy_Outputs'], grad_var[0])
                )

            elif self.params.dummy_objective == 'none':
                self.Tensor['Dummy_Score'] = tf.ones_like(grad_var[0])

            elif self.params.dummy_objective == 'grad':
                self.Tensor['Dummy_Score'] = tf.abs(tf.gradients(self.Tensor['Dummy_Outputs'], grad_var))

            flat_grad_var = tf.concat([tf.reshape(var, [-1]) for var in grad_var], axis=0)

            grad_i = tf.gradients(self.Tensor['Loss'], grad_var)
            flat_grad_i = tf.concat([tf.reshape(grad, [-1]) for grad in grad_i], axis=0)
            self.Tensor['Unit_Hess'] = -tf.abs(flat_grad_var * flat_grad_i)

        if self.params.prune_criteria == 'hess':
            grad_var = self.Model['Unit'].hessian_variables()

            self.Placeholder['Dummy_Inputs'] = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, self.ni])

            self.Placeholder['Dummy_States'] = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, self.ns * self.nh])

            self.Tensor['Dummy_Outputs'], self.Tensor['Dummy_States'], _ = \
                self.Model['Unit'].run_dummy(self.Placeholder['Dummy_Inputs'],
                                             self.Placeholder['Dummy_States'])

            if self.params.dummy_objective == 'jacobian':
                self.Tensor['Dummy_Score'] = tf.abs(
                    self.simple_jacobian_objective(
                        self.Tensor['Dummy_Outputs'], self.Tensor['Dummy_States'],
                        grad_var
                    )
                )
            elif self.params.dummy_objective == 'theta':
                self.Tensor['Dummy_Score'] = tf.abs(
                    grad_var[0] * tf.gradients(self.Tensor['Dummy_Outputs'], grad_var[0])
                )

            elif self.params.dummy_objective == 'none':
                self.Tensor['Dummy_Score'] = tf.ones_like(grad_var[0])

            elif self.params.dummy_objective == 'grad':
                self.Tensor['Dummy_Score'] = tf.abs(tf.gradients(self.Tensor['Dummy_Outputs'], grad_var))

            flat_grad_var = tf.concat([tf.reshape(var, [-1]) for var in grad_var], axis=0)

            grad_i = tf.gradients(self.Tensor['Loss'], grad_var)
            flat_grad_i = tf.concat([tf.reshape(grad, [-1]) for grad in grad_i], axis=0)

            grad_stop = tf.stop_gradient(flat_grad_i)
            grad_dot = tf.reduce_sum(flat_grad_i * grad_stop)
            hess_grad = tf.gradients(grad_dot, grad_var)
            flat_hess_grad = tf.concat([tf.reshape(hess, [-1]) for hess in hess_grad], axis=0)

            self.Tensor['Unit_Hess'] = -flat_grad_var * flat_hess_grad

        elif self.params.prune_criteria == 'jacobian':
            grad_var = self.Model['Unit'].hessian_variables()

            self.Placeholder['Dummy_Inputs'] = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, self.ni])

            self.Placeholder['Dummy_States'] = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, self.ns * self.nh])

            self.Tensor['Dummy_Outputs'], self.Tensor['Dummy_States'] = \
                self.Model['Unit'].run_dummy(self.Placeholder['Dummy_Inputs'],
                    self.Placeholder['Dummy_States'])

            self.Tensor['Dummy_Score'] = tf.abs(
                self.jacobian_objective(
                    tf.reduce_mean(self.Tensor['Dummy_Outputs']), self.Tensor['Dummy_States'],
                    grad_var
                )
            )
            assert self.params.jacobian_horizon < self.params.max_length

            self.Tensor['Unit_Hess'] = [None for _ in range(self.params.jacobian_horizon)]

            for t in range(self.params.jacobian_horizon):
                score = self.jacobian_objective(
                    tf.reduce_mean(self.Tensor['Unit_Hidden'][-(t + 1)]),
                    self.Tensor['Unit_Hidden'][-(t + 2)],
                    grad_var
                )
                self.Tensor['Unit_Hess'][t] = -tf.abs(score) * \
                    self.params.horizon_trace ** t

        elif self.params.prune_criteria == 'jacobian_easy':
            grad_var = self.Model['Unit'].hessian_variables()

            self.Placeholder['Dummy_Inputs'] = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, self.ni])

            self.Placeholder['Dummy_States'] = tf.placeholder(dtype=tf.float32,
                                                              shape=[None, self.ns * self.nh])

            self.Tensor['Dummy_Outputs'], self.Tensor['Dummy_States'], _ = \
                self.Model['Unit'].run_dummy(self.Placeholder['Dummy_Inputs'],
                    self.Placeholder['Dummy_States'])

            self.Tensor['Dummy_Score'] = tf.abs(
                self.simple_jacobian_objective(
                    self.Tensor['Dummy_Outputs'], self.Tensor['Dummy_States'],
                    grad_var
                )
            )
            assert self.params.jacobian_horizon < self.params.max_length

            self.Tensor['Unit_Hess'] = [None for _ in range(self.params.jacobian_horizon)]

            for t in range(self.params.jacobian_horizon):
                score = self.simple_jacobian_objective(
                    self.Tensor['Unit_Hidden'][-(t + 1)],
                    self.Tensor['Unit_Hidden'][-(t + 2)],
                    grad_var
                )
                self.Tensor['Unit_Hess'][t] = -tf.abs(score) * \
                    self.params.horizon_trace ** t

    def simple_jacobian_objective(self, jacob_y, jacob_x, grad_z):
        def vjp(y, x, dx):
            return tf.gradients(y, x, tf.reshape(dx, tf.shape(y)))

        if self.params.prune_criteria == 'jacobian_easy':
            grad_vec = tf.reduce_mean(tf.square(vjp(jacob_y, jacob_x, tf.ones_like(jacob_y))))
        elif self.params.prune_criteria == 'jacobian_l1':
            grad_vec = tf.reduce_mean((vjp(jacob_y, jacob_x, tf.ones_like(jacob_y))))

        score = tf.concat(
            [tf.reshape(tf.abs(tf.gradients(grad_vec, var)), [-1]) for var in grad_z], axis=0
        )
        return score

    def jacobian_objective(self, hess_y, hess_x, jacob_z):
        def vjp(y, x, dx):
            return tf.gradients(y, x, tf.reshape(dx, tf.shape(y)))

        grad_out_in = tf.gradients(hess_y, hess_x)
        if isinstance(grad_out_in, list):
            flat_grad_out_in = tf.concat([tf.reshape(grad, [-1])
                                          for grad in grad_out_in], axis=0
                                         )
        else:
            flat_grad_out_in = grad_out_in

        stop_out_in = tf.stop_gradient(flat_grad_out_in)
        grad_dot = tf.reduce_sum(flat_grad_out_in * stop_out_in)
        hess_grad = tf.gradients(grad_dot, hess_x)

        if isinstance(hess_grad, list):
            flat_hess_grad = tf.concat(
                [tf.reshape(hess, [-1]) for hess in hess_grad], axis=0
            )
        else:
            flat_hess_grad = hess_grad

        theta_jacobian = tf.concat(
            [tf.reshape(vjp(hess_x, var, flat_hess_grad), [-1])
             for var in jacob_z], axis=0
        )

        return theta_jacobian

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