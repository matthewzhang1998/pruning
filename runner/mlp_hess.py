import os.path as osp
import sys

import numpy as np
import tensorflow as tf
from runner.base_runner import *
from util.optimizer_util import *
from model.mlp_regress import *
from model.mlp_unit_hess import *

from util.logger_util import *
from util.initializer_util import *

import matplotlib.pyplot as plt

_EPS = 0.1

class HessRunner(MLPRunner):
    def _build_snip(self):

        self.Model['Unit'] = Unit('unit', self.params,
            self.num_features, self.num_classes, self.params.seed)
        self.Model['Regress'] = Regress('regress', self.params,
            self.num_features, self.num_classes, self.params.seed)

        self.start_ix = 0

        self._preprocess()
        sys.stdout.flush()

        self.Model.pop('Unit', None)

    def _build_base_graph(self):
        with tf.variable_scope(self.scope):

            self.Placeholder['Input_Feature'] = tf.placeholder(
                shape=[None, self.num_features], dtype=tf.float32,
            )

            self.Placeholder['Learning_Rate'] = tf.placeholder(
                tf.float32, []
            )

            self.Placeholder['Input_Label'] = tf.placeholder(
                tf.int32, [None, self.num_classes]
            )

            self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                [None, self.num_classes])

            self.Tensor['Loss_Function'] = \
                SoftmaxCE

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

        self._build_base_graph()
        self._fork_unit_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        info = self.Model['Regress'].Info['Params']
        type = self.Model['Regress'].Info['Type']

        feed_dict = {}

        nl = len(self.params.mlp_hidden_seq) + 1

        self.nh, self.ni = [], []
        self.Tensor['Sample'] = [None for _ in range(nl)]

        for i in range(nl):
            dummy_kernel, roll, sample_inds = self.Placeholder['MLP_Placeholders'][i]

            nh = info[i]['hidden_size']
            ni = info[i]['input_depth']

            self.nh.append(nh)
            self.ni.append(ni)

            matrix = get_init(self.params.mlp_init_type)(
                (ni, nh), self._npr, self.params.mlp_init_scale
            )

            self.Tensor['Sample'][i] = matrix

            shape = (ni, 0)
            indices = np.zeros((0, 2))
            values = np.zeros((0,))

            feed_dict[dummy_kernel] = matrix
            feed_dict[roll] = [0]
            feed_dict[sample_inds] = tf.SparseTensorValue(
                indices=indices, values=values, dense_shape=shape)

        feed_dict[self.Placeholder['Dummy_Inputs']] = \
            self._npr.normal(size=(self.params.dummy_batch, self.num_features), scale=_EPS)

        self.Tensor['Norm'], out = self.sess.run([self.Tensor['Dummy_Score'],
            self.Tensor['Dummy_States']], feed_dict
        )

        for i in range(nl):
            plt.imshow(out[1], cmap=plt.get_cmap('binary'))
            plt.savefig('{}/ds{}.png'.format(self.Dir, i))

            plt.clf()

            dummy_score = self.Tensor['Norm'][i] = \
                self.Tensor['Norm'][i].reshape(self.Tensor['Sample'][i].shape)

            plt.imshow(-dummy_score, cmap=plt.get_cmap('binary'))
            plt.savefig('{}/d{}.png'.format(self.Dir, i))

            plt.clf()

        grads = [np.zeros_like(w) for w in self.Tensor['Sample']]

        for i in range(self.params.eval_iter):
            features, labels = next(self._get_batch())

            feed_dict[self.Placeholder['Input_Feature']] = features
            feed_dict[self.Placeholder['Input_Label']] = labels

            if self.params.prune_criteria in ['jacobian', 'jacobian_easy']:
                scores = self.sess.run(
                    [self.Tensor['Unit_Hess']], feed_dict
                )[0]

                for j in range(nl):
                    grads[j] += np.reshape(scores[j], grads[j].shape)/ \
                                (self.params.eval_iter)/self.Tensor['Norm'][j]

            else:
                pass

        self.weights = [None for _ in range(nl)]

        for j in range(nl):
            t_ix = int((1-self.params.prune_k) * self.ni[j] * self.nh[j])

            plt.imshow(np.log(np.abs(self.Tensor['Sample'][j])), cmap=plt.get_cmap('binary'))
            plt.savefig('{}/w{}.png'.format(self.Dir, j))

            plt.clf()

            plt.imshow(-np.log(np.abs(grads[j])), cmap=plt.get_cmap('binary'))
            plt.savefig('{}/grad{}.png'.format(self.Dir, j))

            plt.clf()

            inds = top_row, top_col = np.unravel_index(
                np.argpartition(grads[j], t_ix, axis=None)[:t_ix],
                (self.ni[j], self.nh[j])
            )

            row_ind = top_row.astype(np.int32)
            col_ind = top_col.astype(np.int32)
            weights = self.Tensor['Sample'][j][inds]

            self.weights[j] = (weights, np.array([row_ind, col_ind]).T)

            self.plot(weights, row_ind, col_ind, [self.ni[j], self.nh[j]], j)

        sys.stdout.flush()

        self.sess.close()
        tf.reset_default_graph()

        self._build_base_graph()

        self._fork_model(self.weights)

        self.Sess = tf.Session()

    def _fork_unit_model(self):
        self.Model['Unit'].set_variables()

        self.Placeholder['MLP_Placeholders'] = \
            self.Model['Unit'].get_placeholders()

        self.loss_params = {}

        self.Tensor['Unit_Out'], self.Tensor['Unit_Hidden'] = \
            self.Model['Unit'].run_fork(
                self.Placeholder['Input_Feature'], return_hidden = True
            )

        self.Tensor['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Tensor['Unit_Out'], self.Placeholder['Input_Label'],
                **self.loss_params
            )
        )

        if self.params.prune_criteria == 'hess':
            pass

        elif self.params.prune_criteria == 'jacobian':
            pass

        elif self.params.prune_criteria == 'jacobian_easy':
            grad_var = self.Model['Unit'].hessian_variables()

            self.Placeholder['Dummy_Inputs'] = tf.placeholder(dtype=tf.float32,
                shape=[None, self.num_features])

            self.Tensor['Dummy_States'] = \
                self.Model['Unit'].run_dummy(self.Placeholder['Dummy_Inputs'])

            self.Tensor['Dummy_Score'] = \
                self.simple_jacobian_objective(
                    self.Tensor['Dummy_States'], grad_var
                )

            self.Tensor['Unit_Hess'] = self.simple_jacobian_objective(
                self.Tensor['Unit_Hidden'], grad_var
            )

    def simple_jacobian_objective(self, jacob_ys, grad_zs):
        def vjp(y, x, dx):
            return tf.gradients(y, x, tf.reshape(dx, tf.shape(y)))

        grad_obj = []
        for i in range(len(grad_zs)):
            grad_vec = tf.reduce_mean(tf.square(
                vjp(jacob_ys[i+1], jacob_ys[i], tf.ones_like(jacob_ys[i+1]))))

            grad = tf.abs(tf.reshape(tf.gradients(grad_vec, grad_zs[i]), [-1]))
            grad_obj.append(grad)

        return grad_obj

    def _fork_model(self, weights):
        self.Model['Regress'].fork_model_fnc(weights, 0)

        self.loss_params = {}

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

        self.Output['Round'] = \
            tf.argmax(self.Placeholder['Input_Logits'], 1)

        self.Output['Error'] = 1 - tf.reduce_mean(
            tf.cast(tf.equal(
                self.Output['Round'],
                tf.argmax(self.Placeholder['Input_Label'], 1)
            ), tf.float32)
        )

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

        plt.clf()