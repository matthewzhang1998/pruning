import os.path as osp

import numpy as np
import tensorflow as tf
from runner.base_runner import *
from util.optimizer_util import *
from model.unit import *
from util.logger_util import *
from util.initializer_util import *
from collections import defaultdict

from data.load_pen import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

class SparseRunner(BillionRunner):
    def _build_snip(self):
        with tf.variable_scope(self.scope):
            self.Model['Unit'] = Unit('unit', self.params,
                self.vocab_size, self.vocab_size, self.params.seed)

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

            self.Model['Unit'].unit(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function'],
                len(self.Model['Unit'].Snip['Dummy_Kernel']) - 1
            )

            self.Tensor['Unit_Grad'] = self.Model['Unit'].Tensor['Unit_Grad']

            self.Placeholder['Unit_Kernel'] = self.Model['Unit'].Snip['Dummy_Kernel']
            self.Placeholder['Unit_Rotate'] = self.Model['Unit'].Snip['Dummy_Roll']

            self.Tensor['Variable_Initializer'] = {}

    def _preprocess(self):
        self.Sess.run(tf.global_variables_initializer())

        for i in reversed(range(len(self.Model['Unit'].Snip['Dummy_Kernel']))):
            self._preprocess_unit(i)

        self._build_summary()
        self.Sess.run(tf.variables_initializer(self.Output['Optimizer'].variables()))

    def _preprocess_unit(self, i):
        info = self.Model['Unit'].Info['Params'][i]
        type = self.Model['Unit'].Info['Type'][i]

        final_list = []

        features, labels = self._get_batch('train')[0]
        if type == 'rnn':
            use_dense = self.params.rnn_use_dense

            if 'lstm' in info['recurrent_cell_type']:
                nh = info['hidden_size']
                ni = info['input_depth']
                nu = self.params.num_unitwise_rnn
                nu = nh if nu > nh else nu

                h_ix = int((1-self.params.prune_k)*(ni+nh)*4*nh/(nh//nu+1))
                t_ix = h_ix*(nh//nu+1)
                top_vals = np.zeros((t_ix, 3), dtype=np.float32)
                ix = 0

                if self.params.rnn_prune_method in ['unit', 'block']:
                    if self.params.rnn_prune_method == 'block':
                        b_ix = int(h_ix * self.params.block_k * nu/(ni+nh))
                        r_ix = h_ix - 4 * b_ix

                        assert (r_ix > 0)

                    for j in range(nh//nu+1):
                        weights = get_init(self.params.rnn_init_type)(
                            (ni+nh,4*nu), self._npr, self.params.rnn_init_scale
                        )

                        feed_dict = {
                            self.Placeholder['Unit_Kernel'][i]: weights,
                            self.Placeholder['Input_Feature']: features,
                            self.Placeholder['Input_Label']: labels,
                            self.Placeholder['Unit_Rotate'][i]: [j*nu]
                        }
                        grads, pred = self.Sess.run(
                            [self.Tensor['Unit_Grad'][i], self.Model['Unit'].Tensor['Unit_Pred']], feed_dict
                        )

                        grads = grads[0]

                        # scipy.misc.imsave(osp.join(self.Dir, 'grad{}.jpg'.format(info['scope'])), grads)

                        if self.params.rnn_prune_method == 'unit':
                            top_k = np.unravel_index(
                                np.argpartition(np.abs(grads), -h_ix, axis=None)[-h_ix:],
                                (ni+nh,4*nu)
                            )
                            for k in range(len(top_k[0])):
                                l,m = top_k[0][k], top_k[1][k]
                                if j*nu + m%nu >= nh:
                                    # ignore
                                    top_vals[ix] = [0,0,0]
                                else:
                                    top_vals[ix] = [weights[l][m], l, m%nu + j*nu + m//nu*nh]

                                ix += 1

                        elif self.params.rnn_prune_method == 'block':
                            block = grads[ni+j*nu:max(ni+(j+1)*nu, ni+nh),:]
                            grads[ni+j*nu:max(ni+(j+1)*nu, ni+nh),:] = 0

                            top_r_k1, top_r_k2 = np.unravel_index(
                                np.argpartition(np.abs(grads), -r_ix, axis=None)[-r_ix:],
                                (ni+nh, 4*nu)
                            )
                            top_b_k1, top_b_k2 = np.unravel_index(
                                np.argpartition(np.abs(block), -b_ix, axis=None)[-b_ix:],
                                block.shape
                            )
                            top_b_k1 += ni+j*nu

                            top_k = (np.concatenate([top_r_k1, top_b_k1]),
                                np.concatenate([top_r_k2, top_b_k2]))

                            for k in range(len(top_k[0])):
                                l, m = top_k[0][k], top_k[1][k]
                                if j * nu + m % nu >= nh:
                                    # ignore
                                    top_vals[ix] = [0, 0, 0]
                                else:
                                    top_vals[ix] = [weights[l][m], l, m % nu + j * nu + m // nu * nh]

                                ix += 1

                elif self.params.prune_method == 'random':
                    weights = get_init(self.params.rnn_init_type)(
                        t_ix, self._npr, self.params.rnn_init_scale
                    )

                    n = np.random.choice(np.arange((ni+nh)*4*nh), size=t_ix, replace=False)
                    inds = np.unravel_index(n, (ni+nh, 4*nh))

                    for k in range(len(weights)):
                        top_vals[k] = [weights[k], inds[0][k], inds[1][k]]

                if self.params.rnn_use_dense:
                    top_list = np.zeros((ni+nh, 4*nh))
                    top_list[top_vals[:,1].astype(np.int32),
                        top_vals[:,2].astype(np.int32)] = top_vals[:,0]

                else:
                    top_list = [top_vals[:,0], top_vals[:, 1:]]

        elif type == 'mlp' and info['scope'] != 'softmax':
            use_dense = True
            nh = info['hidden_size']
            ni = info['input_depth']

            weights = get_init(self.params.rnn_init_type)(
                (ni,nh), self._npr, self.params.rnn_init_scale
            )
            top_list = weights

            # scipy.misc.imsave(osp.join(self.Dir, '{}.jpg'.format(info['scope'])), weights)

        elif type == 'embedding' or info['scope'] == 'softmax':
            use_dense = True
            nh = info['hidden_size']
            ni = info['input_depth']

            weights = get_init(self.params.rnn_init_type)(
                (ni, nh), self._npr, self.params.rnn_init_scale
            )
            top_list = weights

            # scipy.misc.imsave(osp.join(self.Dir, '{}.jpg'.format(info['scope'])), weights)

        self._build_networks(top_list, i, use_dense=use_dense)
        self.Sess.run(self.Tensor['Variable_Initializer'])

    def _build_networks(self, unit_list, i, use_dense=False):
        self.Model['Unit'].build_sparse(unit_list, i, use_dense=use_dense)

        if i != 0:
            self.Model['Unit'].unit(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function'],
                i - 1
            )

            self.Tensor['Unit_Grad'] = self.Model['Unit'].Tensor['Unit_Grad']

            self.Placeholder['Unit_Kernel'] = self.Model['Unit'].Snip['Dummy_Kernel']

            for key in self.Model:
                self.Tensor['Variable_Initializer'][key] = self.Model[key].initialize_op

        else:
            for key in self.Model:
                self.Tensor['Variable_Initializer'][key] = self.Model[key].initialize_op

            self.Output['Unit_Pred'] = self.Model['Unit'].run(
                self.Placeholder['Input_Feature']
            )

            self.Output['Unit_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Unit_Pred'], self.Placeholder['Input_Label']
                )
            )
            self.Output['Unit_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['Unit_Loss'])