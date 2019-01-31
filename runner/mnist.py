import os.path as osp

import numpy as np
import tensorflow as tf
from runner.base_runner import *

from util.optimizer_util import *
from model.unit import *
import scipy.misc
from collections import defaultdict

from tensorflow.contrib import slim

from data.load_pen import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

def get_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = np.concatenate([x_train, x_test[:3000]], axis=0)
    y_train = np.concatenate([y_train, y_test[:3000]], axis=0)
    x_test = x_test[3000:]
    y_test = y_test[3000:]

    x_train /= 255
    x_test /= 255

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)

class MNISTRunner(BaseRunner):
    def __init__(self, scope, params):
        super(MNISTRunner, self).__init__(scope, params)
        self.Dataset = get_mnist_dataset()

        self._npr = np.random.RandomState(params.seed)
        self.Mask = {}
        self._build_snip()

        self._preprocess()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Writer['Random'] = \
            tf.summary.FileWriter(self.Dir+'/random', self.Sess.graph)
        self.Writer['Unit'] = \
            tf.summary.FileWriter(self.Dir+'/unit', self.Sess.graph)

    def _build_snip(self):
        with tf.variable_scope(self.scope):
            self.Model['Random'] = Unit('random', self.params,
                784, 10, self.params.seed)
            self.Model['Unit'] = Unit('unit', self.params,
                784, 10, self.params.seed)

            self.start_ix = 0

            self.Placeholder['Input_Feature'] = tf.placeholder(
                shape=[None, 784], dtype=tf.float32,
            )

            self.Placeholder['Learning_Rate'] = tf.placeholder(
                tf.float32, []
            )

            self.Placeholder['Input_Label'] = tf.placeholder(
                tf.int32, [None, 10]
            )

            self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32,
                [None, 10])

            self.Tensor['Proto_Minibatch'] = {
                'Features': self.Placeholder['Input_Feature'],
                'Labels': self.Placeholder['Input_Label']
            }

            self.Tensor['Loss_Function'] = \
                SoftmaxCE

            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

            self.Model['Unit'].unit(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function'],
                len(self.Model['Unit'].Snip['Dummy_Kernel']) - 1
            )

            self.Tensor['Unit_Grad'] = self.Model['Unit'].Tensor['Unit_Grad']
            self.Placeholder['Unit_Roll'] = self.Model['Unit'].Snip['Roll']

            self.Placeholder['Unit_Kernel'] = self.Model['Unit'].Snip['Dummy_Kernel']

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
        random_list = []

        features, labels = self._get_batch()
        nh = info['hidden_size']
        ni = info['input_depth']
        nu = min(self.params.num_unitwise_mlp, nh)

        if info['scope'] == 'softmax':
            k_ratio = self.params.softmax_sparsity
        elif i == len(self.Model['Unit'].Snip['Dummy_Kernel']) - 1:
            k_ratio = self.params.unit_k
        elif 'mlp' in info['scope']:
            k_ratio = self.params.unit_k
        else:
            k_ratio = 0.99

        h_ix = int((1 - k_ratio) * ni * nh / (nh // nu + 1))
        t_ix = h_ix * (nh // nu + 1)
        top_vals = np.zeros((t_ix, 3))
        ix = 0

        def xavier_initializer(shape, npr, stddev=1.0):
            out = npr.normal(scale=np.sqrt(2/(shape[0]+shape[1])), size=shape).astype(np.float32)
            return out

        grad_im = np.zeros((ni, nu*(nh//nu+1)))

        for j in range(nh // nu + 1):
            weights = xavier_initializer((ni, nu), self._npr)

            feed_dict = {
                self.Placeholder['Unit_Kernel'][i]: weights,
                self.Placeholder['Unit_Roll'][i]: [j*nu],
                self.Placeholder['Input_Feature']: features,
                self.Placeholder['Input_Label']: labels,
            }

            grads, pred = self.Sess.run(
                [self.Tensor['Unit_Grad'][i], self.Model['Unit'].Tensor['Unit_Pred']], feed_dict
            )
            grads = grads[0]
            grad_im[:,j*nu:(j+1)*nu] = grads

            top_k = np.unravel_index(
                np.argpartition(np.abs(grads), -h_ix, axis=None)[-h_ix:],
                (ni, nu)
            )

            for k in range(len(top_k[0])):
                l, m = top_k[0][k], top_k[1][k]
                if j * nu + m >= nh:
                    # just take some random weight
                    top_vals[ix] = [weights[l][m], self._npr.randint(ni, size=(1,)),
                        self._npr.randint(nh, size=(1,))]

                else:
                    top_vals[ix] = [weights[l][m], l, m+j*nu]

                ix += 1

        random_list = [self._npr.randn(t_ix, ),
                       np.vstack([self._npr.randint(ni, size=(t_ix,)),
                                  self._npr.randint(nh, size=(t_ix,))]).T
                       ]
        final_list = [top_vals[:, 0], top_vals[:, 1:]]
        im = np.zeros((ni, nh))
        im[top_vals[:,1].astype(np.int32), top_vals[:,2].astype(np.int32)] = 1

        scipy.misc.imsave(osp.join(self.Dir, 'grad_{}.jpg'.format(info['scope'])), grad_im)
        scipy.misc.imsave(osp.join(self.Dir, '{}.jpg'.format(info['scope'])), im)

        self._build_networks(final_list, random_list, i, use_dense=True)
        self.Sess.run(self.Tensor['Variable_Initializer'])

    def _build_networks(self, unit_list, random_list, i, use_dense=True):
        self.Model['Random'].build_sparse(random_list, i, use_dense)
        self.Model['Unit'].build_sparse(unit_list, i, use_dense)

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

            self.Output['Random_Pred'] = self.Model['Random'].run(
                self.Placeholder['Input_Feature']
            )

            self.Output['Unit_Pred'] = self.Model['Unit'].run(
                self.Placeholder['Input_Feature']
            )

            self.Output['Random_Loss'] = tf.reduce_mean(
               self.Tensor['Loss_Function'](
                   self.Output['Random_Pred'], self.Placeholder['Input_Label']
               )
            )

            self.Output['Unit_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Unit_Pred'], self.Placeholder['Input_Label']
                )
            )

            self.Output['Random_Loss'] += self.params.weight_decay * \
            tf.reduce_sum(
                [tf.nn.l2_loss(t) for t in self.Model['Random'].Tensor['Weights']]
            )
            self.Output['Unit_Loss'] += self.params.weight_decay * \
              tf.reduce_sum(
                  [tf.nn.l2_loss(t) for t in self.Model['Unit'].Tensor['Weights']]
              )

            self.Output['Random_Train'] = \
               self.Output['Optimizer'].minimize(self.Output['Random_Loss'])
            self.Output['Unit_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['Unit_Loss'])

    def _build_summary(self):
        self.Output['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Placeholder['Input_Logits'],
                self.Placeholder['Input_Label']
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

        self.Summary['Train_Error'] = tf.summary.scalar(
            'Train_Error', self.Output['Error']
        )
        self.Summary['Val_Error'] = tf.summary.scalar(
            'Val_Error',  self.Output['Error']
        )

        self.Summary['Train_Loss'] = tf.summary.scalar(
            'Train_Loss', tf.log(self.Output['Loss'])
        )
        self.Summary['Val_Loss'] = tf.summary.scalar(
            'Val_Loss', tf.log(self.Output['Loss'])
        )

        #self.Summary['Weight'] = {}
        #for key in ['Random', 'Unit']:
        #    self.Summary['Weight'][key] = [
        #        tf.summary.histogram(weight.name,
        #            tf.boolean_mask(weight, tf.not_equal(weight, ZERO_32)))
        #        for weight in self.Model.Sparse
        #    ]

        self.Output['Pred'] = {
            'Random': self.Output['Random_Pred'],
            'Unit': self.Output['Unit_Pred']
        }

        self.train_summary = [
            self.Summary['Train_Error'],
            self.Summary['Train_Loss']
        ]
        self.val_summary = [
            self.Summary['Val_Error'],
            self.Summary['Val_Loss']
        ]
        self.train_op = [
            self.Output['Random_Train'],
            self.Output['Unit_Train']
        ]

    def train(self, i, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)
        # print(features, labels)

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        pred, *_ = self.Sess.run(
            [self.Output['Pred']] + self.train_op,
            feed_dict
        )
        #self.Writer['Unit'].add_run_metadata(self.Sess.rmd, 'train' + str(i))
        for key in pred:
            summary = self.Sess.run(
                self.train_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )

            for summ in summary:
                self.Writer[key].add_summary(summ, i)
                self.Writer[key].flush()

        self.learning_rate = self.decay_lr(i, self.learning_rate)
        return features, labels

    def val(self, i):
        features, labels = self.Dataset[1]
        # self.Dataset.test.images, self.Dataset.test.labels

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
        }
        pred = self.Sess.run(
            [self.Output['Pred']], feed_dict)

        pred = pred[0]
        for key in pred:
            summary = self.Sess.run(
                self.val_summary,
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )
            for summ in summary:
                self.Writer[key].add_summary(summ, i)

    def run(self):
        #slim.model_analyzer.analyze_vars(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), print_info=True)

        for e in range(self.params.num_steps):
            features, labels = self._get_batch()
            self.train(e, features, labels)
            if e % self.params.val_steps == 0:
                self.val(e)

    def decay_lr(self, i, learning_rate):
        if self.params.decay_scheme == 'exponential':
            if (i+1) % self.params.decay_iter == 0:
                learning_rate *= self.params.decay_rate

        elif self.params.decay_scheme == 'none':
            pass

        return learning_rate


    def _get_batch(self):
        end_ix = self.start_ix + self.params.batch_size
        if end_ix > len(self.Dataset[0][0]):
            end_ix = end_ix - len(self.Dataset[0][0])

            features = np.concatenate(
                [self.Dataset[0][0][self.start_ix:],
                 self.Dataset[0][0][:end_ix]],
                axis=0
            )
            labels = np.concatenate(
                [self.Dataset[0][1][self.start_ix:],
                 self.Dataset[0][1][:end_ix]],
                axis=0
            )
        else:
            features = self.Dataset[0][0][self.start_ix:end_ix]
            labels = self.Dataset[0][1][self.start_ix:end_ix]
        self.start_ix = end_ix
        return features, labels