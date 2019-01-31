import numpy as np
import tensorflow as tf
from model.mask import *
from runner.base_runner import *
from util.optimizer_util import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

class AddRunner(BaseRunner):
    def __init__(self, scope, params):
        super(AddRunner, self).__init__(scope, params)

        self.Mask = {}

        self._build_outputs()
        self._build_summary()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self._npr = np.random.RandomState(params.seed)

        self.learning_rate = params.learning_rate
        self.pretrain_learning_rate = params.pretrain_learning_rate

        self.Writer['Random'] = \
            tf.summary.FileWriter(self.Dir+'/random', self.Sess.graph)
        self.Writer['Snip'] = \
            tf.summary.FileWriter(self.Dir+'/snip', self.Sess.graph)

    def _build_outputs(self):
        with tf.variable_scope(self.scope):
            self.Model['Snip'] = Mask('snip', self.params, 2, 1, self.params.seed)
            self.Model['Random'] = Mask('random', self.params, 2, 1,
                self.params.seed + 1562)

            self.start_ix = 0

            self.Placeholder['Input_Feature'] = tf.placeholder(
                tf.float32, [None, None, 2]
            )
            self.Placeholder['Learning_Rate'] = tf.placeholder(
                tf.float32, []
            )

            self.Placeholder['Input_Label'] = tf.placeholder(
                tf.float32, [None, 1]
            )

            self.Placeholder['Input_Logits'] = tf.placeholder(tf.float32, [None, 1])

            self.Tensor['Proto_Minibatch'] = {
                'Features': self.Placeholder['Input_Feature'],
                'Labels': self.Placeholder['Input_Label']
            }

            self.Tensor['Loss_Function'] = \
                MSELoss
            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

            self.Model['Snip'].prune(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function']
            )
            self.Model['Random'].prune(
                self.Tensor['Proto_Minibatch'], self.Tensor['Loss_Function']
            )

            self.Placeholder['Random_Mask'] = self.Model['Random'].Placeholder
            self.Placeholder['Snip_Mask'] = self.Model['Snip'].Placeholder

            self.Tensor['Snip_Grad'] = self.Model['Snip'].Tensor['Snip_Grad']
            self.Tensor['Random_Grad'] = self.Model['Random'].Tensor['Snip_Grad']

            self.Output['Random_Pred'] = self.Model['Random'].run(
                self.Placeholder['Input_Feature']
            )
            self.Output['Snip_Pred'] = self.Model['Snip'].run(
                self.Placeholder['Input_Feature']
            )

            self.Output['Random_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Random_Pred'], self.Placeholder['Input_Label']
                )
            )
            self.Output['Snip_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Snip_Pred'], self.Placeholder['Input_Label']
                )
            )

            self.Output['Random_Loss'] += self.params.weight_decay * \
                tf.reduce_sum(
                    [tf.nn.l2_loss(t) for t in self.Model['Random'].Snip['Weight']]
                )

            self.Output['Snip_Loss'] += self.params.weight_decay * \
                tf.reduce_sum(
                    [tf.nn.l2_loss(t) for t in self.Model['Snip'].Snip['Weight']]
                )

            self.Output['Random_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['Random_Loss'])
            self.Output['Snip_Train'] = \
                self.Output['Optimizer'].minimize(self.Output['Snip_Loss'])

    def _build_summary(self):
        self.Output['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Placeholder['Input_Logits'],
                self.Placeholder['Input_Label']
            )
        )

        self.Output['Error'] = tf.reduce_mean(
            tf.abs(self.Placeholder['Input_Logits'] - self.Placeholder['Input_Label'])
        )

        self.Summary['Train_Error'] = tf.summary.scalar(
            'Train_Error', self.Output['Error']
        )
        self.Summary['Val_Error'] = tf.summary.scalar(
            'Val_Error', self.Output['Error']
        )

        self.Summary['Train_Loss'] = tf.summary.scalar(
            'Train_Loss', tf.log(self.Output['Loss'])
        )
        self.Summary['Val_Loss'] = tf.summary.scalar(
            'Val_Loss', tf.log(self.Output['Loss'])
        )

        self.Summary['Weight'] = {}
        for key in ['Snip', 'Random']:
            self.Summary['Weight'][key] = [
                tf.summary.histogram(weight.name,
                    tf.boolean_mask(weight, tf.not_equal(weight, ZERO_32)))
                for weight in self.Model[key].Snip['Comb']
            ]

        self.Output['Pred'] = {
            'Snip': self.Output['Snip_Pred'],
            'Random': self.Output['Random_Pred'],
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
            self.Output['Snip_Train'],
            self.Output['Random_Train'],
        ]

    def preprocess(self, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)

        feed_dict = {
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
        }
        self.Sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        self.Mask['Random'] = {}
        self.Mask['Delta'] = {}

        random_masks = self.Sess.run(self.Model['Random'].Snip['Weight'])
        delta_masks = self.Sess.run(self.Model['Snip'].Snip['Weight'])
        for ix, (mask, delta) in enumerate(zip(random_masks, delta_masks)):
            binary = self._npr.binomial(1, 1 - self.params.random_k, mask.shape)
            self.Mask['Random'][self.Model['Random'].Snip['Mask'][ix]] = \
                np.ones(delta.shape)

            perturb = np.ones(delta.shape)

            self.Mask['Delta'][self.Model['Snip'].Snip['Mask'][ix]] = \
                perturb

        self.Mask['Index'] = self.Sess.run(self.Tensor['Snip_Grad'],
           {**self.Mask['Random'], **self.Mask['Delta'], **feed_dict}
        )

        self.Mask['Snip'] = {}

        if self.params.prune_method == 'together':
            self.prune_together()

        elif self.params.prune_method == 'separate' or 'weighted':
            self.prune_separate()

    def train(self, i, features, labels):
        # self.Dataset.train.next_batch(self.params.batch_size)
        # print(features, labels)

        feed_dict = {
            **self.Mask['Random'],
            **self.Mask['Snip'],
            self.Placeholder['Input_Feature']: features,
            self.Placeholder['Input_Label']: labels,
            self.Placeholder['Learning_Rate']: self.learning_rate
        }
        pred, *_ = self.Sess.run(
            [self.Output['Pred']] + self.train_op,
            feed_dict
        )
        for key in pred:
            summary = self.Sess.run(
                self.train_summary + self.Summary['Weight'][key],
                {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
            )

            for summ in summary:
                self.Writer[key].add_summary(summ, i)

        self.learning_rate = self.decay_lr(i, self.learning_rate)
        return features, labels

    def val(self, i):
        features, labels = self._get_batch()
        # self.Dataset.test.images, self.Dataset.test.labels

        feed_dict = {
            **self.Mask['Random'],
            **self.Mask['Snip'],
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
        features, labels = self._get_batch()
        self.preprocess(features, labels)

        for e in range(self.params.num_steps):
            features, labels = self._get_batch()
            self.train(e, features, labels)
            if e % self.params.val_steps == 0:
                self.val(e)

    def _get_batch(self):
        n_batch = self.params.batch_size
        min_length = self.params.min_length
        max_length = self.params.max_length
        X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                            np.zeros((n_batch, max_length, 1))],
                           axis=-1)
        mask = np.zeros((n_batch, max_length))
        y = np.zeros((n_batch,))
        # Compute masks and correct values
        for n in range(n_batch):
            # Randomly choose the sequence length
            length = np.random.randint(min_length, max_length)
            # Make the mask for this sample 1 within the range of length
            mask[n, :length] = 1
            # Zero out X after the end of the sequence
            X[n, length:, 0] = 0
            # Set the second dimension to 1 at the indices to add
            X[n, np.random.randint(length / 10), 1] = 1
            X[n, np.random.randint(length / 2, length), 1] = 1
            # Multiply and sum the dimensions of X to get the target value
            y[n] = np.sum(X[n, :, 0] * X[n, :, 1])
        # Center the inputs and outputs
        X -= X.reshape(-1, 2).mean(axis=0)
        y -= y.mean()

        return X,y[:,np.newaxis]

    def decay_lr(self, i, learning_rate):
        if self.params.decay_scheme == 'exponential':
            if (i+1) % self.params.decay_iter == 0:
                learning_rate *= self.params.decay_rate

        elif self.params.decay_scheme == 'none':
            pass

        return learning_rate

    def prune_together(self):
        total_shape, total_size, flat_grad = [], [], []
        for ix in range(len(self.Mask['Index'])):
            grad = self.Mask['Index'][ix]
            total_shape.append(grad.shape)
            total_size.append(grad.size)
            flat_grad.append(grad.flatten())

        flat_size = 1
        for size in total_size:
            flat_size += size

        flat_grad = np.concatenate(flat_grad, axis=0)

        k = int((1 - self.params.prune_k) * flat_size)

        zeros = np.zeros_like(flat_grad)

        if self.params.value_method == 'largest':
            ind = np.argpartition(np.abs(flat_grad), -k, axis=None)[-k:]
        elif self.params.value_method == 'smallest':
            ind = np.argpartition(-np.abs(flat_grad), -k, axis=None)[-k:]
        elif self.params.value_method == 'mixed':
            l = int(k / 2)
            ind = np.concatenate(
                [np.argpartition(np.abs(flat_grad), -k, axis=None)[-l:],
                 np.argpartition(np.abs(flat_grad), -k, axis=None)[:k - l]]
            )
        elif self.params.value_method == 'weighted':
            raise ValueError('Cannot prune together with weighted')

        zeros[ind] = 1

        start = 0

        for ix, (shape, size) in enumerate(zip(total_shape, total_size)):
            print(ix)
            end = start + size
            self.Mask['Snip'][self.Model['Snip'].Snip['Mask'][ix]] = \
                np.reshape(zeros[start:end], shape)
            start = end

    def prune_separate(self):
        for ix in range(len(self.Mask['Index'])):
            grad = self.Mask['Index'][ix]
            k = int((1 - self.params.prune_k) * grad.size)
            zeros = np.zeros_like(grad)

            if self.params.prune_method == 'weighted':
                k = (1 - self.params.prune_k) * grad.size
                k *= (ix / len(self.Mask['Index']) + 0.5)
                k = int(k)

            if self.params.value_method == 'largest':
                ind = np.argpartition(np.abs(grad), -k, axis=None)[-k:]
            elif self.params.value_method == 'smallest':
                ind = np.argpartition(-np.abs(grad), -k, axis=None)[-k:]
            elif self.params.value_method == 'mixed':
                l = int(k / 2)
                ind = np.concatenate(
                    [np.argpartition(np.abs(grad), -k, axis=None)[-l:],
                     np.argpartition(np.abs(grad), -k, axis=None)[:k - l]]
                )
            ind = np.unravel_index(ind, zeros.shape)
            zeros[ind] = 1
            zeros[:2,:] = 1

            self.Mask['Snip'][self.Model['Snip'].Snip['Mask'][ix]] = zeros
            self.Mask['Snip'][self.Model['Snip'].Snip['Mask'][ix]] = np.ones_like(self.Mask['Index'][ix])
