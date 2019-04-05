import os.path as osp

import numpy as np
import tensorflow as tf
from model.vanilla import *
from runner.base_runner import *
from util.optimizer_util import *
from util.logger_util import *
from model.snip import *
from util.sparse_util import *
import scipy.misc
from collections import defaultdict

from tensorflow.contrib import slim

from data.load_pen import *

ZERO_32 = tf.constant(0.0, dtype=tf.float32)

class VanillaRunner(BillionRunner):
    def _build_snip(self):
        initializer = tf.random_uniform_initializer(
            -self.params.rnn_init_scale, self.params.rnn_init_scale)

        with tf.variable_scope(self.scope, initializer=initializer):
            self.Model['Small'] = Vanilla('small', self.params,
                self.vocab_size, self.vocab_size, init_path=self.params.weight_dir)

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

            self.Tensor['Loss_Function'] = \
                Seq2SeqLoss

            self.Output['Optimizer'] = get_optimizer(
                self.params, self.Placeholder['Learning_Rate']
            )

            self.Output['Small_Pred'] = self.Model['Small'].run(
                self.Placeholder['Input_Feature']
            )

            self.Output['Small_Loss'] = tf.reduce_mean(
                self.Tensor['Loss_Function'](
                    self.Output['Small_Pred'], self.Placeholder['Input_Label']
                )
            )

            self.Tensor['Train_Var'] = tf.trainable_variables()

            self.Output['Small_Grad'], _ = tf.clip_by_global_norm(
                tf.gradients(self.Output['Small_Loss'], self.Tensor['Train_Var']),
                self.params.max_grad
            )

            self.Output['Small_Train'] = self.Output['Optimizer'].apply_gradients(
                zip(self.Output['Small_Grad'], self.Tensor['Train_Var']),
                global_step=tf.train.get_or_create_global_step()
            )