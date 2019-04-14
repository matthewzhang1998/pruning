import os
import os.path as osp

from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from model.random import *
from util.sparse_util import _compute_factor_logits

from util.logger_util import *

def SoftmaxCE(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels
    )

def SoftmaxSliceCE(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits[:,-1], labels=labels
    )

def MSELoss(logits, labels):
    return tf.losses.mean_squared_error(labels, logits)

def Seq2SeqLoss(logits, labels):
    return tf.contrib.seq2seq.sequence_loss(
        logits, labels, tf.ones_like(logits[:,:,-1], dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=True
    )

def SampleSoftmaxSequenceLoss(logits, labels, weight, bias, num_sample, vocab_size, **kwargs):
    classes = tf.shape(logits)[-1]
    logits_flat = tf.reshape(logits, [-1, classes])
    print(logits_flat)
    labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.int64)
    loss = tf.nn.sampled_softmax_loss(
        weight, bias, labels, logits_flat, num_sample, vocab_size
    )

    return tf.reduce_mean(loss)

def SSSLossWithFactorization(logits, labels, weight, bias, num_sample, vocab_size, num_factor):
    print(logits)
    classes = logits.get_shape().as_list()[-1]
    logits_flat = tf.reshape(logits, [-1, classes])
    labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.int64)
    print(logits_flat)
    logits, labels = _compute_factor_logits(
        weights=weight,
        biases=bias,
        labels=labels,
        inputs=logits_flat,
        num_sampled=num_sample,
        num_classes=vocab_size,
        num_true=1,
        sampled_values=None,
        subtract_log_q=True,
        partition_strategy="div",
        num_factor=num_factor,
        seed=None)
    labels = tf.stop_gradient(labels, name="labels_stop_gradient")
    sampled_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits)

    return tf.reduce_mean(sampled_losses)

class BaseRunner(object):
    def __init__(self, scope, params):
        self.params = params
        self.scope = scope

        self.Tensor = {}
        self.Model = {}
        self.Placeholder = {}
        self.Output = {}
        self.Summary = {}

        self.Debug = []

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options = gpu_options)
        config.gpu_options.allow_growth=True

        self.Dir = get_dir(params.log_dir)

        self.Sess = LogSession(tf.Session(config=config), self.Dir, log_memory=params.log_memory)

    def preprocess(self):
        raise NotImplementedError

    def train(self, i):
        raise NotImplementedError

    def val(self, i):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

class LanguageRunner(BaseRunner):
    def __init__(self, scope, params):
        super(LanguageRunner, self).__init__(scope, params)
        self.loss_params = {}

        self._npr = np.random.RandomState(params.seed)
        self.Mask = {}
        self.Data, self.vocab_size = self._get_dataset()
        self._build_snip()

        self._build_summary()
        self.Writer = {}

        self.Saver = tf.train.Saver()

        self.learning_rate = params.learning_rate

        if params.log_memory:
            self.GraphWriter = {
                key: tf.summary.FileWriter(self.Dir+'/graph/'+ key, self.Sess.graph)
                for key in self.Model
            }

        self.Writer = {
            key: FileWriter(self.Dir+'/epoch/'+key, self.Sess.graph) for key in self.Model
        }

        self.BatchWriter = {
            key: FileWriter(self.Dir+'/batch/'+key, self.Sess.graph) for key in self.Model
        }
        self.batch_ix = {'train': 0, 'val': 0}
        self.epoch_ix = {'train': 0}

    def _build_summary(self):
        self.Output['Loss'] = tf.reduce_mean(
            self.Tensor['Loss_Function'](
                self.Placeholder['Input_Logits'],
                self.Placeholder['Input_Label'],
                **self.loss_params
            )
        )

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

        self.Output['Error'] = tf.exp(self.Output['Loss'])

        self.train_res = {
            'Train_Error': self.Output['Error'],
            'Train_Loss': self.Output['Loss']
        }

        self.val_res = {
            'Val_Error': self.Output['Error'],
            'Val_Loss': self.Output['Loss']
        }

        self.train_placeholder = {
            'Train_Error': self.Placeholder['Train_Error'],
            'Train_Loss': self.Placeholder['Train_Loss']
        }

        self.val_placeholder = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': self.Placeholder['Val_Loss']
        }

        self.Summary['Train_Error'] = tf.summary.scalar(
            'Train_Error', self.Output['Error']
        )
        self.Summary['Val_Error'] = tf.summary.scalar(
            'Val_Error', self.Placeholder['Val_Error']
        )

        self.Summary['Train_Loss'] = tf.summary.scalar(
            'Train_Loss', tf.log(self.Output['Loss'])
        )
        self.Summary['Val_Loss'] = tf.summary.scalar(
            'Val_Loss', tf.log(self.Placeholder['Val_Loss'])
        )

        self.Output['Pred'] = {
            key: self.Output[key + '_Pred'] for key in self.Model
        }

        self.train_summary = {
            'Train_Error': self.Placeholder['Train_Error'],
            'Train_Loss': self.Placeholder['Train_Loss']
        }
        self.val_summary = {
            'Val_Error': self.Placeholder['Val_Error'],
            'Val_Loss': self.Placeholder['Val_Loss']
        }
        self.train_op = [
            self.Output[key +'_Train'] for key in self.Model
        ]

    def train(self, i):
        start = 0

        summary = {key: defaultdict(list) for key in self.Writer}

        for (b_feat, b_lab) in self._get_batch('train'):

            b_feat = np.array(b_feat)
            b_lab = np.array(b_lab)
            if np.any(np.isnan(b_feat)) or np.any(np.isnan(b_lab)):
                assert False
            self.batch_ix['train'] += 1

            if self.batch_ix['train'] % self.params.val_steps == 0:
                self.val(self.batch_ix['train'])

            feed_dict = {
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab,
                self.Placeholder['Learning_Rate']: self.learning_rate
            }
            pred = self.Sess.run(
                [self.Output['Pred']]+self.train_op, feed_dict)

            pred = pred[0]
            for key in pred:
                b_summary = self.Sess.run(
                    self.train_res,
                    {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
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
        start = 0
        summary = {key: defaultdict(list) for key in self.Writer}

        for (b_feat, b_lab) in self._get_batch('val'):
            self.batch_ix['val'] += 1
            feed_dict = {
                self.Placeholder['Input_Feature']: b_feat,
                self.Placeholder['Input_Label']: b_lab,
            }
            pred = self.Sess.run(
                [self.Output['Pred']], feed_dict)

            pred = pred[0]
            for key in pred:
                b_summary = self.Sess.run(
                    self.val_res,
                    {**feed_dict, self.Placeholder['Input_Logits']: pred[key]}
                )
                self.BatchWriter[key].add_summary(b_summary, self.batch_ix['val'])

                for summ in b_summary:
                    summary[key][summ].append(b_summary[summ])

        for key in summary:
            for summ in summary[key]:
                summary[key][summ] = np.mean(summary[key][summ])
                print(i, summ, summary[key][summ])

            write_summary = self.Sess.run(
                self.val_summary,
                {self.val_placeholder[summ]: summary[key][summ]
                 for summ in summary[key]}
            )
        self.Writer[key].add_summary(write_summary, i)

    def run(self):
        self.Sess.run(tf.global_variables_initializer())

        for e in range(self.params.num_steps):
            self.train(e)

    def decay_lr(self, i, learning_rate):
        if self.params.decay_scheme == 'exponential':
            if (i+1) % self.params.decay_iter == 0:
                learning_rate *= self.params.decay_rate ** max(i+1-self.params.start_epoch, 0.0)

        elif self.params.decay_scheme == 'none':
            pass

        return learning_rate

from data.load_pen import *

class PTBRunner(LanguageRunner):
    def _get_dataset(self):
        D = Dataset(self.params, "../data/ptb/data")
        return D, D.vocab_size

    def _get_batch(self, type='train'):
        return self.Data.get_batch(type)

from data.load_1b import *

class BillionRunner(LanguageRunner):
    def _get_dataset(self):
        train, val = load_1b("../data/1b")
        Dataset = {'train': train, 'val': val}
        return Dataset, self.params.vocab_size

    def _get_batch(self, key='train'):
        return self.Data[key].iterate_once(
            self.params.batch_size, self.params.max_length, self.params.vocab_size
        )

class LogSession(object):
    def __init__(self, Session, dir='/log', log_memory=1):
        self.log_memory = log_memory
        if log_memory:
            self.rmd = tf.RunMetadata()
            self.opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.Sess = Session
        self.dir = dir

        self.i = 0

        self.profiler = True
        self.graph = Session.graph
        #self.trace_file = tf.gfile.Open(name=osp.join(self.dir,'timeline'), mode='w')

    def run(self, outputs, feed_dict={}):
        if self.log_memory:
            vals = self.Sess.run(outputs, feed_dict, options=self.opt, run_metadata=self.rmd)
        else:
            vals = self.Sess.run(outputs, feed_dict)

        return vals

