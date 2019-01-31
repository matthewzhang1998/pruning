import os
import os.path as osp

import tensorflow as tf
from tensorflow.python.client import timeline
from model.random import *

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

class BaseRunner(object):
    def __init__(self, scope, params):
        self.params = params
        self.scope = scope

        self.Tensor = {}
        self.Model = {}
        self.Placeholder = {}
        self.Output = {}
        self.Summary = {}

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth=True

        self.Dir = get_dir(params.log_dir)

        self.Sess = LogSession(tf.Session(config=config), self.Dir)

    def preprocess(self):
        raise NotImplementedError

    def train(self, i):
        raise NotImplementedError

    def val(self, i):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

class LogSession(object):
    def __init__(self, Session, dir='/log'):
        #self.rmd = tf.RunMetadata()
        #self.opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.Sess = Session
        self.dir = dir

        self.i = 0

        self.profiler = True
        self.graph = Session.graph
        #self.trace_file = tf.gfile.Open(name=osp.join(self.dir,'timeline'), mode='w')

    def run(self, outputs, feed_dict={}):
        vals = self.Sess.run(outputs, feed_dict,) #options=self.opt, run_metadata=self.rmd)

        return vals

