import tensorflow as tf
import numpy as np
import init_path
from util.sparse_util import *

IN_SHAPE = 100
OUT_SHAPE = 100
BATCH = 10
TIME = 10

class test_op(object):
    def __init__(self):
        return

    def init_model(self):
        with tf.device("/cpu:0"):
            indices = np.array(np.unravel_index(np.arange(IN_SHAPE*4*OUT_SHAPE), (IN_SHAPE, 4*OUT_SHAPE))).T
            weights = np.ones((IN_SHAPE * 4*OUT_SHAPE))

        self.Net = SparseRecurrentNetwork("lstm", 'relu', 'none', 'lstm',
            (weights, indices), True, OUT_SHAPE, IN_SHAPE)
        self.var = tf.placeholder(shape=[None, None, IN_SHAPE], dtype=tf.float32)
        self.label = tf.placeholder(shape=[None, None, OUT_SHAPE], dtype=tf.float32)
        output, _ = self.Net(self.var)
        print(output)

        self.output = tf.reduce_mean(tf.square(output - self.label))
        self.opt = tf.train.AdamOptimizer().minimize(self.output)

        self.Sess = tf.Session()
        self.Sess.run(tf.global_variables_initializer())

        feed_dict = {
            self.var: np.ones((BATCH, TIME, IN_SHAPE)),
            self.label: np.ones((BATCH, TIME, IN_SHAPE))
        }
        self.Sess.run([self.opt], feed_dict)

    def __call__(self):
        for i in range(1000):
            print(i)
            self.init_model()
            tf.reset_default_graph()

if __name__ == '__main__':
    op = test_op()
    op()
