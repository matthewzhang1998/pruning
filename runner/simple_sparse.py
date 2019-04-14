import numpy as np
import tensorflow as tf

class SimpleSparse(object):
    def __init__(self, params):
        self.params = params

        values = np.zeros((params.sparse_size**2))
        indices = np.array(np.unravel_index(np.arange(params.sparse_size**2),
            (params.sparse_size, params.sparse_size))).T

        self.input = tf.placeholder(shape=[None, params.sparse_size], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, params.sparse_size], dtype=tf.float32)
        with tf.device("/cpu:0"):
            self.sparse_values = tf.Variable(values, dtype=tf.float32, trainable=True)
            self.sparse_tensor = tf.SparseTensor(indices, self.sparse_values,
                [params.sparse_size, params.sparse_size])

        input_transpose = tf.transpose(self.input)

        logits_transpose = tf.sparse_tensor_dense_matmul(self.sparse_tensor, input_transpose)

        logits = tf.transpose(logits_transpose)

        loss = tf.reduce_mean(tf.losses.mean_squared_error(self.labels, logits))

        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        batch_size = 1
        while True:
            inputs = np.random.normal(size=(batch_size, self.params.sparse_size))
            labels = np.random.normal(size=(batch_size, self.params.sparse_size))
            feed_dict = {
                self.input: inputs, self.labels: labels
            }

            self.sess.run(self.train_op, feed_dict)

            batch_size += 1

class WhileSparse(object):
    pass

if __name__ == '__main__':
    import init_path
    from config.base_config import *
    from config.test_rnn_config import *

    parser = get_base_parser()
    parser = rnn_parser(parser)
    params = make_parser(parser)
    net = SimpleSparse(params)
    net.train()
