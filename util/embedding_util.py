import tensorflow as tf

class Embedding(object):
    def __init__(self, embed_arr, sess):
        dim_1, dim_2 = embed_arr.shape

        self.W = tf.Variable(tf.constant(0.0, shape=[dim_1, dim_2]))

        self.init_ph = tf.placeholder(tf.float32, [dim_1, dim_2])
        self.init_op = self.W.assign(self.init_ph)

    def __call__(self, input_str):
        return tf.nn.embedding_lookup(self.W, input_str)

