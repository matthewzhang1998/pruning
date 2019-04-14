import tensorflow as tf
import numpy as np
import os
ABS_PATH = os.path.dirname(os.path.realpath(__file__))

print(ABS_PATH)

# enter from main
sparse_grad_module = tf.load_op_library(
    '/h/mzhang/tensorflow_eff_grad/tensorflow/tensorflow/core/user_ops/sparse_tensor_dense_matmul_sorted_grad_op.so')

sparse_grad = sparse_grad_module.sparse_tensor_dense_mat_mul_sorted_grad

from tensorflow.python.ops.sparse_ops import _convert_to_sparse_tensor
from tensorflow.python.ops.gen_sparse_ops import sparse_tensor_dense_mat_mul

@tf.custom_gradient
def efficient_grad_sparse_matmul(sp_ix, sp_v, sp_sh, input):
    with tf.name_scope(None, "SparseTensorDenseMatMul",
                        [sp_ix, sp_v, input]) as name:
        res = sparse_tensor_dense_mat_mul(
            a_indices=sp_ix,
            a_values=sp_v,
            a_shape=sp_sh,
            b=input,
            adjoint_a=True)

    def grad(dy):
        print(dy)
        a_type = sp_v.dtype.base_dtype
        b_type = input.dtype.base_dtype
        if a_type != b_type:
            raise TypeError("SparseTensorDenseMatMul op received operands with "
                            "different types: ", a_type, " and ", b_type)

        # gradient w.r.t. dense
        b_grad = sparse_tensor_dense_mat_mul(
            sp_ix, sp_v, sp_sh, dy, adjoint_a=False)

        sp_grad = sparse_grad(
            a_indices = sp_ix,
            a_shape = sp_sh,
            b = input,
            grad = dy)[:,0]

        return (None, sp_grad, None, b_grad)

    # gradients w.r.t. (a_indices, a_values, a_shape, b)
    return res, grad

class test_op(object):
    def __init__(self):
        init = np.random.normal(size=(100,200))
        self.tf_var = tf.Variable(init.flatten, dtype=tf.float32, trainable=True)
        self.dense_var = tf.Variable(init, dtype=tf.float32, trainable=True)
        indices = np.unravel_index(np.arange(20000), (100,200))
        self.tf_ix = tf.constant(np.array(indices).T, dtype=tf.int64)
        self.tf_sh = tf.constant([100,200], dtype=tf.int64)

        self.b = tf.constant(np.ones((300,100)), dtype=tf.float32)
        label = tf.constant(np.ones((300,200)), dtype=tf.float32)
        out1 = tf.transpose(efficient_grad_sparse_matmul(
            self.tf_ix, self.tf_var, self.tf_sh, tf.transpose(self.b)))
        out2 = tf.einsum('ij,jk->ik', self.b, self.dense_var)

        loss1 = tf.reduce_sum(tf.square(out1 - label))
        loss2 = tf.reduce_sum(tf.square(out2 - label))

        self.grad1 = tf.gradients(loss1, self.tf_var)
        self.grad2 = tf.gradients(loss2, self.dense_var)

    def __call__(self):
        with tf.device("/device:GPU:0"):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
                sess.run(tf.global_variables_initializer())
                grad1, grad2 = sess.run([self.grad1, self.grad2])
                print(grad1, grad2)
                grad1 = grad1[0]
                grad2 = grad2[0]
                grad1 = grad1.reshape(grad2.shape)

                print(grad1 - grad2, np.sum(np.square(grad1-grad2)))

if __name__ == "__main__":
    test = test_op()
    test()