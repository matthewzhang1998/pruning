#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:09:41 2018

@author: matthewszhang
"""
import numpy as np
import tensorflow as tf

# prevent nans
EPS = 1e-8

def gaussian_noise_layer(input_layer, std=0.1):
    noise = tf.random.normal(shape=tf.shape(input_layer),
        mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer+noise


def gauss_log_prob(mu, logstd, labels):
    # calculate the negative of log probability
    neg_log_p_n = \
        0.5 * tf.reduce_sum(tf.square((labels - mu) / \
        (tf.exp(logstd) + EPS)), axis=-1) \
        + tf.reduce_sum(logstd, axis=-1)

    # invert and return
    return -neg_log_p_n

def gauss_kl(mu, logstd, original_mu=0.0, original_logstd=0.0):
    return tf.reduce_sum(
        original_logstd - logstd + \
        (tf.square(tf.exp(logstd)) + tf.square(mu - original_mu))/ \
        (2 * tf.square(tf.exp(original_logstd))) - 0.5
    )

def categorical_log_prob(logits, labels):
    # assume labels are not as one_hot
    one_hot_labels = tf.one_hot(labels, logits.get_shape().as_list()[-1])
    return -tf.nn.softmax_cross_entropy_with_logits(
            logits=logits + EPS,
            labels=one_hot_labels)


class Base_Distribution(object):
    def __init__(self, *args):
        self.sample = None
        self.entropy = None


class Gaussian(object):
    def __init__(self, mu, logsigma):
        self._means = mu
        self._logstds = logsigma
        self._stds = tf.exp(logsigma)

        __uniform = tf.random_normal(tf.shape(self._means))

        self.sample = self._means + self._stds * __uniform
        self.entropy = tf.reduce_sum(
            self._logstds + .5 * np.log(2.0 * np.pi * np.e), axis=-1
        )


class Categorical(object):
    def __init__(self, logits):
        self._logits = logits

        __uniform = tf.random_uniform(tf.shape(self._logits))
        self.sample = tf.argmax(
            self._logits - tf.log(-tf.log(__uniform + EPS) + EPS), axis=-1
        )

        delta_logits = self._logits - \
            tf.reduce_max(self._logits, axis=-1, keep_dims=True)
        exp_delta_logits = tf.exp(delta_logits)
        partition = \
            tf.reduce_sum(exp_delta_logits, axis=-1, keep_dims=True)
        norm_logits = exp_delta_logits / partition

        self.entropy = tf.reduce_sum(norm_logits * \
            (tf.log(partition + EPS) - delta_logits), axis=-1
        )

class Binary_Bernoulli(object):
    def __init__(self, prob):
        self.distrib = tf.contrib.distributions.Bernoulli([prob])

    def __call__(self, weight_like):
        sample = tf.cast(self.distrib.sample(tf.shape(weight_like)), tf.float32)[..., 0]
        binary = tf.round(weight_like + sample)
        return binary
