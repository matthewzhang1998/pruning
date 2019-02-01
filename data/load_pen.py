# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import numpy as np

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
    '''
    :param filename: Path to data
    :return: list of tokens
    '''
    with tf.gfile.GFile(filename, "r") as f:
        # return f.read().decode("utf-8").split('\n').split()
        return f.read().split()

def _build_vocab(filename):
    '''
    :param filename: Path to data (typically train)
    :return: Dictionary which converts word tokens to index. Build based on word frequency
    '''
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    '''
    :param filename: Path to data
    :param word_to_id: Dictionary which converts word tokens to index.
    :return: List of token index
    '''
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/ptb.tgz
    Args:
      data_path: string path to the directory where ptb.tgz has
        been extracted.
    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      vocabulary:  List which converts index to word token
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)  # build vocabulary use train dataset
    # word_to_id is a dictionary with [word] as the key and [index/integer] as the value
    train_data = np.array(_file_to_word_ids(train_path, word_to_id))
    valid_data = np.array(_file_to_word_ids(valid_path, word_to_id))
    test_data = np.array(_file_to_word_ids(test_path, word_to_id))

    vocabulary = len(word_to_id)    # integer to token
    return {'train': train_data,
            'val': valid_data,
            'test': test_data,
            '_vocab': vocabulary}


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.
    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).
    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.
    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    data_len = raw_data.size
    batch_len = data_len // batch_size
    data = np.reshape(raw_data[0: batch_size * batch_len], [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps

    ret = []
    for i in range(epoch_size):
        x = data[:, i * num_steps:(i+1)*num_steps]
        y = data[:, i * num_steps+1:(i+1)*num_steps+1]
        ret.append((x,y))
    return ret

class Dataset(object):
    """The input data."""
    def __init__(self, params, load_path):
        self.batch_size = params.batch_size
        self.num_steps = params.max_length
        self.data = ptb_raw_data(load_path)
        self.vocab_size = 10000 #self.data['_vocab']
        self.i = {key: 0 for key in self.data}

    def get_batch(self, scope='train'):
        return ptb_producer(self.data[scope], self.batch_size, self.num_steps)

if __name__ == '__main__':
    data = ptb_raw_data("ptb/data")
    print(data['val'].size/20/40)