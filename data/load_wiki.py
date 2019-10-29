import os
from io import open

import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = np.zeros((tokens))
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def get_batch(self, dset='train'):
        if dset == 'train':
            x = self.train()

def producer(raw_data, batch_size, num_steps, name=None):
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

    while True:
        for i in range(epoch_size):
            x = data[:, i * num_steps:(i+1)*num_steps]
            y = data[:, i * num_steps+1:(i+1)*num_steps+1]
            yield x, y

class WikiDataset(object):
    """The input data."""
    def __init__(self, params, load_path):
        self.corpus = Corpus(load_path)

        self.batch_size = params.batch_size
        self.num_steps = params.max_length
        self.data = {
            'train': self.corpus.train,
            'val': self.corpus.valid,
            'test': self.corpus.test,
        }
        self.vocab_size = len(self.corpus.dictionary)
        print(self.vocab_size)
        self.i = {key: 0 for key in self.data}

    def get_batch(self, scope='train'):
        return producer(self.data[scope], self.batch_size, self.num_steps)