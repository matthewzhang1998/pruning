import tensorflow as tf
import numpy as np

def unison_shuffle(a,b, npr):
    p = npr.permutation(len(a))
    return a[p], b[p]

class SeqMNISTDataset(object):
    def __init__(self, params):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train = np.concatenate([x_train, x_test[:3000]], axis=0)
        y_train = np.concatenate([y_train, y_test[:3000]], axis=0)
        x_test = x_test[3000:]
        y_test = y_test[3000:]

        x_train /= 10*255
        x_test /= 10*255

        x_train = x_train.reshape([-1, 28, 28])
        x_test = x_test.reshape([-1, 28, 28])

        self.npr = np.random.RandomState(params.seed)
        x_train, y_train = unison_shuffle(x_train, y_train, self.npr)

        num_classes = 10
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        self.batch_size = params.batch_size
        self.seed = params.seed
        self.data = {
            'train': (x_train, y_train),
            'val': (x_test, y_test),
            'test': None
        }

    def get_batch(self, scope='train'):
        return mnist_iterator(self.data[scope], self.batch_size, self.seed)

def mnist_iterator(raw_data, batch_size, seed, name=None):
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
    Data = (raw_data[0], raw_data[1])

    start_ix = 0
    while True:
        end_ix = start_ix + batch_size

        if end_ix > len(Data[0]):
            end_ix = end_ix - len(Data[0])

            features = np.concatenate(
                [Data[0][start_ix:],
                 Data[0][:end_ix]],
                axis=0
            )
            labels = np.concatenate(
                [Data[1][start_ix:],
                 Data[1][:end_ix]],
                axis=0
            )
        else:
            features = Data[0][start_ix:end_ix]
            labels = Data[1][start_ix:end_ix]

        start_ix = end_ix

        yield features, labels