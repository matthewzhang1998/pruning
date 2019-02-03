import init_path
from config.base_config import *
from config.rnn_config import *
from runner.rnn_mnist import *

def main():
    import logging
    import tensorflow

    tf.logging.set_verbosity(tf.logging.INFO)
    parser = get_base_parser()
    parser = rnn_parser(parser)
    params = make_parser(parser)
    Runner = MNISTRunner('mnist', params)
    Runner.run()

if __name__ == '__main__':
    main()