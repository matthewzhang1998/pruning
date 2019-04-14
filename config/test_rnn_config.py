import argparse

def rnn_parser(parser):
    parser.add_argument('--sparse_size', type=int, default=3000)

    return parser