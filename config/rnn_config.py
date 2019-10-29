import argparse

def rnn_parser(parser):
    parser.add_argument('--rnn_cell_type', type=str, default='gru')

    parser.add_argument('--rnn_r_hidden_seq', type=str, default='400')
    parser.add_argument('--rnn_r_act_seq', type=str, default='none,none,none,none,none')
    parser.add_argument('--rnn_r_norm_seq', type=str, default='none,none,none,none,none')
    parser.add_argument('--rnn_l_hidden_seq', type=str, default=None)
    parser.add_argument('--rnn_l_act_seq', type=str, default='none')
    parser.add_argument('--rnn_l_norm_seq', type=str, default='none')

    parser.add_argument('--rnn_bidirectional', type=int, default=0)
    parser.add_argument('--rnn_dilated', type=int, default=0)

    parser.add_argument('--rnn_init_scale', type=float, default=0.1)
    parser.add_argument('--rnn_init_type', type=str, default='normal')

    parser.add_argument('--rnn_pre_init_scale', type=float, default=.07)
    parser.add_argument('--rnn_pre_init_type', type=str, default='normal')

    parser.add_argument('--rnn_swap_memory', type=int, default=1)
    parser.add_argument('--rnn_use_dense', type=int, default=0)

    parser.add_argument('--rnn_prune_method', type=str, default='block_random') # unit, block, random
    parser.add_argument('--use_knet', type=int, default=0)

    parser.add_argument('--use_factor_softmax', type=int, default=0)
    parser.add_argument('--num_factor', type=int, default=20)

    parser.add_argument('--rnn_act_init_type', type=str, default='normal')
    parser.add_argument('--rnn_act_init_scale', type=float, default=0.01)
    parser.add_argument('--input_temperature', type=float, default=1)

    return parser