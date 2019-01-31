import argparse

def mlp_parser(parser):
    parser.add_argument('--mlp_l_hidden_seq', type=str, default='300,100')
    parser.add_argument('--mlp_l_act_seq', type=str, default='relu,relu')
    parser.add_argument('--mlp_l_norm_seq', type=str, default='none,none')

    return parser