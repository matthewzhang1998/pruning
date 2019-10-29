import argparse

def mlp_parser(parser):
    parser.add_argument('--mlp_hidden_seq', type=str, default='256,256,256')
    parser.add_argument('--mlp_act_seq', type=str, default='sigmoid,sigmoid,sigmoid')
    parser.add_argument('--mlp_norm_seq', type=str, default='none,none,none')

    parser.add_argument('--mlp_init_type', type=str, default='xavier')
    parser.add_argument('--mlp_init_scale', type=float, default=.1)

    return parser