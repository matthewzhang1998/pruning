import argparse

def get_base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--decay_scheme', type=str, default='exponential')
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--start_epoch', type=int, default=3)

    parser.add_argument('--max_grad', type=float, default=5)

    parser.add_argument('--decay_iter', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_steps', type=int, default=13)
    parser.add_argument('--val_steps', type=int, default=1)

    parser.add_argument('--seed', type=int, default=12)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # parser.add_argument('--noise_delta', type=float, default=0.1)
    # parser.add_argument('--snip_k', type=float, default=0.99)
    # parser.add_argument('--l2_k', type=float, default=0.99)
    # parser.add_argument('--random_k', type=float, default=0.99)
    parser.add_argument('--prune_k', type=float, default=0.999)
    parser.add_argument('--block_k', type=float, default=0.98)

    parser.add_argument('--log_dir', type=str, default='../log/log')
    parser.add_argument('--model_type', type=str, default='rnn')

    parser.add_argument('--grad_param', type=str, default='Mask') # Weight, Mask, Comb

    parser.add_argument('--prune_method', type=str, default='separate')
    parser.add_argument('--value_method', type=str, default='largest')

    # parser.add_argument('--embed_sparsity', type=float, default=0.95)
    # parser.add_argument('--softmax_sparsity', type=float, default=0.95)

    parser.add_argument('--mlp_sparsity', type=float, default=0.95)

    # parser.add_argument('--pretrain_learning_rate', type=float, default=1e-3)
    # parser.add_argument('--pretrain_num_steps', type=int, default=10)
    # parser.add_argument('--pretrain_weight_decay', type=float, default=0.00)
    # parser.add_argument('--pretrain_kl_beta', type=float, default=0.0)

    # parser.add_argument('--min_length', type=int, default=50)
    parser.add_argument('--max_length', type=int, default=20)

    parser.add_argument('--embed_size', type=int, default=600)

    parser.add_argument('--num_unitwise_rnn', type=int, default=128)
    # parser.add_argument('--num_unitwise_mlp', type=int, default=16)

    parser.add_argument('--l1_mask_penalty', type=float, default=0.00)
    parser.add_argument('--val_size', type=int, default=20)

    parser.add_argument('--drw_k', type=int, default=0.99)
    parser.add_argument('--drw_temperature', type=int, default=0.00)

    parser.add_argument('--weight_dir', type=str, default=None)
    parser.add_argument('--exp_id', type=str, default='ptb6')


    return parser

def make_parser(parser):
    return post_process(parser.parse_args())

def post_process(args):
    # parse the network shape
    for key in dir(args):
        if 'seq' in key:
            if getattr(args, key) is None:
                setattr(args, key, [])
            elif 'hidden' in key:
                setattr(args, key, [int(dim) for dim in getattr(args, key).split(',')])
            else:
                setattr(args, key, [str(dim) for dim in getattr(args, key).split(',')])
    return args


