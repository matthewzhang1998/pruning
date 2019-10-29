import argparse

def get_base_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--decay_scheme', type=str, default='exponential')
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--start_epoch', type=int, default=3)

    parser.add_argument('--max_grad', type=float, default=1)

    parser.add_argument('--decay_iter', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=13)
    parser.add_argument('--val_steps', type=int, default=1000)

    parser.add_argument('--val_iter', type=int, default=400)
    parser.add_argument('--log_steps', type=int, default=1000)

    parser.add_argument('--seed', type=int, default=353)

    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)

    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # parser.add_argument('--noise_delta', type=float, default=0.1)
    # parser.add_argument('--snip_k', type=float, default=0.99)
    # parser.add_argument('--l2_k', type=float, default=0.99)
    # parser.add_argument('--random_k', type=float, default=0.99)
    parser.add_argument('--prune_k', type=float, default=0.98)
    parser.add_argument('--block_k', type=float, default=0.01)

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

    parser.add_argument('--embed_size', type=int, default=400)

    parser.add_argument('--num_unitwise_rnn', type=int, default=128)
    # parser.add_argument('--num_unitwise_mlp', type=int, default=16)

    parser.add_argument('--l1_mask_penalty', type=float, default=0.00)
    parser.add_argument('--val_size', type=int, default=20)

    parser.add_argument('--drw_k', type=int, default=0.99)
    parser.add_argument('--drw_temperature', type=int, default=0.00)

    parser.add_argument('--weight_dir', type=str, default=None)
    parser.add_argument('--exp_id', type=str, default='sparse')

    parser.add_argument('--vocab_size', type=int, default=100000)
    parser.add_argument('--use_sample_softmax', type=int, default=0)
    parser.add_argument('--num_sample', type=int, default=1000)

    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--eval_iter', type=int, default=1)
    parser.add_argument('--test_iter', type=int, default=10)
    parser.add_argument('--num_generate', type=int, default=6)
    parser.add_argument('--num_iterate', type=int, default=1000)
    parser.add_argument('--rand_eps', type=int, default=0.2)
    parser.add_argument('--weight_eps', type=int, default=0.2)

    parser.add_argument('--evolution_lr', type=float, default=0.001)
    parser.add_argument('--meta_opt_method', type=str, default='convex')

    parser.add_argument('--log_memory', type=int, default=0)

    parser.add_argument('--noise_type', type=str, default='replace')

    parser.add_argument('--prune_criteria', type=str, default='jacobian_easy')

    parser.add_argument('--jacobian_horizon', type=int, default=4)
    parser.add_argument('--horizon_trace', type=float, default=0.9)
    parser.add_argument('--deterministic', type=int, default=0)

    parser.add_argument('--uniform_by_gate', type=int, default=0)
    parser.add_argument('--uniform_by_input', type=int, default=0)

    parser.add_argument('--dummy_objective', type=str, default='grad')
    parser.add_argument('--dummy_batch', type=int, default=100)
    parser.add_argument('--unroll_dummy', type=int, default=5)

    parser.add_argument('--dataset', type=str, default='seq_mnist')
    parser.add_argument('--dummy_noise', type=int, default=.1)

    parser.add_argument('--get_jacobian', type=int, default=0)
    parser.add_argument('--plot_jacobian_iter', type=int, default=1000)

    parser.add_argument('--prune_iter', type=int, default=5000)
    parser.add_argument('--prune_iter_k_seq', type=str, default='0.8,0.6,0.4,0.2,0.1,0.05,0.02,0.01')

    parser.add_argument('--train_embed', type=int, default=0)
    parser.add_argument('--plot_jacobian_pre', type=int, default=0)

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
            elif 'k' in key:
                setattr(args, key, [float(dim) for dim in getattr(args, key).split(',')])
            else:
                setattr(args, key, [str(dim) for dim in getattr(args, key).split(',')])
    return args


