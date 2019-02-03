import init_path
import numpy as np
import os
import os.path as osp

from config.base_config import *
from config.rnn_config import *

def weights():
    parser = get_base_parser()
    parser = rnn_parser(parser)
    params = make_parser(parser)

    dir = '../weights/rnn'
    if not osp.exists(dir):
        os.makedirs(dir)

    npr = np.random.RandomState(params.seed)
    embed_weights = npr.uniform(-.1, .1, [10000, params.embed_size])
    np.save(osp.join(dir, '0.npy'), embed_weights)

    ni = params.embed_size
    for ix in range(len(params.rnn_r_hidden_seq)):
        nh = params.rnn_r_hidden_seq[ix]
        if nh < 2000:
            w = npr.normal(0, np.sqrt(2/ni), [ni+nh, 4*nh])
            np.save(osp.join(dir, '{}.npy'.format(ix+1)), w)
        ni = nh

    nl = 1+len(params.rnn_r_hidden_seq)
    w = npr.normal(0, np.sqrt(2/(ni+params.embed_size)), [ni, params.embed_size])
    np.save(osp.join(dir, '{}.npy'.format(nl)), w)

    w = npr.uniform(-.1, .1, [params.embed_size, 10000])
    np.save(osp.join(dir, '{}.npy'.format(nl+1)), w)


if __name__ == '__main__':
    weights()