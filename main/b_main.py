import init_path
from config.base_config import *
from config.rnn_config import *

import runner.1b_vanilla
import runner.1b_sparse
import runner.ptb_drw
import runner.ptb_snip

def main():
    import logging
    import tensorflow

    parser = get_base_parser()
    parser = rnn_parser(parser)
    params = make_parser(parser)
    if params.exp_id == 'sparse':
        Runner = runner.ptb_sparse.PTBRunner('ptb', params)
    elif params.exp_id == 'vanilla':
        Runner = runner.ptb_vanilla.PTBRunner('ptb', params)
    elif params.exp_id == 'snip':
        Runner = runner.ptb_snip.PTBRunner('ptb', params)
    elif params.exp_id == 'drw':
        Runner = runner.ptb_drw.PTBRunner('ptb', params)
    Runner.run()

if __name__ == '__main__':
    main()