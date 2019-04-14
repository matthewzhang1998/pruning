import init_path
from config.base_config import *
from config.rnn_config import *

import runner.b_vanilla
import runner.b_sparse
import runner.b_regress


def main():
    import logging
    import tensorflow

    parser = get_base_parser()
    parser = rnn_parser(parser)
    params = make_parser(parser)

    if params.exp_id == 'vanilla':
        Runner = runner.b_vanilla.VanillaRunner('1b/vanilla', params)
    elif params.exp_id == 'sparse':
        Runner = runner.b_sparse.SparseRunner('1b/sparse', params)
    elif params.exp_id == 'regress':
        Runner = runner.b_regress.RegressionRunner("1b/regress", params)
        
    Runner.run()

if __name__ == '__main__':
    main()