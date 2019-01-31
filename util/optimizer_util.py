import tensorflow as tf

def get_optimizer(params, learning_rate):
    opt_type = params.optimizer_type.lower()
    if opt_type == 'adam':
        return tf.train.AdamOptimizer(learning_rate)

    elif opt_type == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate)

    elif opt_type == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate)

    elif opt_type == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate)

    elif opt_type == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate, params.momentum)

    else:
        raise ValueError("String not found.")