import tensorflow as tf
import numpy as np

def uniform_initializer(size, npr, scale):
    return npr.uniform(size=size, low=-scale, high=scale)

def he_initializer(size, npr):
    pass

def xavier_initializer(size, npr):
    pass

def get_init(type):
    if type == 'uniform':
        return uniform_initializer

    elif type == 'he':
        return he_initializer

    elif type == 'xavier':
        return xavier_initializer

    else:
        pass