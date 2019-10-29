import tensorflow as tf
import numpy as np

def uniform_initializer(size, npr, scale, **kwargs):
    return npr.uniform(size=size, low=-scale, high=scale)

def xavier_initializer(shape, npr, scale, **kwargs):
    out = npr.normal(loc=0, scale=1/np.sqrt(.5*shape[0] + .5*shape[-1]), size=shape)
    return out

def normal_initializer(shape, npr, scale=0.1, **kwargs):
    return npr.normal(loc=0, scale=scale, size=shape)

def glorot_initializer(shape, npr, scale=0.1, arch='lstm'):
    # Xavier for LSTM
    if arch == 'lstm':
        out = npr.normal(loc=0, scale=1/np.sqrt(shape[0]), size=shape)

    elif arch == 'gru':
        out = npr.normal(loc=0, scale=1/np.sqrt(shape[0]), size=shape)

    return out

def camel_initializer(shape, npr, scale, **kwargs):
    out = npr.normal(loc=scale, scale=scale/2, size=shape)\
        * (2*npr.binomial(1, 0.5, shape) - 1)
    return out

def camel_unif_initializer(shape, npr, scale, **kwargs):
    out = npr.uniform(size=shape, low=scale/2, high=3*scale/2)\
        * (2*npr.binomial(1, 0.5, shape) - 1)
    return out

def bidelta_initializer(shape, npr, scale, **kwargs):
    out = (2*npr.binomial(1, 0.5, shape) - 1) * scale
    return out

def he_initializer(shape, npr, scale, arch='lstm', **kwargs):
    if arch == 'lstm':
        ni = shape[0] - shape[-1]//4
        out = npr.normal(loc=0, scale=2/np.sqrt(ni), size=shape)

    elif arch == 'gru':
        ni = shape[0] - shape[-1]//3
        out = npr.normal(loc=0, scale=2/np.sqrt(ni), size=shape)

    return out

def dynamic_initializer(shape, npr, scale, arch='lstm', **kwargs):
    if arch == 'lstm':
        nh = shape[1]//4
        ni = shape[0] - nh

        out = np.zeros(shape)

        out[:ni, :nh] = npr.normal(loc=0, scale=1, size=(ni,nh))
        out[ni:, :nh] = npr.normal(loc=0, scale=1e-5, size=(nh,nh))

        out[:ni, nh:2*nh] = npr.normal(loc=0, scale=1, size=(ni, nh))
        out[ni:, nh:2*nh] = npr.normal(loc=0, scale=1e-5, size=(nh, nh))

        out[:ni, 2*nh:3*nh] = npr.normal(loc=1, scale=0.0, size=(ni, nh))
        out[ni:, 2*nh:3*nh] = npr.normal(loc=1, scale=1e-5, size=(nh, nh))

        out[:ni, 3*nh:] = npr.normal(loc=0, scale=0, size=(ni, nh))
        out[ni:, 3*nh:] = npr.normal(loc=0, scale=1, size=(nh, nh))

        return out

    elif arch == 'gru':
        pass

def get_init(type):
    if type == 'uniform':
        return uniform_initializer

    elif type == 'he':
        return he_initializer

    elif type == 'xavier':
        return xavier_initializer

    elif type == 'camel':
        return camel_initializer

    elif type == 'camel_unif':
        return camel_unif_initializer

    elif type == 'normal':
        return normal_initializer

    elif type == 'bidelta':
        return bidelta_initializer

    elif type == 'dynamic':
        return dynamic_initializer

    elif type == 'glorot':
        return glorot_initializer

    elif type == 'he':
        return he_initializer

    else:
        pass