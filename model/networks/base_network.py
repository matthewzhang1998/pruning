import tensorflow as tf

class BaseModel(object):
    def __init__(self, params):
        self.params = params
        self.base_scope = params.model_type

        self.Tensor = {}
        self.Network = {}
        self.Mask = {}

    def set_weighted_params(self, weighted_dict, scope):
        raise NotImplementedError

    def weighted(self, input, scope):
        raise NotImplementedError

    def weight_variables(self):
        raise NotImplementedError