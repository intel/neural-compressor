from collections import OrderedDict

class Parameter():
    r"""Set the tunable parameters of a given compression algorithm.

    Args:
        name: a unique string to present the tunable parameter
        values: the possible values of this parameter

    """
    def __init__(self, values):
        self.values = values

class Compression():
    r"""The base class of all supported compression algorithms.

    """
    def __init__(self):
        self._parameters = OrderedDict()

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.register_parameter(name, value)

    def register_parameter(self, name, value):
        self._parameters[name] = param

class Quant(Compression):
    def __init__(self, bits, white_list, black_list=None):
        super().__init__(self)
        self.granularity = Parameter(['per_channel', 'per_tensor'])
        self.scheme = Parameter(['sym', 'asym'])

class Quantizer(Compression):
    def __init__(self, weight_quantizer, activation_quantizer, white_list):
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer


class WeightQuantizer(Compression):
    def __init__(self, dtype, reduce_range, scheme, granularity):
        pass

ActivationQuantizer = WeightQuantizer(dtype, reduce_range, scheme, granualarity='per_tensor')


