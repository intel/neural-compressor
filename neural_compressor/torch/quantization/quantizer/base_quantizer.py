from abc import ABC, abstractmethod

QUANTIZER_REGISTRY = {}

def quantizer_register(name):
    def decorator(quantizer):
        QUANTIZER_REGISTRY[name] = quantizer
        return quantizer

    return decorator

def init_quantizer(model, quant_config):
    # according to model device and environment
    if model.device == "hpu":
        return QUANTIZER_REGISTRY["guadi"](quant_config)

class BaseQuantizer(ABC):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    @abstractmethod
    def prepare(self, model):
        raise NotImplementedError

    @abstractmethod
    def prepare(self, model):
        raise NotImplementedError
