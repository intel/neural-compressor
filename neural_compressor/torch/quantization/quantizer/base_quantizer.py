from abc import ABC, abstractmethod

QUANTIZER_REGISTRY = {}
def quantizer_register(cls):
    ...

def init_quantizer(model, quant_cfg):
    # according to model device and environment
    if model.device == "hpu":
        return QUANTIZER_REGISTRY["guadi"](quant_cfg)

class BaseQuantizer(ABC):
    def __init__(self, quant_cfg):
        self.quant_cfg = quant_cfg

    @abstractmethod
    def prepare(self, model, quant_cfg):
        raise NotImplementedError

    @abstractmethod
    def prepare(self, model):
        raise NotImplementedError
