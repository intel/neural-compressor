from abc import ABC, abstractmethod

BACKEND_REGISTRY = {}

def backend_register(name):
    def decorator(backend):
        BACKEND_REGISTRY[name] = backend
        return backend
    return decorator

def init_backend(model):
    # according to model device and environment
    if model.device == "hpu":
        return BACKEND_REGISTRY["hqt"](model.qconfig)

class BaseBackend(ABC):
    def __init__(self, quant_config):
        self.quant_config = quant_config

    @abstractmethod
    def prepare(self, model):
        raise NotImplementedError

    @abstractmethod
    def prepare(self, model):
        raise NotImplementedError
