from abc import ABC, abstractmethod

class BaseQuantizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def prepare(self, model, quant_cfg):
        raise NotImplementedError

    @abstractmethod
    def prepare(self, model):
        raise NotImplementedError

class HQTQuantizer(BaseQuantizer):
    def __init__(self):
        super().__init__()

    def config_mapping(self, model, quant_cfg):
        model_info = quant_cfg.get_model_info(model=model)
        configs_mapping = quant_cfg.to_config_mapping(model_info=model_info)
        print(configs_mapping)
        return configs_mapping

    def prepare(self, model, quant_cfg):
        qconfig_mapping = self.config_mapping(model, quant_cfg)
        from neural_compressor.torch.algorithms.habana_fp8.fp8_quant import prepare
        prepared_model = prepare(model, qconfig_mapping)
        return prepared_model
        # pass

    def convert(self, model):
        from neural_compressor.torch.algorithms.habana_fp8.fp8_quant import convert
        q_model = convert(model)
        return q_model
