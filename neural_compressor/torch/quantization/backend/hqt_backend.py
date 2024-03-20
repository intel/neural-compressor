from neural_compressor.torch.quantization.backend import BaseBackend, backend_register

@backend_register(name="hqt")
class HQTBackend(BaseBackend):
    def __init__(self, quant_config):
        super().__init__(quant_config)

    def prepare(self, model):
        # fp8 refer to habana_quantization_toolkit/_hook_method/measure.py prepare_model
        pass

    def convert(self, model):
        # fp8 refer to habana_quantization_toolkit/_hook_method/quantize.py quantize_hooks
        pass
