
@quantizer_register(name="guadi")
class GuadiQuantizer(BaseQuantizer):
    def __init__(self, quant_cfg):
        super().__init__()

    def prepare(self, model):
        # refer to habana_quantization_toolkit/_hook_method/measure.py prepare_model
        pass

    def convert(self, model):
        # convert
        pass
