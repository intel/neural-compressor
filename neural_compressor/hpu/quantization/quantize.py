from neural_compressor.torch.utils import is_hpex_available


def prepare(model, quant_cfg):
    # if is_hpex_available():
    if True:
        from .quantizer import HQTQuantizer
        quantizer = HQTQuantizer()
        quantizer.prepare(model, quant_cfg)
    else:
        # native PT or other quantizer prepare
        pass

def convert(model):
    if is_hpex_available():
        from .quantizer import HQTQuantizer
        quantizer = HQTQuantizer()
        quantizer.convert(model)
    else:
        # native PT or other quantizer convert
        pass

def save_calib(model):
    # act like habana_quantization_toolkit.finish_measurements(model)
    pass
