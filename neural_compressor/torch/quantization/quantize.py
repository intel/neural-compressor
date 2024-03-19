from neural_compressor.torch.utils import is_hpex_available
from neural_compressor.common.base_config import BaseConfig
from .quantizer.quantizer import init_quantizer
import torch

# another proposal:
# global quant_cfg

def prepare(model: torch.nn.Module, quant_cfg: BaseConfig):
    """Prepare the model for quantization.

    This inserts observers in the model that will observe activation tensors during calibration.

    Args:
        model (torch.nn.Module): origin model
        quant_cfg (BaseConfig): quantization config, including observer method

    Returns:
        model with observers
    """
    if need_calibration():
        quantizer = init_quantizer(model, quant_cfg)
        prepared_model = quantizer.prepare(model)
        return prepared_model
    else:
        return model


def convert(model: torch.nn.Module, quant_cfg: BaseConfig):
    """Convert the origin model to a quantized model.

    Args:
        model (torch.nn.Module): origin model
        quant_cfg (BaseConfig): quantization config, including scale method
    """
    quantizer = init_quantizer(model)
    q_model = quantizer.convert(model, quant_cfg)
    return q_model


def save_calib(model: torch.nn.Module, quant_cfg: BaseConfig):
    """Save calibration result to local file.

    Args:
        model (torch.nn.Module): model with observers (output model of prepare() func)
        quant_cfg (BaseConfig): including save path of calibration results
    """
    if need_calibration():
        # refer to habana_quantization_toolkit.finish_measurements(model)
        _save_calibration_results(model, quant_cfg)
    else:
        return
