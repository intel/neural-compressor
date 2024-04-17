from .._quant_common.quant_config import Fp8cfg, ToolMethod
from .._module_method.quant_util import quant_model

config = Fp8cfg().cfg


def quantize_model(model):
    if config["method"] == ToolMethod.HOOKS:
        print("WARNING: Quantization method HOOKS is deprecated in function quantize_model(). " \
              "Please use prepare_model.prep_model() instead.")
    elif config["method"] == ToolMethod.MODULES:
        quant_model(model)
