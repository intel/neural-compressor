from .._quant_common.quant_config import Fp8cfg, ToolMethod, QuantMode
from .._module_method.prepare import init_model
from .._hook_method import prepare_model
from .._hook_method.measure import save_measurements
from .._module_method.tensor_logger import TensorLogger
import habana_frameworks.torch.core as htcore

config = Fp8cfg().cfg


def prep_model(model):
    htcore.hpu_initialize()
    if config["method"] == ToolMethod.HOOKS:
        prepare_model(model)  # registers hooks
    elif config["method"] == ToolMethod.MODULES:
        init_model(model)  # monkey patch modules
        if config["mode"] == QuantMode.MEASURE:
            model.tl = TensorLogger(module=model)

def finish_measurements(model):
    if config["method"] == ToolMethod.HOOKS:
        save_measurements(model)
        print("Dumping measurements")
    elif config["method"] == ToolMethod.MODULES:
        model.tl.finish()
        print("Dumping measurements")
