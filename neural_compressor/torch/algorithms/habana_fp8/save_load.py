import torch
import json
import os
from neural_compressor.torch.utils import logger
import habana_frameworks.torch.core as htcore
from .modules import (
    # fp32
    Autocast,
    BatchMatmul,
    Matmul,
    # dynamic modules
    FP8DynamicBatchMatmul,
    FP8DynamicLinear,
    FP8DynamicMatmul,
    FP8Cast,
)
from .fp8_quant import FP8_DTYPE, dtype_mapping


def save(model, output_dir="./saved_results"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "quantized_model.pt")
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "qconfig.json")
    # saving process
    with open(qconfig_file_path, "w") as f:
        json.dump(model.qconfig, f, indent=4)

    import fp8_convert
    stat_dict = {}
    for k, v in model.state_dict().items():
        if v.dtype in FP8_DTYPE:
            v = fp8_convert.to_u8(v.to('cpu'))
        stat_dict[k] = v.to('cpu')
    torch.save(stat_dict, qmodel_file_path)

    logger.info("Save state_dict of quantized model to {}.".format(qmodel_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))


def load(model, output_dir="./saved_results"):
    from neural_compressor.torch.utils import fetch_module, set_module
    from .fp8_quant import white_list, quantization_mapping
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "quantized_model.pt")
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "qconfig.json")
    with open(qconfig_file_path, "r") as f:
        model_qconfig = json.load(f)
    # load quantization configuration
    stat_dict = torch.load(qmodel_file_path)
    import fp8_convert
    for op_name, op_qconfig in model_qconfig["per_module_qconfig"].items():
        dtype = op_qconfig['w_dtype']
        choice = 1 if dtype=="fp8_e4m3" else 0
        if op_name+".weight" in stat_dict:
            stat_dict[op_name+".weight"] = fp8_convert.from_u8(stat_dict[op_name+".weight"], choice)
        if dtype not in FP8_DTYPE:
            continue
        module = fetch_module(model, op_name)
        dtype = dtype_mapping[dtype]
        if op_qconfig['approach'] == "static":
            if isinstance(module, white_list):
                QModule = quantization_mapping[type(module)]
                module = QModule(module, dtype)
        else:
            if isinstance(module, torch.nn.Linear):
                # need module for initialization
                module = FP8DynamicLinear(module, dtype)
            elif isinstance(module, Matmul):
                module = FP8DynamicMatmul(dtype)
            elif isinstance(module, BatchMatmul):
                module = FP8DynamicBatchMatmul(dtype)
            elif isinstance(module, Autocast):
                module = FP8Cast(dtype=dtype)
        set_module(model, op_name, module)
        htcore.mark_step()
    model.load_state_dict(stat_dict)
    htcore.mark_step()
    logger.info("Quantized model loading successful.")
    return model
