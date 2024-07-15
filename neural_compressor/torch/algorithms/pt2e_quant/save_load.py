import torch
import os
import json
from neural_compressor.torch.utils import QCONFIG_NAME, WEIGHT_NAME, logger
from neural_compressor.common.utils import load_config_mapping, save_config_mapping

def save(model, example_inputs, output_dir="./saved_results"):
    os.makedirs(output_dir, exist_ok=True)
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), QCONFIG_NAME)
    quantized_ep = torch.export.export(model, example_inputs)
    torch.export.save(quantized_ep, qmodel_file_path)
    for key, op_config in model.qconfig.items():
        model.qconfig[key] = op_config.to_dict()
    with open(qconfig_file_path, "w") as f:
        json.dump(model.qconfig, f, indent=4)
    
    logger.info("Save quantized model to {}.".format(qmodel_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))

def load(output_dir="./saved_results"):
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    loaded_quantized_ep = torch.export.load(qmodel_file_path)
    return loaded_quantized_ep.module()