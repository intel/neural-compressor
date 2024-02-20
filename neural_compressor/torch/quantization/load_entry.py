import os
import json
from neural_compressor.common.utils import FP8_QUANT, GPTQ, RTN  # unified namespace


def load(model, output_dir="./saved_results"):
    qmodel_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "quantized_model.pt")
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "qconfig.json")
    with open(qconfig_file_path, "r") as f:
        model_qconfig = json.load(f)
    if model_qconfig['algorithm'] == FP8_QUANT:
        from neural_compressor.torch.algorithms.habana_fp8 import load
        return load(model, output_dir)
    