"""
# Prerequisite
pip install -r requirements.txt

# Note for static/dynamic W8FP8 quantization:
1. Name convention:
    - weight scale name: "prefix.scale_weight"
    - input scale name: "prefix.scale_input"
2. A json file mapping from tensor name to safetensor file name.

Example:
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5, bias=False)

    def forward(self, inp):
        x1 = self.fc1(inp)
        return x1

1. state dict
{
    "fc1.weight": torch.Tensor(...),
    "fc1.scale_weight": torch.Tensor(...),
    "fc1.scale_input": torch.Tensor(...),
}

2. json file, model.safetensors.index.json
{
    "fc1.weight": "qmodel.safetensors",
    "fc1.scale_weight": "qmodel.safetensors",
    "fc1.scale_input": "qmodel.safetensors"
}

"""

import os
import torch
import tqdm
from loguru import logger
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import json

torch.set_grad_enabled(False)

# CONSTANTS
SAFETENSORS = "safetensors"
WEIGHT_SCALE_NAME = "scale_weight"
SCALE_FILE_NAME = f"scales.{SAFETENSORS}"
FULL_RANGE = torch.finfo(torch.float8_e4m3fn).max
WEIGHT_BACKOFF = 0.5
"""
# https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html?highlight=backoff#supported-json-config-file-options
Similarly, the maxabs value of a weight is scaled to weight_backoff*FP8_143_FULLSCALE. The default values are input_backoff=0.25 and weight_backoff=0.5.
"""
MODEL_STATE_DICT_MAPPING_FILENAME = "model.safetensors.index.json"


def get_all_weight_filename(model_path):
    all_files = os.listdir(model_path)
    all_weight_filename = []
    for file in all_files:
        if file.endswith(f".{SAFETENSORS}"):
            all_weight_filename.append(file)
    return all_weight_filename


# from _fp8_quant/_core/fp_utils.py
def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def quant_tensor(tensor):
    # Note:
    #  1. Check the scale dtype
    #  2. Check the scale shape
    amax = tensor.abs().max()
    scale = calc_maxabs_scale(amax, FULL_RANGE, WEIGHT_BACKOFF)
    qtensor = tensor / scale
    cliped_qtensor = torch.clamp(qtensor, -FULL_RANGE, FULL_RANGE)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return scale, cliped_qtensor_fp8


def _maybe_create_dir(qmodel_path):
    if not os.path.exists(qmodel_path):
        os.makedirs(qmodel_path)


def static_quant_model(model_path, qmodel_path):
    _maybe_create_dir(qmodel_path)
    all_weight_filename = get_all_weight_filename(model_path)
    logger.info(f"Got {len(all_weight_filename)} weight files")
    qtensor_mappping = {}
    for i, filename in tqdm.tqdm(enumerate(all_weight_filename)):
        file_path = os.path.join(model_path, filename)
        qmodel_file_name = filename
        qmodel_file_path = os.path.join(qmodel_path, qmodel_file_name)
        qtensors = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                weight = f.get_tensor(weight_name)
                scale, qtensor = quant_tensor(weight)
                preifx_name = weight_name[: -len(".weight")]
                scale_name = f"{preifx_name}.{WEIGHT_SCALE_NAME}"
                qtensors[scale_name] = scale
                qtensors[weight_name] = qtensor
                qtensor_mappping[scale_name] = qmodel_file_name
                qtensor_mappping[weight_name] = qmodel_file_name
        logger.debug(f"Saving {len(qtensors)} tensors to {qmodel_file_path}")
        save_file(qtensors, os.path.join(qmodel_path, qmodel_file_path))
    # Dump tensor mapping into json file
    model_state_dict_mapping_file_path = os.path.join(qmodel_path, MODEL_STATE_DICT_MAPPING_FILENAME)
    logger.info(f"Saving tensor mapping to {model_state_dict_mapping_file_path}")
    with open(model_state_dict_mapping_file_path, "w") as f:
        json.dump(qtensor_mappping, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--qmodel_path", type=str, required=True)
    args = parser.parse_args()
    static_quant_model(args.model_path, args.qmodel_path)

"""
model_path = "/software/users/yiliu4/HF_HOME/hub/DeepSeek-V3-BF16"
model_path = "/software/users/yiliu4/HF_HOME/hub/deepseekv3-bf16-4l/"
qmodel_path = "/software/users/yiliu4/HF_HOME/hub/deepseekv3-bf16-4l-q/"
static_quant_model(model_path, qmodel_path)
python static_quant.py --model_path /software/users/yiliu4/HF_HOME/hub/deepseekv3-bf16-4l/ --qmodel_path /software/users/yiliu4/HF_HOME/hub/deepseekv3-bf16-4l-q/

"""
