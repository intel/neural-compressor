import os
import torch
import tqdm
from loguru import logger
import logging
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import json

logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

# CONSTANTS
SAFETENSORS = "safetensors"
WEIGHT_SCALE_NAME = "scale_weight"
INPUT_SCALE_NAME = "scale_input"
SCALE_DTYPE = torch.bfloat16
SCALE_FILE_NAME = f"scales.{SAFETENSORS}"
FULL_RANGE = torch.finfo(torch.float8_e4m3fn).max
WEIGHT_BACKOFF = 0.5
QUANT_MODULE_TYPES = (torch.nn.Linear,)
SKIP_WEIGHT_LST = {
    "model.norm",
    "layernorm",
    "e_score_correction_bias",
    # "lm_head.weight",
    "embed_tokens",
    "mlp.gate.weight",  # mlp.gate is not linear
}
"""
# https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html?highlight=backoff#supported-json-config-file-options
Similarly, the maxabs value of a weight is scaled to weight_backoff*FP8_143_FULLSCALE. The default values are input_backoff=0.25 and weight_backoff=0.5.
"""
MODEL_STATE_DICT_MAPPING_FILENAME = "model.safetensors.index.json"


def skip_weight(weight_name):
    return any([skip_name in weight_name for skip_name in SKIP_WEIGHT_LST])


def get_cpu_mem_size_in_gb():
    import psutil

    mem = psutil.virtual_memory()
    return mem.available


from quant import quant_tensor


from torch import nn


# Adapted from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/1d044fd82b15f1cedb197a288e50cc96a2c27205/inference/model.py#L91-L108
class FP8QDQLinear(torch.nn.Linear):
    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=FP8QDQLinear.fp8_dtype), requires_grad=True
        )
        self.scale_weight = nn.Parameter(torch.tensor(0, dtype=FP8QDQLinear.dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def dequant_weight_online(self):
        fp8_weight = self.weight
        qdq_weight = fp8_weight.to(FP8QDQLinear.dtype) * self.scale_weight
        return qdq_weight

    def qdq_input(self, bf16_input: torch.Tensor):
        input_scale, input_fp8 = quant_tensor(bf16_input)
        qdq_input_bf16 = input_fp8.to(FP8QDQLinear.dtype) * input_scale
        return qdq_input_bf16

    @classmethod
    def create_from_linear(cls, linear: nn.Linear):
        qdq_linear = cls(linear.in_features, linear.out_features)
        qdq_linear.weight.data = linear.weight.data
        if linear.bias is not None:
            qdq_linear.bias = linear.bias
        return qdq_linear

    def forward(self, bf16_input: torch.Tensor) -> torch.Tensor:
        qdq_input = self.qdq_input(bf16_input)
        qdq_weight = self.dequant_weight_online()
        out = torch.nn.functional.linear(qdq_input, qdq_weight, self.bias)
        return out


def patch_lin():
    logger.warning("Patching torch.nn.Linear to FP8QDQLinear")
    torch.nn.Linear = FP8QDQLinear


def qdq_eval(model_path, not_patch_lin=False):
    import transformers
    from transformers.modeling_utils import no_init_weights
    from patch_for_ds import patch_transformers

    if not not_patch_lin:
        patch_lin()

    def _patch__initialize_weights(self, module):
        print(f"Skipping init_weights ")
        module._is_hf_initialized = True

    transformers.modeling_utils.PreTrainedModel._initialize_weights = _patch__initialize_weights
    patch_transformers()
    with no_init_weights():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    logger.info(f"Patched model: {model}")
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    encode = tokenizer.encode("Hello, !", return_tensors="pt")
    with torch.no_grad():
        output_tokens = model.generate(encode, max_length=10)
        output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        logger.info(f"Output: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qmodel_path", type=str, required=True)
    parser.add_argument("--not_patch_lin", action="store_true", help="Measure float model")
    args = parser.parse_args()
    qdq_eval(args.qmodel_path, not_patch_lin=args.not_patch_lin)
