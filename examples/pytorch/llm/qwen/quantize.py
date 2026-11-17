# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


topologies_config = {
    "mxfp8": {
        "scheme": "MXFP8",
        "fp_layers": "lm_head,mlp.gate",
        "iters": 0,
    },
    "mxfp4": {
        "scheme": "MXFP4_RCEIL",
        "fp_layers": "lm_head,mlp.gate,self_attn",
        "iters": 0,
    },
    "nvfp4": {
        "scheme": "NVFP4",
        "fp_layers": "lm_head,mlp.gate,self_attn",
        "iters": 0,
    },
    "mxfp4_fp8kv": {
        "scheme": "MXFP4_RCEIL",
        "fp_layers": "lm_head,mlp.gate,self_attn",
        "iters": 0,
        "static_kv_dtype": "fp8",
    },
}

dense_topologies_config = {
    "mxfp8": {
        "scheme": "MXFP8",
        "fp_layers": "lm_head",
        "iters": 0,
    },
    "nvfp4": {
        "scheme": "NVFP4",
        "fp_layers": "lm_head,self_attn",
        "iters": 0,
    },
    "mxfp4": {
        "scheme": "MXFP4",
        "fp_layers": "lm_head,self_attn",
        "iters": 0,
    },
    "mxfp4_fp8kv": {
        "scheme": "MXFP4",
        "fp_layers": "lm_head,self_attn",
        "iters": 0,
        "static_kv_dtype": "fp8",
    },
}


def get_model_and_tokenizer(model_name):
    # Load model and tokenizer
    fp32_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        trust_remote_code=True,
        dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    return fp32_model, tokenizer

def is_dense_model(model_name):
    dense_model_lst = ["Qwen3-32B", "Qwen3-8B", "Qwen3-0.6B"]
    return any(dense_model in model_name for dense_model in dense_model_lst)


def quant_model(args):
    from neural_compressor.torch.quantization import (
        AutoRoundConfig,
        convert,
        prepare,
    )
    if args.dtype == "mxfp4" and args.static_kv_dtype == "fp8":
        args.dtype = "mxfp4_fp8kv"
    if is_dense_model(args.model_name_or_path):
        config = dense_topologies_config[args.dtype]
    else:
        config = topologies_config[args.dtype]
    output_dir = f"{args.export_path}"
    static_kv_dtype = args.static_kv_dtype if args.static_kv_dtype is not None else config.get("static_kv_dtype", None)
    if static_kv_dtype is not None and static_kv_dtype.lower() != "fp8":
        raise ValueError("Only 'fp8' is supported for static_kv_dtype currently.")
    iters = args.iters if args.iters is not None else config["iters"]
    if (static_kv_dtype == "fp8" or args.static_attention_dtype == "fp8") and iters > 0:
        logger.warning("When using static kv dtype or static attn dtype as fp8, setting iters to 0.")
        iters = 0
    fp32_model, tokenizer = get_model_and_tokenizer(args.model_name_or_path)
    # if export_format is llm_compressor, scheme with RCEIL is not supported. 
    scheme = config["scheme"] if args.export_format == "auto_round" else config["scheme"].replace("_RCEIL", "")
    quant_config = AutoRoundConfig(
        tokenizer=tokenizer,
        scheme=scheme,
        enable_torch_compile=True,
        iters=iters,
        ignore_layers=config["fp_layers"],
        export_format=args.export_format,
        disable_opt_rtn=True,
        low_gpu_mem_usage=True,
        static_kv_dtype=static_kv_dtype,
        static_attention_dtype=args.static_attention_dtype,
        output_dir=output_dir,
        device_map=args.device_map,
        reloading=False,
    )

    # quantizer execute
    model = prepare(model=fp32_model, quant_config=quant_config)
    convert(model)
    logger.info(f"Quantized model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select a quantization scheme.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to the pre-trained model or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=topologies_config.keys(),
        default="mxfp4",
        help="Quantization scheme to use. Available options: " + ", ".join(topologies_config.keys()),
    )

    parser.add_argument(
        "--enable_torch_compile",
        action="store_true",
        help="Enable torch compile for the model.",
    )
    parser.add_argument(
        "--export_format",
        type=str,
        choices=["auto_round", "llm_compressor"],
        default="llm_compressor",
        help="Export format for the quantized model. Options are 'auto_round' or 'llm_compressor'.",
    )
    parser.add_argument(
        "--static_attention_dtype",
        default=None,
        type=str,
        choices=["fp8", "float8_e4m3fn"],
        help="Data type for static quantize attention.",
    )
    parser.add_argument(
        "--skip_attn",
        action="store_true",
        help="Skip quantize attention layers.",
    )
    parser.add_argument(
        "--static_kv_dtype",
        default=None,
        type=str,
        choices=["fp8", "float8_e4m3fn"],
        help="Data type for static quantize key and value.",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of iterations for quantization.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="saved_results",
        help="Directory to save the quantized model.",
    )
    parser.add_argument(
        "--device_map", 
        type=str, 
        default="auto", 
        help="device map for model",
    )

    args = parser.parse_args()

    quant_model(args)
