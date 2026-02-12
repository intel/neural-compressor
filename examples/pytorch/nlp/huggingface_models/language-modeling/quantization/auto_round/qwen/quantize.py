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
        "iters": 200,
    },
    "mxfp4_fp8kv": {
        "scheme": "MXFP4_RCEIL",
        "fp_layers": "lm_head,mlp.gate,self_attn",
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


def quant_model(args):
    from neural_compressor.torch.quantization import (
        AutoRoundConfig,
        convert,
        prepare,
    )
    if args.t == "mxfp4" and args.kv_cache_dtype == "fp8":
        args.t = "mxfp4_fp8kv"
    config = topologies_config[args.t]
    export_format = "auto_round" if args.use_autoround_format else "llm_compressor"
    output_dir = f"{args.output_dir}/quantized_model_{args.t}"
    static_kv_dtype = args.static_kv_dtype if args.static_kv_dtype is not None else config.get("static_kv_dtype", None)
    if static_kv_dtype is not None and static_kv_dtype.lower() != "fp8":
        raise ValueError("Only 'fp8' is supported for static_kv_dtype currently.")
    iters = args.iters if args.iters is not None else config["iters"]
    if (static_kv_dtype == "fp8" or args.static_attention_dtype == "fp8") and iters > 0:
        logger.warning("When using static kv dtype or static attn dtype as fp8, setting iters to 0.")
        iters = 0
    fp32_model, tokenizer = get_model_and_tokenizer(args.model)
    quant_config = AutoRoundConfig(
        tokenizer=tokenizer,
        scheme=config["scheme"],
        enable_torch_compile=True,
        iters=iters,
        fp_layers=config["fp_layers"],
        export_format=export_format,
        disable_opt_rtn=True,
        low_gpu_mem_usage=True,
        static_kv_dtype=static_kv_dtype,
        static_attention_dtype=args.static_attention_dtype,
        output_dir=output_dir,
        reloading=False,
    )

    # quantizer execute
    model = prepare(model=fp32_model, quant_config=quant_config)
    inc_model = convert(model)
    logger.info(f"Quantized model saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Select a quantization scheme.")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the pre-trained model or model identifier from Hugging Face Hub.",
    )
    parser.add_argument(
        "-t",
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
        "--use_autoround_format",
        action="store_true",
        help="Use AutoRound format for saving the quantized model.",
    )
    parser.add_argument(
        "--kv_cache_dtype",
        type=str,
        choices=["fp8", "auto"],
        default="auto",
        help="Data type for KV cache. Options are 'fp8' or 'auto'.",
    )
    parser.add_argument(
        "--static_attention_dtype",
        type=str,
        choices=["fp8", None],
        help="Data type to use Attention Cache. e.g. fp8",
    )
    parser.add_argument(
        "--skip_attn",
        action="store_true",
        help="Skip quantize attention layers.",
    )
    parser.add_argument(
        "--static_kv_dtype",
        type=str,
        default=None,
        help="Data type to use KV Cache. e.g. fp8",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Number of iterations for quantization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the quantized model.",
    )

    args = parser.parse_args()

    quant_model(args)
