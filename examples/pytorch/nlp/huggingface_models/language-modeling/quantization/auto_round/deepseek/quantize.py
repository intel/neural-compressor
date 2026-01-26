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
        "fp_layers": "lm_head",
        "iters": 0,
    },
    "mxfp4": {
        "scheme": "MXFP4_RCEIL",
        "fp_layers": "lm_head,self_attn",
        "iters": 0,
    },
    "nvfp4": {
        "scheme": "NVFP4",
        "fp_layers": "lm_head,self_attn",
        "iters": 0,
        "export_format": "llm_compressor",
        "low_cpu_mem_usage": True,
        "low_gpu_mem_usage": True,
        "reloading":False,
    },
}


def get_model_and_tokenizer(model_name):
    # Load model and tokenizer
    fp32_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        trust_remote_code=False,
        dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=False,
    )
    return fp32_model, tokenizer


def quant_model(args):
    from neural_compressor.torch.quantization import (
        AutoRoundConfig,
        convert,
        prepare,
    )

    config = topologies_config[args.t]
    export_format = config.get("export_format", "auto_round")
    output_dir = f"{args.output_dir}/quantized_model_{args.t}"
    fp32_model, tokenizer = get_model_and_tokenizer(args.model)
    quant_config = AutoRoundConfig(
        tokenizer=tokenizer,
        scheme=config["scheme"],
        enable_torch_compile=args.enable_torch_compile,
        iters=config["iters"],
        fp_layers=config["fp_layers"],
        export_format=export_format,
        output_dir=output_dir,
        low_gpu_mem_usage=True,
        static_kv_dtype=args.static_kv_dtype,
        reloading=False,
        trust_remote_code=False,
        disable_trust_remote_code=True,
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
        "--static_kv_dtype",
        type=str,
        default=None,
        help="Data type to use KV Cache. e.g. fp8",
    )
    parser.add_argument(
        "--use_autoround_format",
        action="store_true",
        help="Use AutoRound format for saving the quantized model.",
    )

    parser.add_argument(
        "--skip_attn",
        action="store_true",
        help="Skip quantize attention layers.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=0,
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
