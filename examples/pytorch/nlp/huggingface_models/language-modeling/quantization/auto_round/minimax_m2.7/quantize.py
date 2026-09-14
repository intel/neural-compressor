# Copyright (c) 2026 Intel Corporation
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

import argparse
import logging

from neural_compressor.torch.quantization import AutoRoundConfig, convert, prepare

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_PRESET_CONFIG = {
    # MXFP8 globally, with MoE experts downgraded to MXFP4.
    "mxfp4_mixed": {
        "scheme": "MXFP8",
        "layer_config": {"block_sparse_moe": {"scheme": "MXFP4"}},
    },
    "mxfp8": {
        "scheme": "MXFP8",
        "layer_config": None,
    },
    "mxfp4": {
        "scheme": "MXFP4",
        "layer_config": None,
    },
}


def build_config(args: argparse.Namespace) -> AutoRoundConfig:
    dtype_key = args.dtype.lower()
    if dtype_key not in _PRESET_CONFIG:
        raise ValueError(f"Unsupported dtype: {args.dtype}. Supported: {', '.join(_PRESET_CONFIG.keys())}")

    preset = _PRESET_CONFIG[dtype_key]
    layer_config = preset["layer_config"]
    if args.disable_preset_layer_config:
        layer_config = None

    # static kv/attention quantization needs the real model, so model-free mode must be disabled.
    model_free = not any((args.static_kv_dtype, args.static_attention_dtype))

    return AutoRoundConfig(
        model_free=model_free,
        iters=0,
        scheme=preset["scheme"],
        layer_config=layer_config,
        static_kv_dtype=args.static_kv_dtype,
        static_attention_dtype=args.static_attention_dtype,
        export_format=args.format,
        output_dir=args.output_model,
        reloading=False,
        dataset="HuggingFaceH4/ultrachat_200k",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniMax-M2.7 model-free quantization via INC AutoRound prepare/convert."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=sorted(_PRESET_CONFIG.keys()),
        help="Quantization preset. e.g. mxfp4_mixed",
    )
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="Model name or local path.",
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Output directory for quantized model.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="llm_compressor",
        choices=["auto_round", "llm_compressor"],
        help="Export format.",
    )
    parser.add_argument(
        "--disable_preset_layer_config",
        action="store_true",
        help="Disable preset layer_config for the selected dtype.",
    )
    parser.add_argument(
        "--static_kv_dtype",
        type=str,
        default=None,
        help="Static KV cache data type, e.g. fp8.",
    )
    parser.add_argument(
        "--static_attention_dtype",
        type=str,
        default=None,
        help="Static attention data type, e.g. fp8.",
    )
    args = parser.parse_args()

    quant_config = build_config(args)

    model = args.input_model
    model = prepare(model, quant_config)
    _ = convert(model)
    logger.info("Quantized model saved to %s", args.output_model)


if __name__ == "__main__":
    main()
