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
    "mxfp4": {
        "scheme": "MXFP4",
        "layer_config": None,
    },
    # MXFP8 + experts FP4 mixed setup.
    "mxfp4_mixed": {
        "scheme": "MXFP8",
        "layer_config": {"ffn.experts": {"bits": 4, "data_type": "mx_fp"}},
    },
    "mxfp8": {
        "scheme": "MXFP8",
        "layer_config": None,
    },
    "w4a16": {
        "scheme": "W4A16",
        "layer_config": {"wo_a": {"bits": 16}},
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

    return AutoRoundConfig(
        model_free=True,
        scheme=preset["scheme"],
        ignore_layers=args.ignore_layers,
        layer_config=layer_config,
        export_format=args.export_format,
        output_dir=args.export_path,
        reloading=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek V4 model-free quantization via INC AutoRound prepare/convert.")
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=sorted(_PRESET_CONFIG.keys()),
        help="Quantization preset. e.g. mxfp4 or mxfp4_mixed",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model name or local path.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        required=True,
        help="Output directory for quantized model.",
    )
    parser.add_argument(
        "--ignore_layers",
        type=str,
        default="compressor,indexer.weights_proj",
        help="Comma-separated layer name patterns to skip.",
    )
    parser.add_argument(
        "--export_format",
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
    args = parser.parse_args()

    quant_config = build_config(args)

    model = args.model_name_or_path
    model = prepare(model, quant_config)
    _ = convert(model)
    logger.info("Quantized model saved to %s", args.export_path)


if __name__ == "__main__":
    main()
