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

# Per-model-type presets: scheme, ignore_layers, layer_config
_MODEL_PRESETS = {
	"kimi": {
		"scheme": "MXFP4",
		"ignore_layers": "shared_experts,self_attn,mlp.gate_proj,mlp.up_proj,mlp.down_proj",
		"layer_config": None,
	},
	"glm": {
		"scheme": "BF16",
		"ignore_layers": None,
		"layer_config": {"mlp.experts": {"scheme": "MXFP4"}},
	},
}


def build_config(args: argparse.Namespace) -> AutoRoundConfig:
	model_type = args.model_type.lower()
	if model_type not in _MODEL_PRESETS:
		raise ValueError(f"Unsupported model_type: {args.model_type}. Supported: {', '.join(_MODEL_PRESETS.keys())}")

	preset = _MODEL_PRESETS[model_type]
	model_free = False if any(
		dtype for dtype in (args.static_kv_dtype, args.static_attention_dtype)
	) else True

	return AutoRoundConfig(
		model_free=model_free,
		iters=0,
		disable_opt_rtn=True,
		scheme=preset["scheme"],
		ignore_layers=preset["ignore_layers"],
		layer_config=preset["layer_config"],
		static_kv_dtype=args.static_kv_dtype,
		static_attention_dtype=args.static_attention_dtype,
		export_format=args.format,
		output_dir=args.output_model,
		reloading=False,
		dataset="HuggingFaceH4/ultrachat_200k",
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Kimi/GLM model-free quantization via INC AutoRound prepare/convert.",
	)
	parser.add_argument(
		"--dtype",
		type=str,
		required=True,
		choices=["mxfp4"],
		help="Quantization dtype. Currently only mxfp4 is supported.",
	)
	parser.add_argument(
		"--input_model",
		type=str,
		required=True,
		help="Model name or local path (e.g. moonshotai/Kimi-K2.6, zai-org/GLM-5.2).",
	)
	parser.add_argument(
		"--output_model",
		type=str,
		required=True,
		help="Output directory for quantized model.",
	)
	parser.add_argument(
		"--model_type",
		type=str,
		required=True,
		choices=["kimi", "glm"],
		help="Model type. Determines quantization config (scheme, ignore_layers, layer_config).",
	)
	parser.add_argument(
		"--format",
		type=str,
		default="llm_compressor",
		choices=["auto_round", "llm_compressor"],
		help="Export format.",
	)
	parser.add_argument(
		"--static_kv_dtype",
		type=str,
		default=None,
		help="Static KV cache data type.",
	)
	parser.add_argument(
		"--static_attention_dtype",
		type=str,
		default=None,
		help="Static attention data type.",
	)
	args = parser.parse_args()

	quant_config = build_config(args)
	model = prepare(args.input_model, quant_config)
	_ = convert(model)
	logger.info("Quantized model saved to %s", args.output_model)


if __name__ == "__main__":
	main()
