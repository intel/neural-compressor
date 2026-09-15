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
import json
import logging
import re

from neural_compressor.torch.quantization import AutoRoundConfig, convert, prepare

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_PRESET_CONFIG = {
	"mxfp4": {
		"scheme": "MXFP4",
		"model_name_or_path": "moonshotai/Kimi-K2.6",
		"ignore_layers": "shared_experts,self_attn,mlp.gate_proj,mlp.up_proj,mlp.down_proj",
		"layer_config": None,
	},
}


def parse_layer_config(layer_config: str | None) -> dict | None:
	if not layer_config:
		return None

	candidate = layer_config.strip()
	for content in (candidate, candidate.replace("'", '"')):
		try:
			parsed = json.loads(content)
			if not isinstance(parsed, dict):
				raise ValueError("layer_config must be a JSON object.")
			return parsed
		except json.JSONDecodeError:
			pass

	# Support AutoRound-style shorthand such as: {mlp.experts:{scheme:MXFP4}}
	normalized = re.sub(r"([{,]\s*)([A-Za-z_][\\w.-]*)(\s*:)", r'\1"\2"\3', candidate)
	normalized = re.sub(
		r"(:\s*)([A-Za-z_][\\w.-]*)(\s*[,}])",
		r'\1"\2"\3',
		normalized,
	)
	try:
		parsed = json.loads(normalized)
		if not isinstance(parsed, dict):
			raise ValueError("layer_config must be an object after normalization.")
		return parsed
	except (json.JSONDecodeError, ValueError) as err:
		raise ValueError(
			"Invalid --layer_config format. Use JSON like "
			"'{\"mlp.experts\": {\"scheme\": \"MXFP4\"}}' or AutoRound shorthand "
			"'{mlp.experts:{scheme:MXFP4}}'."
		) from err


def build_config(args: argparse.Namespace) -> AutoRoundConfig:
	dtype_key = args.dtype.lower()
	if dtype_key not in _PRESET_CONFIG:
		raise ValueError(f"Unsupported dtype: {args.dtype}. Supported: {', '.join(_PRESET_CONFIG.keys())}")

	preset = _PRESET_CONFIG[dtype_key]
	ignore_layers = args.ignore_layers or preset["ignore_layers"]
	layer_config = parse_layer_config(args.layer_config) if args.layer_config else preset["layer_config"]
	scheme = args.scheme or preset["scheme"]
	model_free = False if any(
		dtype for dtype in (args.static_kv_dtype, args.static_attention_dtype)
	) else True

	return AutoRoundConfig(
		model_free=model_free,
		iters=0,
		disable_opt_rtn=True,
		scheme=scheme,
		ignore_layers=ignore_layers,
		layer_config=layer_config,
		static_kv_dtype=args.static_kv_dtype,
		static_attention_dtype=args.static_attention_dtype,
		export_format=args.format,
		output_dir=args.output_model,
		reloading=False,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Kimi model-free quantization via INC AutoRound prepare/convert.")
	parser.add_argument(
		"--dtype",
		type=str,
		required=True,
		choices=sorted(_PRESET_CONFIG.keys()),
		help="Quantization preset. e.g. mxfp4",
	)
	parser.add_argument(
		"--input_model",
		type=str,
		default="moonshotai/Kimi-K2.6",
		help="Model name or local path. If not set, use preset default model.",
	)
	parser.add_argument(
		"--output_model",
		type=str,
		required=True,
		help="Output directory for quantized model.",
	)
	parser.add_argument(
		"--ignore_layers",
		type=str,
		default=None,
		help="Comma-separated layer name patterns to skip. If not set, use preset default ignore_layers.",
	)
	parser.add_argument(
		"--scheme",
		type=str,
		default=None,
		help="Override quantization scheme. Example: MXFP4, BF16.",
	)
	parser.add_argument(
		"--layer_config",
		type=str,
		default=None,
		help=(
			"Layer config JSON or AutoRound shorthand. Example JSON: "
			"'{\"mlp.experts\": {\"scheme\": \"MXFP4\"}}'."
		),
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

	preset = _PRESET_CONFIG[args.dtype.lower()]
	model = args.input_model or preset["model_name_or_path"]
	model = prepare(model, quant_config)
	_ = convert(model)
	logger.info("Quantized model saved to %s", args.output_model)


if __name__ == "__main__":
	main()
