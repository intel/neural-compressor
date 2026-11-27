# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export a trained QAD checkpoint to a HuggingFace directory (``compressed_tensors`` or ``bf16``).

Supports bounded-memory streaming export directly from full-FT checkpoints or PEFT LoRA
adapter directories, including automatic VLM text-tower surgery when ``--model`` points
to a multimodal checkpoint (e.g. ``Qwen3.8-27B``):

    python export.py --model /path/Qwen3.8-27B --format compressed_tensors \\
        --recipe recipes/qwen3_5_27b_mxfp4_fp8_full_fsdp2.yaml \\
        --quant-state runs/amax.json \\
        --state-dict runs/<run>/checkpoint-1000/model.safetensors --out runs/export-ct

    python export.py --model /path/Qwen3.8-27B --format compressed_tensors \\
        --recipe recipes/qwen3_5_27b_mxfp4_fp8_lora_path_b.yaml \\
        --quant-state runs/amax.json \\
        --adapter runs/<run>/checkpoint-500 --out runs/export-lora-ct
"""

from __future__ import annotations

import argparse

from neural_compressor.torch.export import export_hf_checkpoint
from neural_compressor.torch.quantization import FakeQuantConfig


def parse() -> argparse.Namespace:
    """Parse the export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="original base HF / VLM checkpoint dir")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--format",
        choices=["compressed_tensors", "bf16", "hf_bf16", "full", "text"],
        default="compressed_tensors",
    )
    parser.add_argument("--keep-vision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", default=None, help="QAD checkpoint or PEFT adapter directory")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory")
    parser.add_argument("--state-dict", default=None, help="full-FT checkpoint dir or model.safetensors")
    parser.add_argument("--recipe", default=None, help="optional QAD YAML recipe containing quant config")
    parser.add_argument("--quant-state", default=None, help="optional calibrated quantizer_state / amax JSON")
    parser.add_argument("--max-shard-size", default="5GB", help="maximum output safetensors shard size")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Stream-export the trained checkpoint or adapter into a deployable HF directory."""
    args = parse()
    source = args.checkpoint or args.adapter or args.state_dict
    if source is None:
        raise ValueError("pass --checkpoint (or --adapter / --state-dict)")

    if args.format == "full":
        export_fmt, keep_vision = "bf16", True
    elif args.format in ("text", "hf_bf16", "bf16"):
        export_fmt = "bf16"
        keep_vision = False if args.format == "text" else args.keep_vision
    else:
        export_fmt, keep_vision = "compressed_tensors", args.keep_vision

    quant_cfg = FakeQuantConfig.from_yaml(args.recipe) if args.recipe else None
    out = export_hf_checkpoint(
        source,
        args.out,
        export_format=export_fmt,
        quant_config=quant_cfg,
        quant_state_path=args.quant_state,
        base_model_path=args.model,
        keep_vision=keep_vision,
        max_shard_size=args.max_shard_size,
    )
    print(f"exported {export_fmt} (keep_vision={keep_vision}) model to {out}", flush=True)


if __name__ == "__main__":
    main()
