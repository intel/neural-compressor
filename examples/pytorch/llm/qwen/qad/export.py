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
"""Export a trained QAD checkpoint to a HuggingFace directory.

The trained weights are produced by the Trainer (``model.safetensors`` for full
fine-tuning, an adapter directory for LoRA). FSDP2 keeps parameters as DTensors
during training, so export is a separate process:

    python export.py --model /path/Qwen3.8-27B --format full \
        --state-dict runs/<run>/checkpoint-1000/model.safetensors --out runs/export

    python export.py --model /path/Qwen3.8-27B --format full \
        --adapter runs/<run>/checkpoint-500 --out runs/export-lora
"""

from __future__ import annotations

import argparse

import torch

from neural_compressor.torch.qad import ExportConfig, export, load_model
from neural_compressor.torch.qad.export import load_state_dict_into_model


def parse() -> argparse.Namespace:
    """Parse the export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="original HF checkpoint dir")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--format", choices=["full", "text"], default="full")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory")
    parser.add_argument("--state-dict", default=None, help="full-FT model.safetensors")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load the trained weights and write the HF artifact."""
    args = parse()
    model = load_model(args.model, dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        model = model.merge_and_unload()
    elif args.state_dict:
        load_state_dict_into_model(model, args.state_dict)
    else:
        raise ValueError("pass --adapter (LoRA) or --state-dict (full fine-tuning)")

    config = ExportConfig(format="hf_bf16", keep_vision=args.format == "full")
    out = export(model, None, config, args.out, base_model=args.model)
    print(f"exported {args.format} model to {out}", flush=True)


if __name__ == "__main__":
    main()
