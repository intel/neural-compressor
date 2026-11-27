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
"""Calibrate per-tensor FP8 ``amax`` for the QAD plan.

This is the fallback when no ModelOpt ``.quant_summary.txt`` exists. It runs
the float model over ``--samples`` packed sequences and records the max
absolute value of every static FP8 target (input and weight), writing:

    {"model.layers.3.self_attn.q_proj": {"weight": 0.12, "input": 8.4}, ...}

Use it as ``quant.amax_source: {file: <out>, translate: identity}``.

For bit-exact parity with the reference experiments prefer the
``.quant_summary.txt``: it calibrates the *fused* ``linear_qkv``/``in_proj``
tensors, while this script calibrates each HF projection separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

EXAMPLE_ROOT = Path(__file__).resolve().parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from data.megatron_indexed_dataset import PackedDataset  # noqa: E402
from modeling.qwen35_text_loader import load_qwen_causal_lm  # noqa: E402

from neural_compressor.torch.quantization import prepare_qat  # noqa: E402
from neural_compressor.torch.quantization.fake_quant import (  # noqa: E402
    FakeQuantConfig,
    calibrate_static_amax,
    save_quantizer_state,
)


def parse() -> argparse.Namespace:
    """Parse the calibration arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", "--config", dest="recipe", required=True, help="QAD recipe YAML")
    parser.add_argument("--model", default=None, help="override model checkpoint path")
    parser.add_argument("--data", required=True, help="Megatron bin/idx prefix")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    """Run calibration forwards and write a lossless ``quantizer_state.json``."""
    args = parse()
    raw = yaml.safe_load(Path(args.recipe).read_text()) or {}
    model_path = args.model or raw.get("model_name_or_path") or (raw.get("model") or {}).get("path")
    if not model_path:
        raise ValueError("Provide `--model <path>` or `model_name_or_path` in the recipe YAML.")
    seq_length = args.seq_length or int(raw.get("seq_length") or (raw.get("train") or {}).get("seq_length", 8192))
    quant_cfg = FakeQuantConfig.from_yaml(args.recipe)
    quant_cfg.quantizer_state_path = None

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_qwen_causal_lm(model_path, dtype=torch.bfloat16 if device.type != "cpu" else torch.float32)
    model = model.to(device).eval()
    prepare_qat(model, quant_cfg)

    dataset = PackedDataset(args.data, seq_length)
    batches = (
        {
            "input_ids": dataset[index]["input_ids"].unsqueeze(0).to(device),
            "attention_mask": dataset[index]["attention_mask"].unsqueeze(0).to(device),
        }
        for index in range(min(args.samples, len(dataset)))
    )
    calibrated = calibrate_static_amax(model, dataloader=batches, num_batches=args.samples)
    save_quantizer_state(model, args.out)
    print(f"calibrated {len(calibrated)} static quantizers -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
