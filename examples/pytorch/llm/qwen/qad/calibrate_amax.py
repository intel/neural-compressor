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
import json

import torch

from neural_compressor.torch.qad import QADConfig, load_model
from neural_compressor.torch.qad.data import PackedDataset
from neural_compressor.torch.quantization.fake_quant import compile_plan


def parse() -> argparse.Namespace:
    """Parse the calibration arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="QAD recipe YAML")
    parser.add_argument("--data", required=True, help="Megatron bin/idx prefix")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    """Run the forward passes and write the amax JSON."""
    args = parse()
    config = QADConfig.from_yaml(args.recipe)
    seq_length = args.seq_length or config.train.seq_length
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(config.model.path, dtype=torch.bfloat16).to(device).eval()
    report = compile_plan(
        model,
        config.quant.rules,
        default=config.quant.default,
        exclude=config.quant.exclude,
    )

    targets = {}
    for name, module_config in report.plan.items():
        if module_config.input is None or module_config.input.scheme != "static":
            continue
        module = model.get_submodule(name)
        targets[name] = {"module": module, "input": 0.0}

    def make_hook(key):
        def hook(module, inputs):
            value = float(inputs[0].detach().abs().amax())
            targets[key]["input"] = max(targets[key]["input"], value)

        return hook

    handles = [targets[name]["module"].register_forward_pre_hook(make_hook(name)) for name in targets]
    dataset = PackedDataset(args.data, seq_length)
    with torch.no_grad():
        for index in range(min(args.samples, len(dataset))):
            batch = dataset[index]
            model(
                batch["input_ids"].unsqueeze(0).to(device),
                batch["attention_mask"].unsqueeze(0).to(device),
            )
    for handle in handles:
        handle.remove()

    payload = {}
    for name, entry in targets.items():
        weight = entry["module"].weight.detach().abs().amax().item()
        payload[name] = {"weight": weight, "input": entry["input"]}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"calibrated {len(payload)} FP8 modules -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
