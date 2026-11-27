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
"""Convert ModelOpt / Megatron-Bridge Qwen3.5 amax summaries to INC ``quantizer_state`` JSON.

Produces lossless IEEE-754 ``amax_hex`` values consumable via
``FakeQuantConfig(quantizer_state_path=...)`` or ``load_quantizer_state(model, path)``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch

_MEGATRON_LAYER_RE = re.compile(
    r"^(?:(?:module|model|language_model)\.)*decoder\.layers\.(\d+)\.(.+?)\.(input|weight|output)_quantizer(?:\._amax)?$"
)

_MEGATRON_SUBMODULE_MAP: dict[str, tuple[str, ...]] = {
    "self_attention.linear_qkv": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
    ),
    "self_attention.linear_proj": (
        "self_attn.o_proj",
        "linear_attn.out_proj",
    ),
    "self_attention.in_proj": (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
    ),
    "self_attention.out_proj": (
        "linear_attn.out_proj",
    ),
    "mlp.linear_fc1": (
        "mlp.gate_proj",
        "mlp.up_proj",
    ),
    "mlp.linear_fc2": ("mlp.down_proj",),
    "mlp.router": ("mlp.gate",),
    "mlp.shared_experts.linear_fc1": (
        "mlp.shared_expert.gate_proj",
        "mlp.shared_expert.up_proj",
    ),
    "mlp.shared_experts.linear_fc2": ("mlp.shared_expert.down_proj",),
}


def translate_megatron_quantizer_key(
    megatron_key: str,
    *,
    hf_prefix: str = "model.language_model.layers",
) -> list[str]:
    """Translate one Megatron-Bridge quantizer key into candidate HuggingFace quantizer keys."""
    match = _MEGATRON_LAYER_RE.match(megatron_key)
    if match is None:
        cleaned = megatron_key[:-6] if megatron_key.endswith("._amax") else megatron_key
        return [cleaned]

    layer_idx, submod, slot = match.groups()
    targets = _MEGATRON_SUBMODULE_MAP.get(submod)
    if targets is None:
        expert_match = re.match(r"^mlp\.experts\.linear_fc([12])\.weight(\d+)$", submod)
        if expert_match:
            fc_id, exp_idx = expert_match.groups()
            projs = ("gate_proj", "up_proj") if fc_id == "1" else ("down_proj",)
            targets = tuple(f"mlp.experts.{exp_idx}.{p}" for p in projs)
        else:
            targets = (submod,)

    return [f"{hf_prefix}.{layer_idx}.{target}.{slot}_quantizer" for target in targets]


_TEXT_SUMMARY_RE = re.compile(
    r"^\s*(\S+)\s+TensorQuantizer\(.*?amax=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\b"
)


def extract_raw_amax_map(source: str | Path | Mapping[str, Any]) -> dict[str, float]:
    """Read float amax entries from a ModelOpt ``.pt`` state file, JSON summary, ``.txt`` log, or dict."""
    if isinstance(source, Mapping):
        raw_obj: Any = source
    else:
        path = Path(source)
        if path.suffix in (".pt", ".pth", ".bin"):
            raw_obj = torch.load(path, map_location="cpu", weights_only=False)
        else:
            text = path.read_text()
            stripped = text.lstrip()
            if stripped.startswith("{"):
                raw_obj = json.loads(text)
            else:
                parsed_txt: dict[str, float] = {}
                for line in text.splitlines():
                    m = _TEXT_SUMMARY_RE.match(line)
                    if m is not None:
                        parsed_txt[m.group(1)] = float(m.group(2))
                return parsed_txt

    if isinstance(raw_obj, Mapping) and "modelopt_state_weights" in raw_obj:
        raw_obj = raw_obj["modelopt_state_weights"]
    if isinstance(raw_obj, Mapping) and "quantizer_state" in raw_obj:
        raw_obj = raw_obj["quantizer_state"]
    if isinstance(raw_obj, Mapping) and "quantizers" in raw_obj:
        raw_obj = raw_obj["quantizers"]

    if not isinstance(raw_obj, Mapping):
        raise TypeError(f"Unsupported amax container type: {type(raw_obj).__name__}")

    out: dict[str, float] = {}
    for key, val in raw_obj.items():
        if isinstance(val, Mapping):
            if "_amax" in val:
                val = val["_amax"]
            elif "amax_hex" in val and val["amax_hex"] is not None:
                out[str(key)] = float.fromhex(str(val["amax_hex"]))
                continue
            elif "amax" in val:
                val = val["amax"]
            else:
                continue
        if isinstance(val, torch.Tensor):
            if val.numel() == 0:
                continue
            out[str(key)] = float(val.detach().to(torch.float32).amax().cpu())
        elif isinstance(val, (int, float)):
            out[str(key)] = float(val)
    return out


def convert_modelopt_amax(
    source: str | Path | Mapping[str, Any],
    output_path: str | Path | None = None,
    *,
    hf_prefixes: tuple[str, ...] = ("model.language_model.layers", "model.layers"),
) -> dict[str, dict[str, Any]]:
    """Convert Megatron/HF ModelOpt amax state into INC lossless ``quantizers`` dict."""
    raw_amax = extract_raw_amax_map(source)
    quantizers: dict[str, dict[str, Any]] = {}
    for key, val in sorted(raw_amax.items()):
        entry = {
            "format": "fp8_e4m3",
            "dynamic": False,
            "amax": float(val),
            "amax_hex": float(val).hex(),
        }
        if _MEGATRON_LAYER_RE.match(key):
            for prefix in hf_prefixes:
                for hf_key in translate_megatron_quantizer_key(key, hf_prefix=prefix):
                    quantizers[hf_key] = dict(entry)
        else:
            cleaned = key[:-6] if key.endswith("._amax") else key
            quantizers[cleaned] = entry

    payload = {"version": 1, "quantizers": quantizers}
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, help="Input ModelOpt .pt or JSON amax file")
    parser.add_argument("--output", "-o", required=True, help="Output INC quantizer_state.json path")
    parser.add_argument(
        "--hf-prefix",
        action="append",
        default=None,
        help="HF layer prefix(es) to emit (default: both model.language_model.layers and model.layers)",
    )
    args = parser.parse_args()
    prefixes = tuple(args.hf_prefix) if args.hf_prefix else ("model.language_model.layers", "model.layers")
    payload = convert_modelopt_amax(args.input, args.output, hf_prefixes=prefixes)
    print(f"Wrote {len(payload['quantizers'])} quantizer entries to {args.output}")


if __name__ == "__main__":
    main()
