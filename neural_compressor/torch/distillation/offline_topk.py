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
"""Offline precomputed Teacher Top-K extraction and data collation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset


@torch.no_grad()
def extract_teacher_topk_batch(
    teacher_logits: torch.Tensor,
    top_k: int = 4096,
    *,
    values_dtype: torch.dtype = torch.bfloat16,
    indices_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract compact ``(topk_values, topk_indices)`` tensors suitable for disk caching."""
    eff_k = min(top_k, teacher_logits.shape[-1])
    vals, idx = torch.topk(teacher_logits, k=eff_k, dim=-1)
    return vals.to(dtype=values_dtype).cpu(), idx.to(dtype=indices_dtype).cpu()


def save_precomputed_topk_shard(
    samples: Sequence[Mapping[str, torch.Tensor]],
    path: Union[str, Path],
) -> Path:
    """Persist a list of sample dicts containing ``input_ids`` and ``teacher_topk_{values,indices}``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialized = [{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in s.items()} for s in samples]
    torch.save(serialized, out)
    return out


class PrecomputedTopKDataset(Dataset):
    """Dataset wrapper loading offline teacher Top-K ``.pt`` shards produced by :func:`save_precomputed_topk_shard`."""

    def __init__(self, shard_paths: Union[str, Path, Iterable[Union[str, Path]]]) -> None:
        if isinstance(shard_paths, (str, Path)):
            root = Path(shard_paths)
            paths = sorted(root.glob("*.pt")) if root.is_dir() else [root]
        else:
            paths = [Path(p) for p in shard_paths]
        if not paths:
            raise FileNotFoundError(f"No precomputed Top-K .pt shards found at {shard_paths!r}")
        self.samples: list[dict[str, Any]] = []
        for p in paths:
            loaded = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(loaded, list):
                self.samples.extend(loaded)
            else:
                raise TypeError(f"Expected list of sample dicts in {p}, got {type(loaded).__name__}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class PrecomputedTopKCollator:
    """Pad and stack batches containing ``input_ids`` plus offline ``teacher_topk_values`` & ``teacher_topk_indices``."""

    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(int(f["input_ids"].shape[0]) for f in features)
        batch_ids = []
        batch_attn = []
        batch_loss_mask = []
        batch_vals = []
        batch_idx = []
        batch_pos = []
        has_pos = "position_ids" in features[0]

        for feat in features:
            ids = torch.as_tensor(feat["input_ids"], dtype=torch.long)
            seq_len = ids.shape[0]
            pad = max_len - seq_len

            attn = (
                torch.as_tensor(feat["attention_mask"], dtype=torch.long)
                if "attention_mask" in feat
                else torch.ones(seq_len, dtype=torch.long)
            )
            lmask = torch.as_tensor(feat["loss_mask"], dtype=torch.float32) if "loss_mask" in feat else attn.float()
            vals = torch.as_tensor(feat["teacher_topk_values"])
            inds = torch.as_tensor(feat["teacher_topk_indices"], dtype=torch.long)

            if pad > 0:
                ids = torch.cat([ids, ids.new_full((pad,), self.pad_token_id)], dim=0)
                attn = torch.cat([attn, attn.new_zeros((pad,))], dim=0)
                lmask = torch.cat([lmask, lmask.new_zeros((pad,))], dim=0)
                vals = torch.cat([vals, vals.new_zeros((pad, vals.shape[-1]))], dim=0)
                inds = torch.cat([inds, inds.new_zeros((pad, inds.shape[-1]))], dim=0)

            batch_ids.append(ids)
            batch_attn.append(attn)
            batch_loss_mask.append(lmask)
            batch_vals.append(vals)
            batch_idx.append(inds)
            if has_pos:
                pos = torch.as_tensor(feat["position_ids"], dtype=torch.long)
                if pad > 0:
                    pos = torch.cat([pos, torch.arange(pad, dtype=torch.long)], dim=0)
                batch_pos.append(pos)

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(batch_ids, dim=0),
            "attention_mask": torch.stack(batch_attn, dim=0),
            "loss_mask": torch.stack(batch_loss_mask, dim=0),
            "teacher_topk_values": torch.stack(batch_vals, dim=0),
            "teacher_topk_indices": torch.stack(batch_idx, dim=0),
        }
        if has_pos:
            out["position_ids"] = torch.stack(batch_pos, dim=0)
        return out


__all__ = [
    "PrecomputedTopKCollator",
    "PrecomputedTopKDataset",
    "extract_teacher_topk_batch",
    "save_precomputed_topk_shard",
]
