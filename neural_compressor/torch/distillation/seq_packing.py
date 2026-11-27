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
"""Document-boundary-aware SFT sequence packing with reset ``position_ids`` and ``loss_mask``."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch


def pack_sft_sequences(
    samples: Iterable[Mapping[str, Any]],
    seq_length: int,
    *,
    pad_token_id: int = 0,
    drop_tail: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Pack variable-length tokenized SFT samples into ``seq_length`` bins without cross-document RoPE bleed.

    Each packed document receives reset ``position_ids`` starting from ``0`` (``[0..L1-1, 0..L2-1, ...]``),
    enabling HuggingFace ``flash_attention_2`` varlen boundary detection while masking prompt/padding tokens
    via ``loss_mask`` and ``labels`` (``-100``).
    """
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")

    packed: list[dict[str, torch.Tensor]] = []
    cur_ids: list[torch.Tensor] = []
    cur_pos: list[torch.Tensor] = []
    cur_mask: list[torch.Tensor] = []
    cur_labels: list[torch.Tensor] = []
    cur_len = 0

    def _flush() -> None:
        nonlocal cur_ids, cur_pos, cur_mask, cur_labels, cur_len
        if cur_len == 0:
            return
        if drop_tail and cur_len < seq_length:
            cur_ids, cur_pos, cur_mask, cur_labels, cur_len = [], [], [], [], 0
            return
        ids = torch.cat(cur_ids, dim=0)
        pos = torch.cat(cur_pos, dim=0)
        mask = torch.cat(cur_mask, dim=0)
        labels = torch.cat(cur_labels, dim=0)
        attn = torch.ones_like(ids, dtype=torch.long)
        pad = seq_length - cur_len
        if pad > 0:
            ids = torch.cat([ids, torch.full((pad,), pad_token_id, dtype=torch.long)], dim=0)
            pos = torch.cat([pos, torch.arange(pad, dtype=torch.long)], dim=0)
            mask = torch.cat([mask, torch.zeros((pad,), dtype=torch.float32)], dim=0)
            labels = torch.cat([labels, torch.full((pad,), -100, dtype=torch.long)], dim=0)
            attn = torch.cat([attn, torch.zeros((pad,), dtype=torch.long)], dim=0)
        packed.append(
            {
                "input_ids": ids,
                "attention_mask": attn,
                "position_ids": pos,
                "loss_mask": mask,
                "labels": labels,
            }
        )
        cur_ids, cur_pos, cur_mask, cur_labels, cur_len = [], [], [], [], 0

    for sample in samples:
        ids = torch.as_tensor(sample["input_ids"], dtype=torch.long).flatten()
        if ids.numel() == 0:
            continue
        if ids.numel() > seq_length:
            ids = ids[:seq_length]
        doc_len = int(ids.numel())

        if "labels" in sample:
            lbl = torch.as_tensor(sample["labels"], dtype=torch.long).flatten()[:doc_len]
        else:
            lbl = ids.clone()
        if "loss_mask" in sample:
            lmask = torch.as_tensor(sample["loss_mask"], dtype=torch.float32).flatten()[:doc_len]
        else:
            # First token of a standalone document has no prior context within that document.
            lmask = (lbl != -100).float()
            if doc_len > 0 and "labels" not in sample:
                lmask[0] = 0.0
        if cur_len + doc_len > seq_length:
            _flush()

        if "labels" not in sample:
            lbl = lbl.clone()
            lbl[lmask <= 0] = -100
        if cur_labels:
            # In causal LM CE, logits[t] predicts labels[t+1]; setting token 0 of the newly appended doc to -100
            # ensures the final token of the previous doc does not predict the BOS of the next doc.
            lbl = lbl.clone()
            lbl[0] = -100

        cur_ids.append(ids)
        cur_pos.append(torch.arange(doc_len, dtype=torch.long))
        cur_mask.append(lmask)
        cur_labels.append(lbl)
        cur_len += doc_len
        if cur_len == seq_length:
            _flush()

    _flush()
    return packed


__all__ = ["pack_sft_sequences"]
