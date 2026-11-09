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
"""Dataset adapters for QAD training.

``PackedDataset`` reads Megatron ``.bin/.idx`` files directly (no Megatron
import) and packs documents end-to-end, matching ``GPTDataset`` with
``reset_attention_mask=False``. HuggingFace ``datasets`` can be plugged in by
the caller; the trainer only needs ``input_ids`` / ``attention_mask`` /
``loss_mask``.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

_HEADER = b"MMIDIDX\x00\x00"
_DTYPE_CODES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float64,
    7: np.float32,
    8: np.uint16,
}

_BATCH_PRINTED = False


class IndexedDataset:
    """Minimal reader for Megatron's indexed token files."""

    def __init__(self, prefix: str) -> None:
        """Parse ``<prefix>.idx`` and memory-map ``<prefix>.bin``."""
        self.prefix = prefix
        with open(f"{prefix}.idx", "rb") as handle:
            if handle.read(9) != _HEADER:
                raise ValueError(f"bad idx header: {prefix}.idx")
            version = struct.unpack("<Q", handle.read(8))[0]
            if version != 1:
                raise ValueError(f"bad idx version {version}: {prefix}.idx")
            dtype_code = struct.unpack("<B", handle.read(1))[0]
            self.dtype = np.dtype(_DTYPE_CODES[dtype_code])
            self.sequence_count = struct.unpack("<Q", handle.read(8))[0]
            self.document_count = struct.unpack("<Q", handle.read(8))[0]
            offset = handle.tell()

        with open(f"{prefix}.idx", "rb") as handle:
            handle.seek(offset)
            self.sequence_lengths = np.frombuffer(handle.read(self.sequence_count * 4), dtype=np.int32).copy()
            self.sequence_pointers = np.frombuffer(handle.read(self.sequence_count * 8), dtype=np.int64).copy()
            self.document_indices = np.frombuffer(handle.read(self.document_count * 8), dtype=np.int64).copy()

        self.data = np.memmap(f"{prefix}.bin", dtype=self.dtype, mode="r")

    def __len__(self) -> int:
        return int(self.sequence_count)

    def get(self, index: int) -> np.ndarray:
        """Return one document's token ids."""
        length = int(self.sequence_lengths[index])
        start = int(self.sequence_pointers[index]) // self.dtype.itemsize
        return np.asarray(self.data[start : start + length])


class PackedDataset(Dataset):
    """Concatenate all documents, then slice fixed-length chunks."""

    def __init__(self, prefix: str, seq_length: int, *, drop_tail: bool = True) -> None:
        """Build the chunk index; ``drop_tail`` mirrors ``drop_last`` packing."""
        self.dataset = IndexedDataset(prefix)
        self.seq_length = seq_length
        self.total_tokens = int(self.dataset.sequence_lengths.sum())
        self.n_chunks = self.total_tokens // seq_length
        if not drop_tail and self.total_tokens % seq_length:
            self.n_chunks += 1
        if self.n_chunks == 0:
            raise ValueError(f"no chunks of length {seq_length} in {prefix}")

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, index: int) -> dict:
        """Return one packed chunk with masks."""
        start = index * self.seq_length
        chunk = np.asarray(self.dataset.data[start : start + self.seq_length])
        if chunk.size < self.seq_length:
            chunk = np.concatenate([chunk, np.zeros(self.seq_length - chunk.size, dtype=chunk.dtype)])
        input_ids = torch.from_numpy(chunk.astype(np.int64, copy=False))
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "loss_mask": torch.ones_like(input_ids, dtype=torch.float32),
        }


def default_collator(features):
    """Stack samples and optionally print a batch fingerprint for A/B runs."""
    global _BATCH_PRINTED
    batch = {key: torch.stack([sample[key] for sample in features], dim=0) for key in features[0]}
    if os.environ.get("NC_QAD_DEBUG_BATCH") in ("1", "all") and (
        os.environ.get("NC_QAD_DEBUG_BATCH") == "all" or not _BATCH_PRINTED
    ):
        _BATCH_PRINTED = True
        ids = batch["input_ids"]
        print(
            f"[batch] rank={os.environ.get('RANK', '0')} shape={tuple(ids.shape)} "
            f"sum={ids.sum().item()} first8={ids.flatten()[:8].tolist()}",
            flush=True,
        )
    return batch


def make_loader(
    prefix: str,
    *,
    seq_length: int,
    micro_batch_size: int,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 1234,
) -> tuple[DataLoader, DistributedSampler]:
    """Build a sharded loader over a packed ``.bin/.idx`` stream."""
    dataset = PackedDataset(prefix, seq_length)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=default_collator,
    )
    return loader, sampler


__all__ = ["IndexedDataset", "PackedDataset", "default_collator", "make_loader"]
