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
"""Megatron ``.bin/.idx`` binary token reader and packed sequence dataset for QAD examples.

Reads Megatron ``MMIDIDX`` token files directly via ``numpy.memmap`` without importing
``megatron-core``, packing documents end-to-end to match ``GPTDataset(reset_attention_mask=False)``.
"""

from __future__ import annotations

import struct

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


class IndexedDataset:
    """Minimal memory-mapped reader for Megatron ``<prefix>.idx`` and ``<prefix>.bin`` files."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        with open(f"{prefix}.idx", "rb") as handle:
            if handle.read(9) != _HEADER:
                raise ValueError(f"Invalid Megatron .idx header in {prefix}.idx")
            version = struct.unpack("<Q", handle.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported Megatron .idx version {version} in {prefix}.idx")
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
        length = int(self.sequence_lengths[index])
        start = int(self.sequence_pointers[index]) // self.dtype.itemsize
        return np.asarray(self.data[start : start + length])


class PackedDataset(Dataset):
    """Concatenate all documents in an :class:`IndexedDataset` and slice fixed ``seq_length`` chunks."""

    def __init__(self, prefix: str, seq_length: int, *, drop_tail: bool = True) -> None:
        self.dataset = IndexedDataset(prefix)
        self.seq_length = seq_length
        self.total_tokens = int(self.dataset.sequence_lengths.sum())
        self.n_chunks = self.total_tokens // seq_length
        if not drop_tail and self.total_tokens % seq_length:
            self.n_chunks += 1
        if self.n_chunks == 0:
            raise ValueError(f"No chunks of length {seq_length} found in {prefix}")

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.seq_length
        chunk = np.asarray(self.dataset.data[start : start + self.seq_length])
        if chunk.size < self.seq_length:
            chunk = np.concatenate([chunk, np.zeros(self.seq_length - chunk.size, dtype=chunk.dtype)])
        input_ids = torch.from_numpy(chunk.astype(np.int64, copy=False))
        valid_tokens = min(start + self.seq_length, self.total_tokens) - start
        mask = torch.arange(self.seq_length) < valid_tokens
        labels = input_ids.clone()
        if valid_tokens < self.seq_length:
            labels[valid_tokens:] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": mask.long(),
            "loss_mask": mask.float(),
            "labels": labels,
        }


class SyntheticDataset(Dataset):
    """Deterministic synthetic token dataset for smoke runs and memory profiling."""

    def __init__(self, num_samples: int, seq_length: int, vocab_size: int = 128) -> None:
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,), generator=generator)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "loss_mask": torch.ones_like(input_ids, dtype=torch.float32),
            "labels": input_ids.clone(),
        }


def default_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack fixed-length token samples into a batch dictionary."""
    return {key: torch.stack([sample[key] for sample in features], dim=0) for key in features[0]}


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
    """Build a distributed ``DataLoader`` over a packed Megatron ``.bin/.idx`` token stream."""
    dataset = PackedDataset(prefix, seq_length)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
        seed=seed,
    )
    pin_mem = bool(getattr(torch, "accelerator", None) and torch.accelerator.is_available())
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        collate_fn=default_collator,
    )
    return loader, sampler


__all__ = ["IndexedDataset", "PackedDataset", "SyntheticDataset", "default_collator", "make_loader"]
