# Copyright (c) 2026 Intel Corporation
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
"""Tokenize a HF dataset into a Megatron IndexedDataset (.bin/.idx).

This produces the Megatron IndexedDataset (`.bin/.idx`) format used by the
reference QAD experiments; it only depends on transformers/datasets:

    python prepare_data.py --model Qwen/Qwen2.5-0.5B \
        --dataset nvidia/Nemotron-Cascade-2-SFT-Data --subset chat \
        --field messages --max-samples 2000 --out data/quick/train

Then train with `--data data/quick/train`.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

MAGIC = b"MMIDIDX\x00\x00"


def build_text(tokenizer, messages):
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        if "role" in messages[0]:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        if "from" in messages[0]:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            converted = [
                {"role": role_map.get(m["from"], m["from"]), "content": m["value"]}
                for m in messages
            ]
            return tokenizer.apply_chat_template(
                converted, tokenize=False, add_generation_prompt=False
            )
    return messages if isinstance(messages, str) else json.dumps(messages, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="tokenizer path")
    parser.add_argument("--dataset", default="nvidia/Nemotron-Cascade-2-SFT-Data")
    parser.add_argument("--subset", default="chat")
    parser.add_argument("--split", default="train")
    parser.add_argument("--field", default="messages")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="0 = all samples (streaming)")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--out", required=True, help="output prefix (writes .bin/.idx)")
    args = parser.parse_args()

    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset(
        args.dataset, args.subset, split=args.split, streaming=True
    )

    tokens: list[int] = []
    lengths: list[int] = []
    for i, example in enumerate(ds):
        if args.max_samples and i >= args.max_samples:
            break
        text = build_text(tokenizer, example[args.field])
        ids = tokenizer(text, add_special_tokens=False).input_ids[: args.max_seq_length]
        ids.append(tokenizer.eos_token_id)
        tokens.extend(ids)
        lengths.append(len(ids))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(tokens, dtype=np.int32)
    arr.tofile(str(out) + ".bin")

    pointers = np.zeros(len(lengths), dtype=np.int64)
    if lengths:
        pointers[1:] = np.cumsum(lengths[:-1], dtype=np.int64) * 4
    doc_idx = np.zeros(len(lengths) + 1, dtype=np.int64)
    if lengths:
        doc_idx[1:] = np.cumsum(lengths, dtype=np.int64)

    with open(str(out) + ".idx", "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", 4))  # int32
        f.write(struct.pack("<Q", len(lengths)))
        f.write(struct.pack("<Q", len(doc_idx)))
        f.write(np.asarray(lengths, dtype=np.int32).tobytes())
        f.write(pointers.tobytes())
        f.write(doc_idx.tobytes())

    print(
        f"wrote {out}.bin ({arr.nbytes / 1e6:.1f} MB), {len(lengths)} docs, "
        f"{len(tokens)} tokens",
        flush=True,
    )


if __name__ == "__main__":
    main()
