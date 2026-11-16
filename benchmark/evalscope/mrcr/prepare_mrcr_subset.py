# Copyright (c) 2026 Intel Corporation
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
"""Materialize the fixed 300-sample OpenAI MRCR subset used by run_evalscope.sh.

The subset is described by ``openai_mrcr_300.json`` (committed next to this script): a pinned
``openai/mrcr`` revision plus the shard file, row index and sha256 of every selected sample.

Default mode reads the manifest, downloads the referenced shards from the Hugging Face Hub, verifies
each sample against its recorded sha256 and writes ``<subset>_train.jsonl`` files (``2needle``,
``4needle``, ``8needle``) that evalscope's ``openai_mrcr`` benchmark loads through ``dataset_id``.

``--build_manifest`` regenerates the manifest. Selection rule: for every needle count, keep the rows
whose o200k_base token count (prompt contents + answer, counted like evalscope) lies in
[min_total_tokens, max_total_tokens], then draw ``samples_per_needle`` rows with
``random.Random(seed + needle_count)``.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile

DEFAULT_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openai_mrcr_300.json")
DATASET_REPO = "openai/mrcr"
DATASET_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
NEEDLE_COUNTS = (2, 4, 8)
SHARDS_PER_NEEDLE = 2
SEED = 42
SAMPLES_PER_NEEDLE = 100
TIKTOKEN_ENCODING = "o200k_base"
# 4096 is the lower edge of the first official MRCR bin. 112000 o200k tokens keeps the prompt plus
# 2048 generated tokens within a 131072 context for tokenizers that need up to ~12% more tokens.
MIN_TOTAL_TOKENS = 4096
MAX_TOTAL_TOKENS = 112000
RECORD_FIELDS = (
    "prompt",
    "answer",
    "random_string_to_prepend",
    "n_needles",
    "desired_msg_index",
    "total_messages",
    "n_chars",
)
STAMP_FILE = ".manifest_sha256"


def subset_name(needle_count):
    return f"{needle_count}needle"


def shard_files(needle_count):
    return [f"{needle_count}needle/{needle_count}needle_{i}.parquet" for i in range(SHARDS_PER_NEEDLE)]


def download_shard(filename, revision):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(DATASET_REPO, filename, repo_type="dataset", revision=revision)


def read_rows(path, indices=None):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=list(RECORD_FIELDS))
    if indices is not None:
        table = table.take(indices)
    return table.to_pylist()


def to_record(row):
    return {field: row[field] for field in RECORD_FIELDS}


def record_sha256(record):
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(manifest_path):
    import tiktoken

    enc = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    samples = []
    for needle_count in NEEDLE_COUNTS:
        eligible = []
        for filename in shard_files(needle_count):
            for row_index, row in enumerate(read_rows(download_shard(filename, DATASET_REVISION))):
                messages = json.loads(row["prompt"])
                input_tokens = sum(len(enc.encode(m.get("content", ""))) for m in messages if isinstance(m, dict))
                answer_tokens = len(enc.encode(row["answer"]))
                if MIN_TOTAL_TOKENS <= input_tokens + answer_tokens <= MAX_TOTAL_TOKENS:
                    eligible.append((filename, row_index, input_tokens, answer_tokens, record_sha256(to_record(row))))
        if len(eligible) < SAMPLES_PER_NEEDLE:
            raise RuntimeError(f"Only {len(eligible)} eligible rows for {needle_count} needles.")
        picked = sorted(random.Random(SEED + needle_count).sample(range(len(eligible)), SAMPLES_PER_NEEDLE))
        for index in picked:
            filename, row_index, input_tokens, answer_tokens, sha256 = eligible[index]
            samples.append(
                {
                    "subset": subset_name(needle_count),
                    "file": filename,
                    "row": row_index,
                    "input_tokens": input_tokens,
                    "answer_tokens": answer_tokens,
                    "sha256": sha256,
                }
            )
        print(f"{subset_name(needle_count)}: picked {SAMPLES_PER_NEEDLE} of {len(eligible)} eligible rows")

    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "seed": SEED,
        "samples_per_needle": SAMPLES_PER_NEEDLE,
        "tokenizer": TIKTOKEN_ENCODING,
        "min_total_tokens": MIN_TOTAL_TOKENS,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "samples": samples,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"Wrote {len(samples)} samples to {manifest_path}")


def prepare(manifest_path, output_dir):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_digest = file_sha256(manifest_path)
    subsets = sorted({s["subset"] for s in manifest["samples"]})
    stamp_path = os.path.join(output_dir, STAMP_FILE)
    outputs = [os.path.join(output_dir, f"{subset}_train.jsonl") for subset in subsets]
    if os.path.isfile(stamp_path) and all(os.path.isfile(p) for p in outputs):
        with open(stamp_path, encoding="utf-8") as f:
            if f.read().strip() == manifest_digest:
                print(f"MRCR subset already prepared at {output_dir}")
                return

    by_file = {}
    for sample in manifest["samples"]:
        by_file.setdefault(sample["file"], []).append(sample)
    records = {subset: [] for subset in subsets}
    for filename in sorted(by_file):
        samples = by_file[filename]
        rows = read_rows(download_shard(filename, manifest["revision"]), indices=[s["row"] for s in samples])
        for sample, row in zip(samples, rows):
            record = to_record(row)
            if record_sha256(record) != sample["sha256"]:
                raise RuntimeError(f"sha256 mismatch for {filename} row {sample['row']}")
            records[sample["subset"]].append(record)

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(stamp_path):
        os.remove(stamp_path)
    tmp_dir = tempfile.mkdtemp(dir=output_dir, prefix=".tmp_")
    try:
        for subset, subset_records in records.items():
            name = f"{subset}_train.jsonl"
            with open(os.path.join(tmp_dir, name), "w", encoding="utf-8") as f:
                for record in subset_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(os.path.join(tmp_dir, name), os.path.join(output_dir, name))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    with open(stamp_path, "w", encoding="utf-8") as f:
        f.write(manifest_digest + "\n")
    counts = ", ".join(f"{subset}={len(subset_records)}" for subset, subset_records in records.items())
    print(f"Prepared MRCR subset ({counts}) at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Sample manifest (default: %(default)s).")
    parser.add_argument("--output_dir", help="Directory for the <subset>_train.jsonl files.")
    parser.add_argument("--build_manifest", action="store_true", help="Regenerate --manifest from the dataset.")
    args = parser.parse_args()

    if args.build_manifest:
        build_manifest(args.manifest)
    elif args.output_dir:
        prepare(args.manifest, os.path.expanduser(args.output_dir))
    else:
        parser.error("--output_dir is required unless --build_manifest is set")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"[prepare_mrcr_subset] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
