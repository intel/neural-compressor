#!/usr/bin/env python3
import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Split s2v manifest into per-shard JSON files.")
    parser.add_argument("--manifest_path", required=True, type=str, help="Path to full s2v manifest JSON")
    parser.add_argument("--num_shards", required=True, type=int, help="Total shard count")
    parser.add_argument("--output_root", required=True, type=str, help="Root dir to write shard JSON files")
    return parser.parse_args()


def split_list_items(items, num_shards):
    buckets = [[] for _ in range(num_shards)]
    for idx, item in enumerate(items):
        buckets[idx % num_shards].append(item)
    return buckets


def split_dict_items(items, num_shards):
    buckets = [dict() for _ in range(num_shards)]
    for idx, key in enumerate(items.keys()):
        shard_id = idx % num_shards
        buckets[shard_id][key] = items[key]
    return buckets


def main():
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not os.path.isfile(args.manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {args.manifest_path}")

    with open(args.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if isinstance(manifest, list):
        shard_buckets = split_list_items(manifest, args.num_shards)
    elif isinstance(manifest, dict):
        shard_buckets = split_dict_items(manifest, args.num_shards)
    else:
        raise ValueError("Manifest must be a JSON object or list")

    os.makedirs(args.output_root, exist_ok=True)
    written_shards = 0
    for shard_id, shard_manifest in enumerate(shard_buckets):
        if len(shard_manifest) == 0:
            continue
        shard_dir = os.path.join(args.output_root, f"shard_{shard_id}")
        os.makedirs(shard_dir, exist_ok=True)
        shard_manifest_path = os.path.join(shard_dir, "manifest.json")
        with open(shard_manifest_path, "w", encoding="utf-8") as f:
            json.dump(shard_manifest, f, ensure_ascii=False, indent=2)
        written_shards += 1

    total_count = len(manifest)
    print(
        f"Split {total_count} s2v samples into {written_shards} non-empty shards "
        f"(requested {args.num_shards}) under {args.output_root}"
    )


if __name__ == "__main__":
    main()
