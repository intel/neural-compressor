#!/usr/bin/env python3
import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Split i2v info_json into per-shard files for one dimension.")
    parser.add_argument("--info_json", required=True, type=str, help="Path to full i2v info json")
    parser.add_argument("--dimension", required=True, type=str, help="Target dimension")
    parser.add_argument("--num_shards", required=True, type=int, help="Total shard count")
    parser.add_argument("--output_root", required=True, type=str, help="Root dir to write shard json files")
    return parser.parse_args()


def has_dimension(info, target_dimension):
    dims = info.get("dimension", [])
    if isinstance(dims, str):
        dims = [dims]
    return target_dimension in dims


def main():
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not os.path.isfile(args.info_json):
        raise FileNotFoundError(f"Info json not found: {args.info_json}")

    with open(args.info_json, "r", encoding="utf-8") as f:
        info_list = json.load(f)

    filtered = [item for item in info_list if has_dimension(item, args.dimension)]

    shard_buckets = [[] for _ in range(args.num_shards)]
    for idx, item in enumerate(filtered):
        shard_buckets[idx % args.num_shards].append(item)

    os.makedirs(args.output_root, exist_ok=True)
    for shard_id, shard_items in enumerate(shard_buckets):
        shard_dir = os.path.join(args.output_root, f"shard_{shard_id}")
        os.makedirs(shard_dir, exist_ok=True)
        shard_info_json = os.path.join(shard_dir, "info.json")
        with open(shard_info_json, "w", encoding="utf-8") as f:
            json.dump(shard_items, f, ensure_ascii=False, indent=2)

    print(
        f"Split {len(filtered)} i2v entries for dimension '{args.dimension}' "
        f"into {args.num_shards} shards under {args.output_root}"
    )


if __name__ == "__main__":
    main()
