#!/usr/bin/env python3
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Split t2v prompt file into per-shard prompt folders.")
    parser.add_argument("--prompt_file", required=True, type=str, help="Path to <dimension>.txt")
    parser.add_argument("--num_shards", required=True, type=int, help="Total shard count")
    parser.add_argument("--output_root", required=True, type=str, help="Root directory to write shard folders")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not os.path.isfile(args.prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

    dimension = os.path.splitext(os.path.basename(args.prompt_file))[0]

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    os.makedirs(args.output_root, exist_ok=True)

    shard_buckets = [[] for _ in range(args.num_shards)]
    for idx, prompt in enumerate(prompts):
        shard_buckets[idx % args.num_shards].append(prompt)

    for shard_id, shard_prompts in enumerate(shard_buckets):
        shard_dir = os.path.join(args.output_root, f"shard_{shard_id}")
        os.makedirs(shard_dir, exist_ok=True)
        shard_prompt_file = os.path.join(shard_dir, f"{dimension}.txt")
        with open(shard_prompt_file, "w", encoding="utf-8") as f:
            for prompt in shard_prompts:
                f.write(prompt + "\n")

    print(
        f"Split {len(prompts)} prompts from {args.prompt_file} into {args.num_shards} shards under {args.output_root}"
    )


if __name__ == "__main__":
    main()
