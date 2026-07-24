#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXT_PRIORITY = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

def load_prompts(prompt_dir: Path):
    prompts = {}
    if not prompt_dir.exists():
        return prompts
    for path in sorted(prompt_dir.glob("*.txt")):
        stem = path.stem
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            prompts[stem] = text
    return prompts


def build_audio_index(audio_dir: Path):
    index = {}
    if not audio_dir.exists():
        return index
    for path in sorted(audio_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in AUDIO_EXT_PRIORITY:
            continue
        index.setdefault(path.stem, []).append(path)
    return index


def pick_audio(paths):
    best = None
    best_rank = 10**9
    for p in paths:
        rank = AUDIO_EXT_PRIORITY.index(p.suffix.lower())
        if rank < best_rank:
            best = p
            best_rank = rank
    return best


def build_manifest(dataset_dir: Path):
    dataset_dir = dataset_dir.resolve()
    img_dir = dataset_dir / "imgs"
    audio_dir = dataset_dir / "audios"
    prompt_dir = dataset_dir / "prompts"

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {img_dir}")

    prompts = load_prompts(prompt_dir)
    audio_index = build_audio_index(audio_dir)

    manifest = {}
    skipped = []

    for image_path in sorted(img_dir.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        sample_id = image_path.stem
        prompt = prompts.get(sample_id, None)
        audio_candidates = audio_index.get(sample_id, [])
        audio_path = pick_audio(audio_candidates) if audio_candidates else None

        reasons = []
        if not prompt:
            reasons.append("missing_prompt")
        if not audio_path:
            reasons.append("missing_audio")
        if reasons:
            skipped.append({"id": sample_id, "reasons": reasons})
            continue

        item = {
            "prompt": prompt or "",
            "image": str(image_path.resolve()),
            "audio": str(audio_path.resolve()) if audio_path else "",
        }
        manifest[sample_id] = item

    return manifest, skipped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build s2v manifest from a local EchoMimicV3 repo"
    )
    parser.add_argument(
        "--repo-dir",
        required=True,
        help="Local path for EchoMimicV3 repo (must already exist).",
    )
    parser.add_argument(
        "--manifest-out",
        required=True,
        help="Output manifest path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo directory not found: {repo_dir}")

    dataset_dir = (repo_dir / "datasets" / "echomimicv3_demos").resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    manifest, skipped = build_manifest(dataset_dir=dataset_dir)

    out_path = Path(args.manifest_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=4), encoding="utf-8")

    summary = {
        "repo_dir": str(repo_dir),
        "dataset_dir": str(dataset_dir),
        "manifest_out": str(out_path),
        "total_samples": len(manifest),
        "skipped_samples": len(skipped),
        "first_skipped": skipped[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
