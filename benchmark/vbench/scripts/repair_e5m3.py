#!/usr/bin/env python3
"""Restore exact source proj_out tensors without decoding any quantized weights.

Unchanged weight files are hardlinked, never opened for writing. Touched shards
are streamed into new files. JSON files are independent copies. The audit binds
every tensor's raw bytes, dtype and shape, including all untouched block tensors.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def tensors(directory):
    result = {}
    for path in sorted(directory.glob("*.safetensors")):
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(length))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            if key in result:
                raise ValueError(f"Duplicate tensor: {key}")
            result[key] = (path, 8 + length, info)
    return result


def stream(entry, output=None):
    path, base, info = entry
    begin, end = info["data_offsets"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(base + begin)
        remaining = end - begin
        while remaining:
            data = handle.read(min(8 * 1024 * 1024, remaining))
            if not data:
                raise ValueError(f"Truncated tensor in {path}")
            digest.update(data)
            if output is not None:
                output.write(data)
            remaining -= len(data)
    return dict(dtype=info["dtype"], shape=info["shape"], bytes=end - begin, sha256=digest.hexdigest())


def rewrite(path, entries):
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for key, entry in entries.items():
        info = entry[2]
        size = info["data_offsets"][1] - info["data_offsets"][0]
        header[key] = dict(info, data_offsets=[offset, offset + size])
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for entry in entries.values():
            stream(entry, handle)
        handle.flush()
        os.fsync(handle.fileno())


def repair(original, source, destination):
    original, source, destination = map(lambda p: Path(p).resolve(), (original, source, destination))
    if destination.exists():
        raise ValueError(f"Refusing existing destination: {destination}")
    if destination.is_relative_to(original) or destination.is_relative_to(source):
        raise ValueError("Destination must be separate from input checkpoints")
    plans = {}
    for component in ("transformer", "transformer_2"):
        old, src = tensors(original / component), tensors(source / component)
        removed = {k for k in old if k.startswith("proj_out.")}
        if removed != {"proj_out.bias", "proj_out.weight_packed", "proj_out.weight_scale"}:
            raise ValueError(f"Unexpected proj_out tensors: {removed}")
        restored = {"proj_out.weight", "proj_out.bias"}
        if not restored <= src.keys():
            raise ValueError("Source proj_out incomplete")
        quantized = sorted(k.removesuffix(".weight_packed") for k in old if k.endswith(".weight_packed"))
        unexpected = [k for k in quantized if not k.startswith("blocks.") and k != "proj_out"]
        if unexpected:
            raise ValueError(f"Unexpected quantization outside blocks: {unexpected}")
        plans[component] = old, src, removed, restored, quantized
    destination.mkdir()
    touched = {
        entry[0].relative_to(original)
        for old, _, removed, _, _ in plans.values()
        for k, entry in old.items()
        if k in removed
    }
    for path in original.rglob("*"):
        rel = path.relative_to(original)
        target = destination / rel
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif rel not in touched:
            if path.suffix in (".safetensors", ".bin", ".pt", ".pth"):
                os.link(path, target)
            else:
                shutil.copy2(path, target)
    report = dict(
        original=str(original),
        source=str(source),
        destination=str(destination),
        policy="Exact source bytes, no dequantization or dtype conversion; runtime casts to BF16",
        components={},
    )
    for component, (old, src, removed, restored, quantized) in plans.items():
        target = destination / component
        host = old["proj_out.weight_packed"][0]
        for rel in sorted(touched):
            if rel.parts[0] != component:
                continue
            entries = {k: entry for k, entry in old.items() if entry[0] == original / rel and k not in removed}
            if original / rel == host:
                entries.update({k: src[k] for k in sorted(restored)})
            rewrite(destination / rel, entries)
        quant = read_json(target / "quantization_config.json")
        quant.setdefault("extra_config", {})["proj_out"] = dict(bits=16, act_bits=16)
        write_json(target / "quantization_config.json", quant)
        config = read_json(target / "config.json")
        config["quantization_config"] = quant
        write_json(target / "config.json", config)
        new = tensors(target)
        expected = (old.keys() - removed) | restored
        if new.keys() != expected:
            raise ValueError("Repaired key set mismatch")
        index_files = list(target.glob("*.safetensors.index.json"))
        if len(index_files) != 1:
            raise ValueError("Expected exactly one transformer index")
        index = read_json(index_files[0])
        index["weight_map"] = {k: v[0].name for k, v in sorted(new.items())}
        index.setdefault("metadata", {})["total_size"] = sum(
            v[2]["data_offsets"][1] - v[2]["data_offsets"][0] for v in new.values()
        )
        write_json(index_files[0], index)
        records = {}
        for key in sorted(new):
            baseline = src[key] if key in restored else old[key]
            before, after = stream(baseline), stream(new[key])
            if before != after:
                raise ValueError(f"Tensor changed unexpectedly: {component}.{key}")
            records[key] = dict(after, comparison="source" if key in restored else "original_export")
        # Audit all nonquantized source tensors too, not only proj_out.
        source_equal = []
        for key in sorted(src.keys() & new.keys()):
            actual = {k: records[key][k] for k in ("dtype", "shape", "bytes", "sha256")}
            if stream(src[key]) != actual:
                raise ValueError(f"Nonquantized source tensor mismatch: {component}.{key}")
            source_equal.append(key)
        report["components"][component] = dict(
            removed=sorted(removed),
            restored={k: records[k] for k in sorted(restored)},
            quantized_block_modules=[k for k in quantized if k.startswith("blocks.")],
            unexpected_quantized_modules=[],
            source_equal_nonquantized=source_equal,
            rewritten_shards=[str(r) for r in sorted(touched) if r.parts[0] == component],
            hardlinked_shards=[
                p.name
                for p in target.glob("*.safetensors")
                if p.stat().st_ino == (original / component / p.name).stat().st_ino
            ],
            tensors=records,
        )
    write_json(destination / "repair_audit.json", report)
    print(f"REPAIRED_AND_VERIFIED {destination}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    repair(args.original, args.source, args.destination)
