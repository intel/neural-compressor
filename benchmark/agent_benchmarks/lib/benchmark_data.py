#!/usr/bin/env python3
"""Small data-conversion helpers used by the benchmark shell runners."""

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path


def pro_config(args):
    import yaml
    from minisweagent.config import builtin_config_dir

    config = yaml.safe_load((builtin_config_dir / "extra" / "swebench.yaml").read_text())
    config["agent"]["step_limit"] = args.step_limit
    environment = config.setdefault("environment", {})
    environment["pull_timeout"] = args.pull_timeout
    environment["timeout"] = args.command_timeout
    environment["forward_env"] = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ]
    config.setdefault("run", {})[
        "env_startup_command"
    ] = "git clone https://github.com/{{ repo }}.git . && {{ before_repo_set_cmd }}"
    old = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"
    new = (
        "cd /testbed && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
        "git diff -- . ':(exclude)test/**' ':(exclude)tests/**'"
    )
    template = config["agent"]["instance_template"]
    if old not in template:
        raise RuntimeError("Expected submission command not found in mini-SWE-agent config")
    config["agent"]["instance_template"] = template.replace(old, new)
    config["model"] = {
        "model_name": args.model,
        "cost_tracking": "ignore_errors",
        "model_kwargs": {
            "api_base": args.base_url,
            "api_key": args.api_key,
            "drop_params": True,
            "temperature": 0.0,
            "max_tokens": int(os.getenv("AGENT_MAX_TOKENS", "8192")),
        },
    }
    Path(args.output).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print(f"Runtime config: {args.output}")


def normalize_patch(text):
    text = (text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    if not text or "*** Begin Patch" in text:
        return ""

    git_header = re.search(r"(?m)^diff --git ", text)
    if git_header:
        patch = text[git_header.start() :].lstrip()
        file_header = re.search(r"(?m)^--- (?:a/|/dev/null)\S*\n\+\+\+ (?:b/|/dev/null)\S*", patch)
        return patch if file_header and re.search(r"(?m)^@@ ", patch[file_header.end() :]) else ""

    file_header = re.search(r"(?m)^--- \S+\n\+\+\+ \S+", text)
    if not file_header:
        return ""
    patch = text[file_header.start() :].lstrip()
    return patch if re.search(r"(?m)^@@ ", text[file_header.end() :]) else ""


def normalize_pro(args):
    predictions = json.loads(Path(args.source).read_text())
    items = predictions.items() if isinstance(predictions, dict) else enumerate(predictions)
    patches, invalid, test_changes = [], 0, 0
    for key, record in items:
        if not isinstance(record, dict):
            record = {"model_patch": str(record)}
        instance_id = record.get("instance_id") or (key if isinstance(key, str) else None)
        if not instance_id:
            continue
        patch = next(
            (
                patch
                for field in ("model_patch", "patch", "prediction", "completion", "response", "output")
                if isinstance(record.get(field), str) and (patch := normalize_patch(record[field]))
            ),
            "",
        )
        files = re.findall(r"(?m)^diff --git a/.*? b/(.*?)$", patch) or re.findall(r"(?m)^\+\+\+ b/(.*?)$", patch)
        test_suffixes = ("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts", "_test.go")
        if patch and any(
            any(part in {"test", "tests"} for part in Path(path).parts[:-1])
            or Path(path).name.startswith("test_")
            or Path(path).name.endswith(test_suffixes)
            for path in files
        ):
            patch, test_changes = "", test_changes + 1
        if not patch:
            invalid += 1
        patches.append({"instance_id": instance_id, "patch": patch, "prefix": args.prefix})
    Path(args.output).write_text(json.dumps(patches, indent=2))
    print(f"Wrote {len(patches)} patches to {args.output}")
    print(f"Invalid or empty patches: {invalid}")
    print(f"Patches touching test files: {test_changes}")


def select_pro(args):
    import pandas as pd
    from datasets import load_dataset

    instances = list(load_dataset("ScaleAI/SWE-bench_Pro", split="test"))
    if args.slice:
        bounds = [int(value) if value else None for value in args.slice.split(":")]
        instances = instances[slice(*bounds)]
    if not instances:
        raise RuntimeError(f"Dataset slice selected no instances: {args.slice or 'all'}")
    pd.DataFrame(instances).to_csv(args.csv, index=False)
    images = sorted({f"jefzda/sweap-images:{item['dockerhub_tag']}" for item in instances if item.get("dockerhub_tag")})
    Path(args.images).write_text("\n".join(images) + ("\n" if images else ""))
    print(f"Wrote {len(instances)} instances to {args.csv}")
    print(f"Tracked {len(images)} Docker images in {args.images}")


def select_verified(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    instances = list(load_dataset("princeton-nlp/SWE-bench_Verified", split="test"))
    bounds = [int(value) if value else None for value in args.slice.split(":")]
    instances = instances[slice(*bounds)]
    if not instances:
        raise RuntimeError(f"Dataset slice selected no instances: {args.slice}")
    images = [
        "docker.io/swebench/sweb.eval.x86_64." + item["instance_id"].replace("__", "_1776_").lower() + ":latest"
        for item in instances
    ]
    Path(args.images).write_text("\n".join(images) + "\n")
    print(f"Tracked {len(images)} Docker images in {args.images}")


def read_batch_dirs(path: str) -> list[Path]:
    batch_dirs = [Path(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not batch_dirs:
        raise RuntimeError(f"Batch list is empty: {path}")
    return batch_dirs


def merge_prediction_files(paths: list[Path]) -> dict | list:
    merged = None
    for path in paths:
        payload = json.loads(path.read_text())
        if merged is None:
            merged = {} if isinstance(payload, dict) else []
        if isinstance(merged, dict) and isinstance(payload, dict):
            merged.update(payload)
        elif isinstance(merged, list) and isinstance(payload, list):
            merged.extend(payload)
        else:
            raise TypeError(f"Incompatible prediction format in {path}")
    return merged if merged is not None else {}


def merge_verified(args: argparse.Namespace) -> None:
    batch_dirs = read_batch_dirs(args.batch_list)
    predictions = merge_prediction_files([path / "preds.json" for path in batch_dirs])
    Path(args.predictions).write_text(json.dumps(predictions, indent=2))

    records = predictions.items() if isinstance(predictions, dict) else enumerate(predictions)
    with Path(args.jsonl).open("w") as output:
        for instance_id, prediction in records:
            record = prediction if isinstance(prediction, dict) else {"model_patch": prediction}
            record.setdefault("instance_id", instance_id)
            output.write(json.dumps(record) + "\n")

    report_paths = [path / "report.json" for path in batch_dirs if (path / "report.json").is_file()]
    if report_paths:
        report = {}
        for path in report_paths:
            payload = json.loads(path.read_text())
            for key, value in payload.items():
                if key == "schema_version":
                    report[key] = max(report.get(key, value), value)
                elif isinstance(value, list):
                    report.setdefault(key, []).extend(value)
                elif isinstance(value, int) and not isinstance(value, bool):
                    report[key] = report.get(key, 0) + value
                else:
                    report[key] = value
        Path(args.report).write_text(json.dumps(report, indent=2))
    else:
        Path(args.report).unlink(missing_ok=True)
    print(f"Merged {len(batch_dirs)} SWE-bench Verified batches")


def merge_csv_files(paths: list[Path], output: str) -> None:
    header = None
    with Path(output).open("w", newline="") as destination:
        writer = csv.writer(destination)
        for path in paths:
            with path.open(newline="") as source:
                reader = csv.reader(source)
                current_header = next(reader)
                if header is None:
                    header = current_header
                    writer.writerow(header)
                elif current_header != header:
                    raise RuntimeError(f"CSV header mismatch in {path}")
                writer.writerows(reader)


def merge_pro(args: argparse.Namespace) -> None:
    batch_dirs = read_batch_dirs(args.batch_list)
    predictions = merge_prediction_files([path / "preds.json" for path in batch_dirs])
    Path(args.predictions).write_text(json.dumps(predictions, indent=2))

    patches = []
    images = set()
    reports = {}
    for path in batch_dirs:
        patches.extend(json.loads((path / "patches.json").read_text()))
        images.update(line for line in (path / "images.txt").read_text().splitlines() if line)
        report_path = path / "evaluation" / "eval_results.json"
        if report_path.is_file():
            reports.update(json.loads(report_path.read_text()))

    Path(args.patches).write_text(json.dumps(patches, indent=2))
    Path(args.images).write_text("\n".join(sorted(images)) + ("\n" if images else ""))
    merge_csv_files([path / "instances.csv" for path in batch_dirs], args.instances)
    if reports:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(reports, indent=2))
    else:
        Path(args.report).unlink(missing_ok=True)
    print(f"Merged {len(batch_dirs)} SWE-bench Pro batches")


def pro_report(args):
    results = json.loads(Path(args.report).read_text())
    passed, total = sum(bool(value) for value in results.values()), len(results)
    print(f"Local accuracy: {passed}/{total} = {passed / total * 100 if total else 0:.1f}%")
    for instance_id, resolved in sorted(results.items(), key=lambda item: (not item[1], item[0])):
        print(f"  {'PASS' if resolved else 'FAIL'}  {instance_id}")


def verified_jsonl(args):
    predictions = json.loads(Path(args.source).read_text())
    with open(args.output, "w") as output:
        for instance_id, prediction in predictions.items():
            record = prediction if isinstance(prediction, dict) else {"model_patch": prediction}
            record.setdefault("instance_id", instance_id)
            output.write(json.dumps(record) + "\n")
    print(f"Wrote {len(predictions)} predictions to {args.output}")


def verified_ids(args):
    for line in Path(args.source).read_text().splitlines():
        if line.strip():
            print(json.loads(line)["instance_id"])


def verified_report(args):
    report = json.loads(Path(args.report).read_text())
    submitted, resolved = report.get("submitted_instances", 0), report.get("resolved_instances", 0)
    print(f"Local accuracy : {resolved}/{submitted} = {resolved / submitted * 100 if submitted else 0:.1f}%")
    print(f"Completed      : {report.get('completed_instances', 0)}")
    print(f"Unresolved     : {report.get('unresolved_instances', 0)}")
    print(f"Errors         : {report.get('error_instances', 0)}")


def atlas_groundtruth(args):
    from datasets import load_dataset

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dataset = load_dataset("ScaleAI/MCP-Atlas", split="train")
    dataset.to_pandas().to_csv(args.output, index=False)
    print(f"Wrote {len(dataset)} tasks to {args.output}")


def atlas_report(args):
    files = sorted(glob.glob(os.path.join(args.directory, "coverage_stats_*_combined.json")))
    if not files:
        raise RuntimeError(f"No combined coverage report found in {args.directory}")
    report = json.loads(Path(files[-1]).read_text())
    stats = report.get("all", report)
    print(f"Tasks         : {stats.get('total_tasks', '?')}")
    print(f"Pass@0.75     : {stats.get('pass_rate_0.75', '?')}%")
    print(f"Mean coverage : {float(stats.get('mean_coverage', 0)):.4f}")
    print(f"Report        : {files[-1]}")


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)

    command = commands.add_parser("pro-config")
    for name in ("output", "model", "base-url", "api-key"):
        command.add_argument(f"--{name}", required=True)
    for name in ("step-limit", "pull-timeout", "command-timeout"):
        command.add_argument(f"--{name}", required=True, type=int)
    command.set_defaults(func=pro_config)

    command = commands.add_parser("normalize-pro")
    for name in ("source", "output", "prefix"):
        command.add_argument(f"--{name}", required=True)
    command.set_defaults(func=normalize_pro)

    command = commands.add_parser("select-pro")
    command.add_argument("--csv", required=True)
    command.add_argument("--images", required=True)
    command.add_argument("--slice", default="")
    command.set_defaults(func=select_pro)

    command = commands.add_parser("select-verified")
    command.add_argument("--images", required=True)
    command.add_argument("--slice", required=True)
    command.set_defaults(func=select_verified)

    command = commands.add_parser("merge-verified")
    for name in ("batch-list", "predictions", "jsonl", "report"):
        command.add_argument(f"--{name}", required=True)
    command.set_defaults(func=merge_verified)

    command = commands.add_parser("merge-pro")
    for name in ("batch-list", "predictions", "patches", "instances", "images", "report"):
        command.add_argument(f"--{name}", required=True)
    command.set_defaults(func=merge_pro)

    for name, func in (("pro-report", pro_report), ("verified-report", verified_report)):
        command = commands.add_parser(name)
        command.add_argument("--report", required=True)
        command.set_defaults(func=func)

    command = commands.add_parser("verified-jsonl")
    command.add_argument("--source", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(func=verified_jsonl)

    command = commands.add_parser("verified-ids")
    command.add_argument("--source", required=True)
    command.set_defaults(func=verified_ids)

    command = commands.add_parser("atlas-groundtruth")
    command.add_argument("--output", required=True)
    command.set_defaults(func=atlas_groundtruth)

    command = commands.add_parser("atlas-report")
    command.add_argument("--directory", required=True)
    command.set_defaults(func=atlas_report)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
