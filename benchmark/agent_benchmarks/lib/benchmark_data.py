#!/usr/bin/env python3
"""Small data-conversion helpers used by the benchmark shell runners."""

import argparse
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
    config.setdefault("environment", {})["pull_timeout"] = args.pull_timeout
    config["environment"]["timeout"] = args.command_timeout
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
    errors = re.compile(
        r"(?i)(traceback|exception|calledprocesserror|no space left on device|"
        r"not a git repository|error response from daemon)"
    )
    if not text or "*** Begin Patch" in text or errors.search(text):
        return ""
    index = text.find("diff --git ")
    if index >= 0:
        text = text[index:].lstrip()
        return text if "--- a/" in text and "+++ b/" in text else ""
    match = re.search(r"(?m)^--- [^\n]+\n\+\+\+ [^\n]+", text)
    return text[match.start() :].lstrip() if match and "@@" in text[match.start() :] else ""


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
