#!/usr/bin/env python3
"""Select pinned V2 Harbor tasks and summarize only pristine-sandbox rewards."""

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit


def select_tasks(root, subset, num_tasks="", slice_spec=""):
    """Select deterministically from the checked-out task tree, never HF default."""
    tasks = sorted(path for path in (root / "tasks").iterdir() if (path / "task.toml").is_file())
    if len(tasks) != 642:
        raise ValueError(f"Expected the V2 642-task tree, found {len(tasks)}")
    if subset == "hard":
        hard = set((root / "hard51_ids.txt").read_text().splitlines())
        if len(hard) != 51 or not hard.issubset({path.name for path in tasks}):
            raise ValueError("Invalid V2 HARD-51 task list")
        tasks = [path for path in tasks if path.name in hard]
    if num_tasks and slice_spec:
        raise ValueError("--num-tasks and --slice are mutually exclusive")
    if num_tasks:
        if int(num_tasks) <= 0:
            raise ValueError("--num-tasks must be positive")
        tasks = tasks[: int(num_tasks)]
    if slice_spec:
        bounds = [int(value) if value else None for value in slice_spec.split(":")]
        if len(bounds) != 2 or any(value is not None and value < 0 for value in bounds):
            raise ValueError("--slice must use nonnegative START:END")
        tasks = tasks[slice(*bounds)]
    if not tasks:
        raise ValueError("Dataset selection is empty")
    return tasks


def prepare(args):
    import yaml

    endpoint = urlsplit(args.base_url)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password:
        raise ValueError("Model endpoint must be an HTTP(S) URL without credentials")
    tasks = select_tasks(args.root, args.subset, args.num_tasks, args.slice)
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "tasks"
    target.mkdir()
    # Copy the exact official task directories; never rewrite instructions or verifiers.
    for task in tasks:
        shutil.copytree(task, target / task.name)
    config = yaml.safe_load((args.root / "tooling/configs/mini_toolcall.yaml").read_text())
    config["agent"]["step_limit"] = args.step_limit
    # Local served model names need not exist in LiteLLM's price registry.
    config["agent"]["cost_limit"] = 0
    config["model"]["cost_tracking"] = "ignore_errors"
    config["model"].setdefault("model_kwargs", {})["max_tokens"] = args.max_tokens
    (args.output / "agent.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "benchmark": "swebench_pro_v2",
        "dataset_version": "2.0.0",
        "commit": args.commit,
        "agent_version": args.agent_version,
        "subset": args.subset,
        "instance_ids": [task.name for task in tasks],
        "step_limit": args.step_limit,
        "max_tokens": args.max_tokens,
        "model_endpoint": args.base_url,
        "protocol": "locked-agent-then-fresh-sandbox-replay",
    }
    (args.output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Selected {len(tasks)} V2 tasks ({args.subset})")


def check_generation(args):
    """Do not misclassify failed generation or missing captured patches as wrong answers."""
    selected = set(json.loads((args.output / "selection.json").read_text())["instance_ids"])
    seen = set()
    failures = {}
    for path in sorted((args.output / "jobs/generation").glob("*/result.json")):
        trial = json.loads(path.read_text())
        task = trial["task_name"].split("/")[-1]
        if task not in selected or task in seen:
            raise ValueError(f"Unexpected or duplicate generation trial: {task}")
        seen.add(task)
        if trial.get("exception_info"):
            failures[task] = trial["exception_info"]
        elif not (path.parent / "agent/model.patch").is_file():
            failures[task] = {"exception_message": "Missing captured model.patch"}
        else:
            trajectory_path = path.parent / "agent/mini-swe-agent.trajectory.json"
            if not trajectory_path.is_file():
                failures[task] = {"exception_message": "Missing native mini-SWE trajectory"}
                continue
            trajectory = json.loads(trajectory_path.read_text())
            status = trajectory.get("info", {}).get("exit_status")
            # Mini-SWE catches exceptions internally; Harbor can report success
            # even after a dependency/API error before the first model response.
            if status not in {"Submitted", "LimitsExceeded"}:
                failures[task] = {"exception_message": f"mini-SWE exit_status: {status}"}
    for task in selected - seen:
        failures[task] = {"exception_message": "Missing generation trial"}
    if failures:
        (args.output / "generation_errors.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise SystemExit("Generation incomplete; no V2 score reported. See generation_errors.json")


def build_report(output):
    """Count infrastructure errors and missing trials in the selected denominator."""
    manifest = json.loads((output / "selection.json").read_text())
    selected = set(manifest["instance_ids"])
    results = {}
    errors = {}
    for path in sorted((output / "jobs/regrade").glob("*/result.json")):
        trial = json.loads(path.read_text())
        task = trial["task_name"].split("/")[-1]
        if task not in selected or task in results:
            raise ValueError(f"Unexpected or duplicate regrade trial: {task}")
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        exception = trial.get("exception_info")
        if exception or reward not in (0, 1):
            errors[task] = exception or {"exception_message": "Missing or invalid verifier reward"}
        results[task] = not exception and reward == 1
    for task in selected - results.keys():
        results[task] = False
        errors[task] = {"exception_message": "Missing regrade trial"}
    resolved = sorted(task for task, passed in results.items() if passed)
    total = len(selected)
    report = {
        **manifest,
        "score_source": "jobs/regrade",
        "total_instances": total,
        "evaluated_instances": total - len(errors),
        "resolved_instances": len(resolved),
        "error_instances": len(errors),
        "accuracy": len(resolved) / total,
        "accuracy_percent": 100 * len(resolved) / total,
        "resolved_ids": resolved,
        "unresolved_ids": sorted(selected - set(resolved)),
        "error_ids": sorted(errors),
        "errors": errors,
        "results": dict(sorted(results.items())),
    }
    return report


def report(args):
    result = build_report(args.output)
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Fresh-sandbox V2 accuracy: {result['resolved_instances']}/{result['total_instances']} "
        f"= {result['accuracy_percent']:.2f}%; infrastructure/missing errors: {result['error_instances']}"
    )
    if result["error_instances"]:
        raise SystemExit(
            "Incomplete V2 evaluation; see report.json error_ids (errors are not dropped from the denominator)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--subset", choices=("full", "hard"), default="full")
    command.add_argument("--num-tasks", default="")
    command.add_argument("--slice", default="")
    command.add_argument("--step-limit", type=int, required=True)
    command.add_argument("--max-tokens", type=int, required=True)
    command.add_argument("--base-url", required=True)
    command.add_argument("--commit", required=True)
    command.add_argument("--agent-version", required=True)
    command.set_defaults(func=prepare)
    command = commands.add_parser("report")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=report)
    command = commands.add_parser("check-generation")
    command.add_argument("--output", type=Path, required=True)
    command.set_defaults(func=check_generation)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
