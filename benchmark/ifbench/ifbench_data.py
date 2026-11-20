#!/usr/bin/env python3
"""State and summary helper for IFBench suites."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_state(path: Path) -> dict:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid state file: {path}")
    return payload


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    path.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")


def metrics(values: list[float]) -> dict:
    if not values:
        return {}
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def cmd_suite_start(args: argparse.Namespace) -> None:
    path = Path(args.state)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = utc_now()
    state = {
        "schema_version": 1,
        "name": args.name,
        "kind": "ifbench-suite",
        "dataset": "IFBench",
        "status": "running",
        "pid": args.pid,
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
        "runs_total": args.runs,
        "current_run": 1,
        "runs_finished": 0,
        "runs_succeeded": 0,
        "exit_code": None,
        "message": f"run 1/{args.runs}",
        "meta": {
            "dataset": "IFBench",
            "model": args.model,
            "host": args.host,
            "port": str(args.port),
            "api_url": args.api_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "enable_thinking": args.enable_thinking,
            "workers": args.workers,
            "num_prompts": args.num_prompts,
            "env": getattr(args, "env", "") or "",
            "python": getattr(args, "python", "") or "",
            "command": args.command or "",
        },
        "progress": {
            "current_run": 1,
            "runs_total": args.runs,
            "runs_finished": 0,
            "expected": args.num_prompts,
            "percent": 0.0,
        },
        "runs": [],
        "result": None,
    }
    save_state(path, state)


def cmd_run_start(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    run = {
        "index": args.index,
        "tag": args.tag,
        "status": "running",
        "started_at": args.started_at or utc_now(),
        "finished_at": None,
        "log": args.log,
        "command": args.command or "",
    }
    state.setdefault("runs", []).append(run)
    state["current_run"] = args.index
    state["message"] = f"run {args.index}/{state.get('runs_total', '?')}"
    progress = state.setdefault("progress", {})
    progress["current_run"] = args.index
    save_state(path, state)


def cmd_run_finish(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    run = next((item for item in state.get("runs", []) if item.get("index") == args.index), None)
    if run is None:
        run = {"index": args.index}
        state.setdefault("runs", []).append(run)

    payload = load_json(args.results)
    strict = (payload.get("results") or {}).get("strict") or {}
    loose = (payload.get("results") or {}).get("loose") or {}
    accuracy = strict.get("prompt_level_mean")
    loose_accuracy = loose.get("prompt_level_mean")
    instruction_accuracy = strict.get("instruction_level_mean")
    num_prompts = payload.get("num_prompts")
    completed = args.exit_code == 0 and accuracy is not None

    run.update(
        {
            "status": "completed" if completed else "failed",
            "exit_code": args.exit_code,
            "finished_at": args.finished_at or utc_now(),
            "results": args.results,
            "log": args.log,
            "num_prompts": num_prompts,
            "accuracy": float(accuracy) if accuracy is not None else None,
            "loose_accuracy": float(loose_accuracy) if loose_accuracy is not None else None,
            "instruction_accuracy": float(instruction_accuracy) if instruction_accuracy is not None else None,
            "tokens_generated": args.tokens_generated,
            "prompt_tokens": args.prompt_tokens,
        }
    )
    if accuracy is not None:
        run["submitted"] = num_prompts
        run["resolved"] = round(float(accuracy) * int(num_prompts)) if num_prompts else None
        run["result"] = {"accuracy": float(accuracy)}
    if args.tokens_generated is not None:
        run["tokens_generated"] = args.tokens_generated
    if args.prompt_tokens is not None:
        run["prompt_tokens"] = args.prompt_tokens
    if run.get("started_at") and run.get("finished_at"):
        try:
            start = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(run["finished_at"].replace("Z", "+00:00"))
            run["duration_seconds"] = int((end - start).total_seconds())
        except ValueError:
            pass

    state["runs_finished"] = sum(1 for item in state.get("runs", []) if item.get("status") in ("completed", "failed"))
    state["runs_succeeded"] = sum(1 for item in state.get("runs", []) if item.get("status") == "completed")
    total = state.get("runs_total") or 1
    state["progress"] = {
        "current_run": state.get("current_run", args.index),
        "runs_total": total,
        "runs_finished": state["runs_finished"],
        "expected": state.get("progress", {}).get("expected", num_prompts),
        "percent": round(state["runs_finished"] / total * 100, 1),
    }
    state["message"] = f"run {args.index}/{total} finished"
    save_state(path, state)


def cmd_aggregate(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    records = []
    accuracy_values = []
    loose_values = []
    instruction_values = []
    token_values = []
    prompt_values = []
    for run in sorted(state.get("runs", []), key=lambda item: item.get("index", 0)):
        record = dict(run)
        if run.get("status") == "completed":
            if isinstance(run.get("accuracy"), (int, float)):
                accuracy_values.append(float(run["accuracy"]))
            if isinstance(run.get("loose_accuracy"), (int, float)):
                loose_values.append(float(run["loose_accuracy"]))
            if isinstance(run.get("instruction_accuracy"), (int, float)):
                instruction_values.append(float(run["instruction_accuracy"]))
            if isinstance(run.get("tokens_generated"), (int, float)):
                token_values.append(float(run["tokens_generated"]))
            if isinstance(run.get("prompt_tokens"), (int, float)):
                prompt_values.append(float(run["prompt_tokens"]))
        records.append(record)

    metrics_by_name = {
        "accuracy": metrics(accuracy_values),
        "loose_accuracy": metrics(loose_values),
        "instruction_accuracy": metrics(instruction_values),
    }
    if token_values:
        metrics_by_name["tokens_generated"] = {
            **metrics(token_values),
            "sum": int(sum(token_values)),
        }
    summary = {
        "schema_version": 1,
        "name": state.get("name"),
        "kind": "ifbench-suite",
        "dataset": "IFBench",
        "model": state.get("meta", {}).get("model", ""),
        "mode": "independent",
        "runs_total": state.get("runs_total", len(records)),
        "runs_recorded": len(accuracy_values),
        "started_at": state.get("started_at"),
        "finished_at": utc_now(),
        "metrics": metrics_by_name,
        "runs": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# IFBench: {summary['name']}",
        "",
        f"- Model: {summary.get('model') or 'unknown'}",
        f"- Runs: {summary['runs_total']}",
        "",
        "| Run | Status | Prompts | Strict | Loose | Instruction | Tokens |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for run in records:
        strict = f"{run['accuracy'] * 100:.2f}%" if isinstance(run.get("accuracy"), (int, float)) else "-"
        loose = f"{run['loose_accuracy'] * 100:.2f}%" if isinstance(run.get("loose_accuracy"), (int, float)) else "-"
        instruction = (
            f"{run['instruction_accuracy'] * 100:.2f}%"
            if isinstance(run.get("instruction_accuracy"), (int, float))
            else "-"
        )
        lines.append(
            f"| {run.get('index')} | {run.get('status')} | {run.get('num_prompts', '-')} | "
            f"{strict} | {loose} | {instruction} | {run.get('tokens_generated', '-')} |"
        )
    accuracy = metrics_by_name.get("accuracy") or {}
    if accuracy:
        lines.extend(
            [
                "",
                f"**Mean strict prompt-level: {accuracy['mean'] * 100:.2f}% ± "
                f"{accuracy['std'] * 100:.2f}%** (n={len(accuracy['values'])})",
            ]
        )
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    all_completed = len(accuracy_values) == state.get("runs_total") and all(
        run.get("status") == "completed" for run in records
    )
    state["status"] = "completed" if all_completed else ("partial" if accuracy_values else "failed")
    state["exit_code"] = 0 if all_completed else 1
    state["finished_at"] = utc_now()
    state["message"] = "finished"
    state["runs"] = records
    state["runs_finished"] = sum(1 for run in records if run.get("status") in ("completed", "failed"))
    state["runs_succeeded"] = sum(1 for run in records if run.get("status") == "completed")
    state["result"] = {
        "accuracy_mean": accuracy.get("mean"),
        "accuracy_std": accuracy.get("std", 0.0),
        "runs_recorded": len(accuracy_values),
        "loose_accuracy_mean": (metrics_by_name.get("loose_accuracy") or {}).get("mean"),
    }
    state["progress"] = {
        "current_run": state.get("current_run", len(records)),
        "runs_total": state.get("runs_total", len(records)),
        "runs_finished": state["runs_finished"],
        "percent": (
            100.0 if all_completed else round(state["runs_finished"] / max(1, state.get("runs_total", 1)) * 100, 1)
        ),
    }
    save_state(path, state)
    print("\n".join(lines), end="")


def cmd_suite_fail(args: argparse.Namespace) -> None:
    path = Path(args.state)
    if not path.is_file():
        return
    state = load_state(path)
    state["status"] = args.status
    state["exit_code"] = args.exit_code
    state["finished_at"] = utc_now()
    state["message"] = args.message or state.get("message", "")
    save_state(path, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    start = commands.add_parser("suite-start")
    for name in ("state", "name", "model", "host", "port", "api-url"):
        start.add_argument(f"--{name}", required=True)
    for name in ("temperature", "top-p"):
        start.add_argument(f"--{name}", type=float, required=True)
    start.add_argument("--max-tokens", type=int, required=True)
    start.add_argument("--enable-thinking", action="store_true")
    start.add_argument("--workers", type=int, required=True)
    start.add_argument("--num-prompts", type=int, required=True)
    start.add_argument("--runs", type=int, required=True)
    start.add_argument("--pid", type=int, default=None)
    start.add_argument("--env", default="")
    start.add_argument("--python", default="")
    start.add_argument("--command", default="")
    start.set_defaults(func=cmd_suite_start)

    run_start = commands.add_parser("suite-run-start")
    run_start.add_argument("--state", required=True)
    run_start.add_argument("--index", type=int, required=True)
    run_start.add_argument("--tag", required=True)
    run_start.add_argument("--started-at", default="")
    run_start.add_argument("--log", default="")
    run_start.add_argument("--command", default="")
    run_start.set_defaults(func=cmd_run_start)

    run_finish = commands.add_parser("suite-run-finish")
    run_finish.add_argument("--state", required=True)
    run_finish.add_argument("--index", type=int, required=True)
    run_finish.add_argument("--exit-code", type=int, required=True)
    run_finish.add_argument("--results", default="")
    run_finish.add_argument("--log", default="")
    run_finish.add_argument("--finished-at", default="")
    run_finish.add_argument("--tokens-generated", type=int)
    run_finish.add_argument("--prompt-tokens", type=int)
    run_finish.set_defaults(func=cmd_run_finish)

    aggregate = commands.add_parser("suite-aggregate")
    aggregate.add_argument("--state", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--markdown", required=True)
    aggregate.set_defaults(func=cmd_aggregate)

    fail = commands.add_parser("suite-fail")
    fail.add_argument("--state", required=True)
    fail.add_argument("--status", default="failed")
    fail.add_argument("--exit-code", type=int, default=1)
    fail.add_argument("--message", default="")
    fail.set_defaults(func=cmd_suite_fail)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
