#!/usr/bin/env python3
"""State and summary helper for the EvalScope seven-run suite."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

SCORE_KEYS = ("score", "accuracy", "pass@1", "pass_at_1", "acc")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def load_state(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"State file not found: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid state file: {path}")
    return payload


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    save_json(path, state)


def metrics(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def normalize_score(value) -> float:
    score = float(value)
    # EvalScope reports are usually fractions; tolerate percent-style reports.
    if 1.0 < score <= 100.0:
        score /= 100.0
    return score


def report_score(payload) -> float | None:
    metrics_value = payload.get("metrics")
    rows = metrics_value if isinstance(metrics_value, list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in SCORE_KEYS:
            if key in row and row[key] is not None:
                return normalize_score(row[key])
    for key in SCORE_KEYS:
        if key in payload and payload[key] is not None:
            return normalize_score(payload[key])
    return None


def find_report(work_dir: Path, dataset: str) -> Path | None:
    candidates = sorted(work_dir.glob(f"**/reports/*/{dataset}.json"))
    if candidates:
        return candidates[-1]
    candidates = sorted(work_dir.glob(f"**/{dataset}.json"))
    return candidates[-1] if candidates else None


def collect_scores(work_dir: Path, tasks: list[str]) -> tuple[dict, dict]:
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    for task in tasks:
        report = find_report(work_dir, task)
        if report is None:
            errors[task] = "report missing"
            continue
        try:
            score = report_score(load_json(report))
        except Exception as error:  # noqa: BLE001 - report the malformed file
            errors[task] = f"{type(error).__name__}: {error}"
            continue
        if score is None:
            errors[task] = f"no score in {report}"
            continue
        scores[task] = score
    return scores, errors


def cmd_suite_start(args: argparse.Namespace) -> None:
    tasks = args.tasks.split()
    state = {
        "schema_version": 1,
        "name": args.name,
        "kind": "evalscope-suite",
        "status": "running",
        "pid": args.pid,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "runs_total": args.runs,
        "current_run": 1,
        "runs_finished": 0,
        "runs_succeeded": 0,
        "exit_code": None,
        "message": f"run 1/{args.runs}",
        "meta": {
            "dataset": "EvalScope",
            "tasks": tasks,
            "model": args.model,
            "host": args.host,
            "port": str(args.port),
            "api_url": args.api_url,
            "reasoning_effort": args.reasoning_effort,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "env": getattr(args, "env", "") or "",
            "python": getattr(args, "python", "") or "",
            "command": args.command or "",
        },
        "progress": {
            "expected": len(tasks),
            "runs_finished": 0,
            "runs_total": args.runs,
            "percent": 0.0,
        },
        "result": None,
        "runs": [
            {
                "index": index,
                "tag": f"{args.name}_run{index:02d}",
                "status": "pending",
                "datasets": {},
                "accuracy": None,
                "result": {},
            }
            for index in range(1, args.runs + 1)
        ],
    }
    save_state(Path(args.state), state)


def cmd_suite_run_start(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    run = next((item for item in state.get("runs", []) if item.get("index") == args.index), None)
    if run is None:
        run = {"index": args.index}
        state.setdefault("runs", []).append(run)
    run.update(
        {
            "tag": args.tag,
            "status": "running",
            "exit_code": None,
            "started_at": args.started_at or utc_now(),
            "finished_at": None,
            "log": args.log,
        }
    )
    if args.command:
        run["command"] = args.command
        state.setdefault("meta", {}).setdefault("run_commands", {})[args.tag] = args.command
    elif "command" not in run:
        run["command"] = ""
    state["current_run"] = args.index
    state["status"] = "running"
    state["message"] = f"run {args.index}/{state.get('runs_total', args.index)}"
    save_state(path, state)


def cmd_suite_run_finish(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    run = next((item for item in state.get("runs", []) if item.get("index") == args.index), None)
    if run is None:
        run = {"index": args.index}
        state.setdefault("runs", []).append(run)
    tasks = args.tasks.split()
    scores, errors = collect_scores(Path(args.work_dir), tasks)
    if len(scores) == len(tasks):
        status = "completed"
    elif scores:
        status = "partial"
    else:
        status = "failed"
    accuracy = statistics.mean(scores.values()) if scores else None
    run.update(
        {
            "status": status,
            "exit_code": args.exit_code,
            "finished_at": args.finished_at or utc_now(),
            "log": args.log,
            "work_dir": args.work_dir,
            "datasets": scores,
            "errors": errors,
            "accuracy": accuracy,
            "result": {
                "value": accuracy,
                "accuracy": accuracy,
                "datasets": scores,
            },
        }
    )
    runs = state.get("runs", [])
    state["runs_finished"] = sum(
        1 for item in runs if item.get("status") in ("completed", "partial", "failed")
    )
    state["runs_succeeded"] = sum(1 for item in runs if item.get("status") == "completed")
    total = state.get("runs_total") or len(runs) or 1
    state["progress"] = {
        "expected": len(tasks),
        "runs_finished": state["runs_finished"],
        "runs_total": total,
        "percent": round(state["runs_finished"] / total * 100, 1) if total else 0.0,
    }
    state["message"] = f"run {args.index}/{total} finished"
    save_state(path, state)


def render_markdown(summary: dict) -> str:
    tasks = summary.get("tasks", [])
    lines = [
        f"# EvalScope seven-run suite: {summary['name']}",
        "",
        f"- Model: `{summary.get('model', 'unknown')}`",
        f"- API: `{summary.get('api_url', '')}`",
        f"- reasoning_effort: `{summary.get('reasoning_effort', '')}`",
        f"- Temperature: {summary.get('temperature')}, top_p: {summary.get('top_p')}, "
        f"max_tokens: {summary.get('max_tokens')}",
        f"- Runs: {summary.get('runs_total')}",
        "",
    ]
    header = "| Run | Status | " + " | ".join(tasks) + " | Mean |"
    lines.append(header)
    lines.append("|---:|---|" + "---:|" * (len(tasks) + 1))
    for run in summary.get("runs", []):
        cells = []
        for task in tasks:
            value = (run.get("datasets") or {}).get(task)
            cells.append(f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "-")
        mean = run.get("accuracy")
        mean_text = f"{mean * 100:.2f}%" if isinstance(mean, (int, float)) else "-"
        lines.append(
            f"| {run.get('index')} | {run.get('status')} | "
            + " | ".join(cells)
            + f" | {mean_text} |"
        )
    overall = (summary.get("metrics") or {}).get("accuracy") or {}
    if overall:
        lines.extend(
            [
                "",
                f"**Mean score: {overall['mean'] * 100:.2f}% ± "
                f"{overall['std'] * 100:.2f}%** "
                f"(min {overall['min'] * 100:.2f}%, max {overall['max'] * 100:.2f}%, "
                f"n={len(overall['values'])})",
            ]
        )
    for task in tasks:
        item = (summary.get("metrics") or {}).get(task) or {}
        if item:
            lines.append(
                f"- **{task}**: {item['mean'] * 100:.2f}% ± {item['std'] * 100:.2f}% "
                f"(n={len(item['values'])})"
            )
    lines.append("")
    return "\n".join(lines)


def cmd_suite_aggregate(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    tasks = list(state.get("meta", {}).get("tasks") or [])
    records = []
    overall_values: list[float] = []
    per_task: dict[str, list[float]] = {task: [] for task in tasks}
    for run in sorted(state.get("runs", []), key=lambda item: item.get("index", 0)):
        record = {
            "index": run.get("index"),
            "tag": run.get("tag"),
            "status": run.get("status"),
            "exit_code": run.get("exit_code"),
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "log": run.get("log"),
            "command": run.get("command", ""),
            "work_dir": run.get("work_dir"),
            "datasets": run.get("datasets") or {},
            "errors": run.get("errors") or {},
            "accuracy": run.get("accuracy"),
        }
        if isinstance(record["accuracy"], (int, float)):
            overall_values.append(float(record["accuracy"]))
        for task in tasks:
            value = record["datasets"].get(task)
            if isinstance(value, (int, float)):
                per_task[task].append(float(value))
        records.append(record)

    metrics_by_name = {"accuracy": metrics(overall_values)}
    for task, values in per_task.items():
        metrics_by_name[task] = metrics(values)
    summary = {
        "schema_version": 1,
        "name": state.get("name"),
        "kind": "evalscope-suite",
        "dataset": "EvalScope",
        "tasks": tasks,
        "model": state.get("meta", {}).get("model", ""),
        "host": state.get("meta", {}).get("host", ""),
        "port": state.get("meta", {}).get("port", ""),
        "api_url": state.get("meta", {}).get("api_url", ""),
        "reasoning_effort": state.get("meta", {}).get("reasoning_effort", ""),
        "temperature": state.get("meta", {}).get("temperature"),
        "top_p": state.get("meta", {}).get("top_p"),
        "max_tokens": state.get("meta", {}).get("max_tokens"),
        "started_at": state.get("started_at"),
        "finished_at": utc_now(),
        "runs_total": state.get("runs_total", len(records)),
        "runs_recorded": len(overall_values),
        "metrics": metrics_by_name,
        "runs": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    markdown = render_markdown(summary)
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(markdown)

    all_completed = (
        len(overall_values) == state.get("runs_total")
        and all(run.get("status") == "completed" for run in records)
    )
    state["status"] = "completed" if all_completed else ("partial" if overall_values else "failed")
    state["exit_code"] = 0 if all_completed else 1
    state["finished_at"] = utc_now()
    state["message"] = "finished"
    state["runs"] = records
    state["runs_finished"] = sum(
        1 for run in records if run.get("status") in ("completed", "partial", "failed")
    )
    state["runs_succeeded"] = sum(1 for run in records if run.get("status") == "completed")
    accuracy = metrics_by_name.get("accuracy", {})
    state["result"] = {
        "accuracy_mean": accuracy.get("mean"),
        "accuracy_std": accuracy.get("std"),
        "runs_recorded": len(overall_values),
        "tasks": tasks,
    }
    total = state.get("runs_total") or len(records) or 1
    state["progress"] = {
        "expected": len(tasks),
        "runs_finished": state["runs_finished"],
        "runs_total": total,
        "percent": 100.0 if all_completed else round(state["runs_finished"] / total * 100, 1),
    }
    save_state(path, state)
    print(markdown, end="")


def cmd_suite_finish(args: argparse.Namespace) -> None:
    path = Path(args.state)
    if not path.is_file():
        return
    state = load_state(path)
    if state.get("status") in ("completed", "partial", "failed"):
        return
    state["status"] = args.status
    state["exit_code"] = args.exit_code
    state["finished_at"] = utc_now()
    state["message"] = args.message or state.get("message", "")
    save_state(path, state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EvalScope state and summary helper")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("suite-start")
    start.add_argument("--state", required=True)
    start.add_argument("--name", required=True)
    start.add_argument("--runs", type=int, required=True)
    start.add_argument("--model", required=True)
    start.add_argument("--host", default="")
    start.add_argument("--port", default="")
    start.add_argument("--api-url", default="")
    start.add_argument("--tasks", required=True)
    start.add_argument("--reasoning-effort", default="")
    start.add_argument("--temperature", type=float, default=1.0)
    start.add_argument("--top-p", type=float, default=0.95)
    start.add_argument("--max-tokens", type=int, default=131072)
    start.add_argument("--pid", type=int)
    start.add_argument("--env", default="")
    start.add_argument("--python", default="")
    start.add_argument("--command", default="")
    start.set_defaults(func=cmd_suite_start)

    run_start = sub.add_parser("suite-run-start")
    run_start.add_argument("--state", required=True)
    run_start.add_argument("--index", type=int, required=True)
    run_start.add_argument("--tag", required=True)
    run_start.add_argument("--started-at", default="")
    run_start.add_argument("--log", default="")
    run_start.add_argument("--command", default="")
    run_start.set_defaults(func=cmd_suite_run_start)

    run_finish = sub.add_parser("suite-run-finish")
    run_finish.add_argument("--state", required=True)
    run_finish.add_argument("--index", type=int, required=True)
    run_finish.add_argument("--exit-code", type=int, required=True)
    run_finish.add_argument("--work-dir", required=True)
    run_finish.add_argument("--tasks", required=True)
    run_finish.add_argument("--log", default="")
    run_finish.add_argument("--finished-at", default="")
    run_finish.set_defaults(func=cmd_suite_run_finish)

    aggregate = sub.add_parser("suite-aggregate")
    aggregate.add_argument("--state", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--markdown", required=True)
    aggregate.set_defaults(func=cmd_suite_aggregate)

    finish = sub.add_parser("suite-finish")
    finish.add_argument("--state", required=True)
    finish.add_argument("--status", required=True)
    finish.add_argument("--exit-code", type=int, required=True)
    finish.add_argument("--message", default="")
    finish.set_defaults(func=cmd_suite_finish)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
