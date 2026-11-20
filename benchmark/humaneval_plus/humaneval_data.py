#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


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


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_official(log_text: str) -> dict:
    text = strip_ansi(log_text)
    official = {}
    for marker, key in (
        ("humaneval+ (base + extra tests)", "plus"),
        ("humaneval (base tests)", "base"),
    ):
        if marker not in text:
            continue
        segment = text.split(marker, 1)[1][:500]
        for pattern, name in ((r"pass@1:\s*([0-9.]+)", "pass_at_1"), (r"pass@10:\s*([0-9.]+)", "pass_at_10")):
            match = re.search(pattern, segment)
            if match:
                official[f"{name}_{key}"] = float(match.group(1))
    return official


def pass_at_k(n: int, correct: int, k: int) -> float:
    if n <= 0:
        return 0.0
    if n - correct < k:
        return 1.0
    probability = 1.0
    for index in range(k):
        probability *= (n - correct - index) / (n - index)
    return 1.0 - probability


def summarize(
    result_payload: dict,
    log_text: str,
    runs_total: int,
    expected_tasks: int,
    name: str,
    state: dict,
) -> dict:
    eval_map = result_payload.get("eval", {})
    tasks: dict[str, list[dict]] = {}
    for task_id, samples in eval_map.items():
        rows = []
        for sample in samples:
            base = sample.get("base_status") == "pass"
            plus = base and sample.get("plus_status") == "pass"
            rows.append({"base": base, "plus": plus})
        tasks[task_id] = rows

    expected = expected_tasks or len(tasks)
    max_samples = max((len(rows) for rows in tasks.values()), default=0)
    sample_runs = runs_total or max_samples
    runs = []
    for index in range(sample_runs):
        evaluated = base_passed = plus_passed = 0
        for rows in tasks.values():
            if index < len(rows):
                evaluated += 1
                base_passed += int(rows[index]["base"])
                plus_passed += int(rows[index]["plus"])
        accuracy = plus_passed / evaluated if evaluated else None
        base_accuracy = base_passed / evaluated if evaluated else None
        status = "completed" if evaluated == expected else ("partial" if evaluated else "failed")
        runs.append(
            {
                "index": index + 1,
                "tag": f"sample_{index + 1:02d}",
                "status": status,
                "evaluated": evaluated,
                "passed": plus_passed,
                "base_passed": base_passed,
                "accuracy": accuracy,
                "base_accuracy": base_accuracy,
                "result": (
                    {
                        "submitted": evaluated,
                        "resolved": plus_passed,
                        "accuracy": accuracy,
                        "base_accuracy": base_accuracy,
                    }
                    if accuracy is not None
                    else {}
                ),
                "started_at": state.get("started_at"),
                "finished_at": utc_now(),
                "duration_seconds": None,
            }
        )

    completed_runs = [run for run in runs if run["status"] == "completed"]
    accuracy_values = [run["accuracy"] for run in completed_runs if run["accuracy"] is not None]
    base_values = [run["base_accuracy"] for run in completed_runs if run["base_accuracy"] is not None]
    recorded_values = [run["accuracy"] for run in runs if run["accuracy"] is not None]
    total_evaluated = sum(run["evaluated"] for run in runs)
    total_passed = sum(run["passed"] for run in runs)
    total_base_passed = sum(run["base_passed"] for run in runs)

    instances = []
    for task_id, rows in sorted(tasks.items()):
        evaluated = len(rows)
        passed = sum(1 for row in rows if row["plus"])
        base_passed = sum(1 for row in rows if row["base"])
        instances.append(
            {
                "task_id": task_id,
                "evaluated": evaluated,
                "passed": passed,
                "base_passed": base_passed,
                "rate": passed / evaluated if evaluated else 0.0,
                "base_rate": base_passed / evaluated if evaluated else 0.0,
            }
        )
    flaky = [item for item in instances if 0 < item["rate"] < 1]
    always_failed = [item["task_id"] for item in instances if item["rate"] == 0]
    pass_at_7 = None
    if instances and all(item["evaluated"] >= 7 for item in instances):
        pass_at_7 = statistics.mean([pass_at_k(item["evaluated"], item["passed"], 7) for item in instances])

    status = "failed"
    if recorded_values:
        status = "completed" if all(run["status"] == "completed" for run in runs) else "partial"

    summary = {
        "schema_version": 1,
        "name": name,
        "kind": "humaneval-plus",
        "dataset": "HumanEval+",
        "model": state.get("meta", {}).get("model", ""),
        "temperature": state.get("meta", {}).get("temperature"),
        "started_at": state.get("started_at"),
        "finished_at": utc_now(),
        "runs_total": sample_runs,
        "runs_recorded": len(recorded_values),
        "metrics": {
            "accuracy": metrics(accuracy_values),
            "base_accuracy": metrics(base_values),
        },
        "overall": {
            "evaluated": total_evaluated,
            "passed": total_passed,
            "base_passed": total_base_passed,
            "pass_at_1": total_passed / total_evaluated if total_evaluated else 0.0,
            "base_pass_at_1": total_base_passed / total_evaluated if total_evaluated else 0.0,
            "pass_at_7": pass_at_7,
        },
        "official": parse_official(log_text),
        "instances": instances,
        "flaky": flaky,
        "always_failed": always_failed,
        "runs": runs,
        "paths": {
            "samples": state.get("meta", {}).get("samples", ""),
            "result": state.get("meta", {}).get("result", ""),
            "log": state.get("meta", {}).get("log", ""),
        },
    }
    return summary


def render_markdown(summary: dict) -> str:
    accuracy = summary["metrics"].get("accuracy", {})
    base = summary["metrics"].get("base_accuracy", {})
    lines = [
        f"# HumanEval+ suite: {summary['name']}",
        "",
        f"- Model: {summary.get('model', 'unknown')}",
        f"- Temperature: {summary.get('temperature')}",
        f"- Samples per problem: {summary['runs_total']}",
        f"- Tasks: {len(summary['instances'])}",
        "",
        "| Sample | Status | Evaluated | Passed | pass@1 | base pass@1 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        accuracy_text = f"{run['accuracy'] * 100:.2f}%" if run["accuracy"] is not None else "-"
        base_text = f"{run['base_accuracy'] * 100:.2f}%" if run["base_accuracy"] is not None else "-"
        lines.append(
            f"| {run['index']} | {run['status']} | {run['evaluated']} | {run['passed']} | "
            f"{accuracy_text} | {base_text} |"
        )
    if accuracy:
        lines.extend(
            [
                "",
                f"**Mean pass@1: {accuracy['mean'] * 100:.2f}% ± {accuracy['std'] * 100:.2f}%** "
                f"(min {accuracy['min'] * 100:.2f}%, max {accuracy['max'] * 100:.2f}%, "
                f"n={len(accuracy['values'])})",
            ]
        )
    if base:
        lines.append(f"**Mean base pass@1: {base['mean'] * 100:.2f}% ± {base['std'] * 100:.2f}%**")
    tokens = summary["metrics"].get("tokens_generated")
    if tokens:
        lines.append(
            f"**Generated tokens: {tokens['sum']:,} total; "
            f"{tokens['mean']:,.0f} mean/run** (vLLM generation_tokens delta)"
        )
    overall = summary["overall"]
    lines.append(
        f"**Overall pass@1: {overall['pass_at_1'] * 100:.2f}%** " f"({overall['passed']}/{overall['evaluated']})"
    )
    if overall.get("pass_at_7") is not None:
        lines.append(f"**pass@7: {overall['pass_at_7'] * 100:.2f}%**")
    official = summary.get("official", {})
    if "pass_at_1_plus" in official:
        lines.append(f"- Official EvalPlus pass@1 (base + extra): {official['pass_at_1_plus'] * 100:.2f}%")
    if "pass_at_1_base" in official:
        lines.append(f"- Official EvalPlus base pass@1: {official['pass_at_1_base'] * 100:.2f}%")
    lines.append(f"- Flaky tasks: {len(summary['flaky'])}; always failed: {len(summary['always_failed'])}")
    if summary["flaky"]:
        preview = ", ".join(
            f"{item['task_id']} ({item['passed']}/{item['evaluated']})"
            for item in sorted(summary["flaky"], key=lambda item: item["rate"])[:10]
        )
        lines.append(f"- Flaky preview: {preview}")
    lines.append("")
    return "\n".join(lines)


def cmd_start(args: argparse.Namespace) -> None:
    state = {
        "schema_version": 1,
        "name": args.name,
        "kind": "humaneval-plus",
        "status": "running",
        "pid": args.pid,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "runs_total": args.runs,
        "current_run": args.runs,
        "runs_finished": 0,
        "runs_succeeded": 0,
        "exit_code": None,
        "message": "starting",
        "meta": {
            "dataset": "HumanEval+",
            "model": args.model,
            "host": args.host,
            "port": str(args.port),
            "temperature": args.temperature,
            "samples": args.samples,
            "result": args.result,
            "log": args.log,
            "command": args.command or "",
        },
        "progress": {"expected": args.expected, "generated": 0, "percent": 0.0},
        "result": None,
        "runs": [
            {
                "index": index,
                "tag": f"sample_{index:02d}",
                "status": "pending",
                "evaluated": None,
                "passed": None,
                "accuracy": None,
                "result": {},
            }
            for index in range(1, args.runs + 1)
        ],
    }
    save_state(Path(args.state), state)


def cmd_progress(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    expected = state.get("progress", {}).get("expected", 0)
    state["progress"] = {
        "expected": expected,
        "generated": args.generated,
        "percent": round(args.generated / expected * 100, 1) if expected else 0.0,
    }
    if args.message is not None:
        state["message"] = args.message
    save_state(path, state)


def cmd_fail(args: argparse.Namespace) -> None:
    path = Path(args.state)
    if not path.is_file():
        return
    state = load_state(path)
    state["status"] = "interrupted" if args.exit_code in (130, 143) else "failed"
    state["exit_code"] = args.exit_code
    state["finished_at"] = utc_now()
    state["message"] = args.message or state.get("message", "")
    save_state(path, state)


def cmd_summarize(args: argparse.Namespace) -> None:
    state_path = Path(args.state)
    state = load_state(state_path)
    result_payload = load_json(Path(args.result))
    log_text = Path(args.log).read_text(errors="replace") if Path(args.log).is_file() else ""
    summary = summarize(
        result_payload,
        log_text,
        args.runs,
        args.expected,
        state.get("name", state_path.parent.name),
        state,
    )
    output_path = Path(args.output)
    save_json(output_path, summary)
    markdown = render_markdown(summary)
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown)

    accuracy = summary["metrics"].get("accuracy", {})
    state["status"] = (
        "completed"
        if summary["runs_recorded"] == summary["runs_total"]
        and all(run["status"] == "completed" for run in summary["runs"])
        else ("partial" if summary["runs_recorded"] else "failed")
    )
    state["exit_code"] = 0 if state["status"] == "completed" else 1
    state["finished_at"] = utc_now()
    state["message"] = "finished"
    state["runs"] = summary["runs"]
    state["runs_finished"] = summary["runs_recorded"]
    state["runs_succeeded"] = sum(1 for run in summary["runs"] if run["status"] == "completed")
    state["result"] = {
        "accuracy_mean": accuracy.get("mean"),
        "accuracy_std": accuracy.get("std"),
        "runs_recorded": summary["runs_recorded"],
        "pass_at_1": summary["overall"]["pass_at_1"],
        "pass_at_7": summary["overall"].get("pass_at_7"),
    }
    state["progress"] = {
        "expected": args.expected or len(summary["instances"]),
        "generated": args.expected or len(summary["instances"]),
        "percent": 100.0,
    }
    save_state(state_path, state)
    if state["exit_code"]:
        raise SystemExit(state["exit_code"])
    print(markdown, end="")


def cmd_count(args: argparse.Namespace) -> None:
    samples = Path(args.samples)
    if not samples.is_file():
        print(0)
        return
    task_ids = set()
    try:
        with samples.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    task_ids.add(json.loads(line)["task_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    except OSError:
        print(0)
        return
    print(len(task_ids))


def load_summary(path: str | None) -> dict | None:
    if not path:
        return None
    summary_path = Path(path)
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def cmd_suite_start(args: argparse.Namespace) -> None:
    state = {
        "schema_version": 1,
        "name": args.name,
        "kind": "humaneval-plus",
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
            "dataset": "HumanEval+",
            "model": args.model,
            "host": args.host,
            "port": str(args.port),
            "temperature": args.temperature,
            "mode": "independent",
            "runs_dir": args.runs_dir,
            "command": args.command or "",
        },
        "progress": {
            "expected": args.expected,
            "generated": 0,
            "percent": 0.0,
            "runs_finished": 0,
            "runs_total": args.runs,
        },
        "result": None,
        "runs": [
            {
                "index": index,
                "tag": f"{args.name}_run{index:02d}",
                "status": "pending",
                "evaluated": None,
                "passed": None,
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
    summary = load_summary(args.summary)
    metrics = {}
    if summary:
        sample = (summary.get("runs") or [{}])[0]
        overall = summary.get("overall") or {}
        evaluated = sample.get("evaluated", overall.get("evaluated"))
        passed = sample.get("passed", overall.get("passed"))
        accuracy = sample.get("accuracy", overall.get("pass_at_1"))
        base_passed = sample.get("base_passed")
        base_accuracy = sample.get("base_accuracy")
        if accuracy is not None:
            metrics = {
                "evaluated": evaluated,
                "passed": passed,
                "accuracy": accuracy,
                "base_passed": base_passed,
                "base_accuracy": base_accuracy,
            }
    status = "completed" if args.exit_code == 0 and metrics else "failed"
    run.update(
        {
            "status": status,
            "exit_code": args.exit_code,
            "finished_at": args.finished_at or utc_now(),
            "summary": args.summary,
            "log": args.log,
            **metrics,
        }
    )
    if getattr(args, "tokens_generated", None) is not None:
        run["tokens_generated"] = args.tokens_generated
    if metrics:
        run["result"] = {
            "submitted": metrics.get("evaluated"),
            "resolved": metrics.get("passed"),
            "accuracy": metrics.get("accuracy"),
        }
    runs = state.get("runs", [])
    state["runs_finished"] = sum(1 for item in runs if item.get("status") in ("completed", "failed"))
    state["runs_succeeded"] = sum(1 for item in runs if item.get("status") == "completed")
    state["progress"] = {
        "expected": state.get("progress", {}).get("expected", 164),
        "generated": state.get("progress", {}).get("generated", 0),
        "percent": round(state["runs_finished"] / state["runs_total"] * 100, 1) if state.get("runs_total") else 0.0,
        "runs_finished": state["runs_finished"],
        "runs_total": state.get("runs_total"),
    }
    state["message"] = f"run {args.index}/{state.get('runs_total', args.index)} finished"
    save_state(path, state)


def render_independent_markdown(summary: dict) -> str:
    accuracy = summary["metrics"].get("accuracy", {})
    base = summary["metrics"].get("base_accuracy", {})
    lines = [
        f"# HumanEval+ independent runs: {summary['name']}",
        "",
        f"- Model: {summary.get('model', 'unknown')}",
        f"- Temperature: {summary.get('temperature')}",
        f"- Independent runs: {summary['runs_total']}",
        f"- Tasks per run: {len(summary['instances'])}",
        "",
        "| Run | Status | Evaluated | Passed | pass@1 | base pass@1 | Generated tokens |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        accuracy_text = f"{run['accuracy'] * 100:.2f}%" if run.get("accuracy") is not None else "-"
        base_text = f"{run['base_accuracy'] * 100:.2f}%" if run.get("base_accuracy") is not None else "-"
        tokens_text = (
            f"{int(run['tokens_generated']):,}" if isinstance(run.get("tokens_generated"), (int, float)) else "-"
        )
        lines.append(
            f"| {run.get('index')} | {run.get('status')} | {run.get('evaluated', '-')} | "
            f"{run.get('passed', '-')} | {accuracy_text} | {base_text} | {tokens_text} |"
        )
    if accuracy:
        lines.extend(
            [
                "",
                f"**Mean pass@1: {accuracy['mean'] * 100:.2f}% ± {accuracy['std'] * 100:.2f}%** "
                f"(min {accuracy['min'] * 100:.2f}%, max {accuracy['max'] * 100:.2f}%, "
                f"n={len(accuracy['values'])})",
            ]
        )
    if base:
        lines.append(f"**Mean base pass@1: {base['mean'] * 100:.2f}% ± {base['std'] * 100:.2f}%**")
    tokens = summary["metrics"].get("tokens_generated")
    if tokens:
        lines.append(
            f"**Generated tokens: {tokens['sum']:,} total; "
            f"{tokens['mean']:,.0f} mean/run** (vLLM generation_tokens delta)"
        )
    overall = summary["overall"]
    lines.append(
        f"**Overall pass@1: {overall['pass_at_1'] * 100:.2f}%** " f"({overall['passed']}/{overall['evaluated']})"
    )
    if overall.get("pass_at_7") is not None:
        lines.append(f"**pass@7: {overall['pass_at_7'] * 100:.2f}%**")
    lines.append(f"- Flaky tasks: {len(summary['flaky'])}; always failed: {len(summary['always_failed'])}")
    lines.append("")
    return "\n".join(lines)


def cmd_suite_aggregate(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    records = []
    per_task: dict[str, dict[str, int]] = {}
    run_accuracies = []
    base_accuracies = []
    token_values = []
    for run in sorted(state.get("runs", []), key=lambda item: item.get("index", 0)):
        summary = load_summary(run.get("summary")) if run.get("status") == "completed" else None
        record = {
            "index": run.get("index"),
            "tag": run.get("tag"),
            "status": run.get("status"),
            "exit_code": run.get("exit_code"),
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "summary": run.get("summary"),
            "log": run.get("log"),
            "command": run.get("command", ""),
            "tokens_generated": run.get("tokens_generated"),
        }
        if summary:
            sample = (summary.get("runs") or [{}])[0]
            overall = summary.get("overall") or {}
            accuracy = sample.get("accuracy", overall.get("pass_at_1"))
            base_accuracy = sample.get("base_accuracy", overall.get("base_pass_at_1"))
            if accuracy is not None:
                run_accuracies.append(float(accuracy))
                record.update(
                    {
                        "evaluated": sample.get("evaluated", overall.get("evaluated")),
                        "passed": sample.get("passed", overall.get("passed")),
                        "base_passed": sample.get("base_passed", overall.get("base_passed")),
                        "accuracy": float(accuracy),
                        "base_accuracy": float(base_accuracy) if base_accuracy is not None else None,
                        "result": {
                            "submitted": sample.get("evaluated", overall.get("evaluated")),
                            "resolved": sample.get("passed", overall.get("passed")),
                            "accuracy": float(accuracy),
                        },
                    }
                )
            if base_accuracy is not None:
                base_accuracies.append(float(base_accuracy))
            for instance in summary.get("instances", []):
                entry = per_task.setdefault(instance["task_id"], {"passed": 0, "evaluated": 0, "base_passed": 0})
                entry["passed"] += int(instance.get("passed", 0))
                entry["evaluated"] += int(instance.get("evaluated", 0))
                entry["base_passed"] += int(instance.get("base_passed", 0))
        records.append(record)
        if isinstance(run.get("tokens_generated"), (int, float)):
            token_values.append(float(run["tokens_generated"]))

    metrics_by_name = {
        "accuracy": metrics(run_accuracies),
        "base_accuracy": metrics(base_accuracies),
    }
    if token_values:
        metrics_by_name["tokens_generated"] = {
            **metrics(token_values),
            "sum": int(sum(token_values)),
        }
    instances = []
    for task_id, entry in sorted(per_task.items()):
        instances.append(
            {
                "task_id": task_id,
                "resolved": entry["passed"],
                "base_passed": entry["base_passed"],
                "evaluated": entry["evaluated"],
                "rate": entry["passed"] / entry["evaluated"] if entry["evaluated"] else 0.0,
                "base_rate": entry["base_passed"] / entry["evaluated"] if entry["evaluated"] else 0.0,
            }
        )
    flaky = [item for item in instances if 0 < item["rate"] < 1]
    always_failed = [item["task_id"] for item in instances if item["rate"] == 0]
    total_evaluated = sum(item["evaluated"] for item in instances)
    total_passed = sum(item["resolved"] for item in instances)
    total_base_passed = sum(item["base_passed"] for item in instances)
    pass_at_7 = None
    if instances and all(item["evaluated"] >= 7 for item in instances):
        pass_at_7 = statistics.mean([pass_at_k(item["evaluated"], item["resolved"], 7) for item in instances])
    summary = {
        "schema_version": 1,
        "name": state.get("name"),
        "kind": "humaneval-plus",
        "dataset": "HumanEval+",
        "model": state.get("meta", {}).get("model", ""),
        "temperature": state.get("meta", {}).get("temperature"),
        "mode": "independent",
        "runs_total": state.get("runs_total", len(records)),
        "runs_recorded": len(run_accuracies),
        "started_at": state.get("started_at"),
        "finished_at": utc_now(),
        "metrics": metrics_by_name,
        "overall": {
            "evaluated": total_evaluated,
            "passed": total_passed,
            "base_passed": total_base_passed,
            "pass_at_1": total_passed / total_evaluated if total_evaluated else 0.0,
            "base_pass_at_1": total_base_passed / total_evaluated if total_evaluated else 0.0,
            "pass_at_7": pass_at_7,
        },
        "instances": instances,
        "flaky": flaky,
        "always_failed": always_failed,
        "runs": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown = render_independent_markdown(summary)
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown)

    all_completed = len(run_accuracies) == state.get("runs_total") and all(
        run.get("status") == "completed" for run in records
    )
    state["status"] = "completed" if all_completed else ("partial" if run_accuracies else "failed")
    state["exit_code"] = 0 if all_completed else 1
    state["finished_at"] = utc_now()
    state["message"] = "finished"
    state["runs"] = records
    state["runs_finished"] = sum(1 for run in records if run.get("status") in ("completed", "failed"))
    state["runs_succeeded"] = sum(1 for run in records if run.get("status") == "completed")
    accuracy = metrics_by_name.get("accuracy", {})
    state["result"] = {
        "accuracy_mean": accuracy.get("mean"),
        "accuracy_std": accuracy.get("std"),
        "runs_recorded": len(run_accuracies),
        "pass_at_1": summary["overall"]["pass_at_1"],
        "pass_at_7": pass_at_7,
    }
    state["progress"] = {
        "expected": args.expected,
        "generated": args.expected,
        "percent": 100.0 if all_completed else round(state["runs_finished"] / state["runs_total"] * 100, 1),
        "runs_finished": state["runs_finished"],
        "runs_total": state.get("runs_total"),
    }
    save_state(path, state)
    print(markdown, end="")


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
    parser = argparse.ArgumentParser(description="HumanEval+ state and summary helper")
    commands = parser.add_subparsers(required=True)

    command = commands.add_parser("start")
    for name in ("state", "name", "model", "host", "samples", "result", "log"):
        command.add_argument(f"--{name}", required=True)
    command.add_argument("--pid", type=int, required=True)
    command.add_argument("--port", type=int, required=True)
    command.add_argument("--runs", type=int, required=True)
    command.add_argument("--temperature", type=float, required=True)
    command.add_argument("--expected", type=int, default=164)
    command.add_argument("--command")
    command.set_defaults(func=cmd_start)

    command = commands.add_parser("progress")
    command.add_argument("--state", required=True)
    command.add_argument("--generated", type=int, required=True)
    command.add_argument("--message")
    command.set_defaults(func=cmd_progress)

    command = commands.add_parser("fail")
    command.add_argument("--state", required=True)
    command.add_argument("--exit-code", type=int, default=1)
    command.add_argument("--message")
    command.set_defaults(func=cmd_fail)

    command = commands.add_parser("summarize")
    command.add_argument("--state", required=True)
    command.add_argument("--result", required=True)
    command.add_argument("--log", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--markdown", required=True)
    command.add_argument("--runs", type=int, required=True)
    command.add_argument("--expected", type=int, default=164)
    command.set_defaults(func=cmd_summarize)

    command = commands.add_parser("suite-start")
    for name in ("state", "name", "model", "host", "runs-dir"):
        command.add_argument(f"--{name}", required=True)
    command.add_argument("--pid", type=int, required=True)
    command.add_argument("--port", type=int, required=True)
    command.add_argument("--runs", type=int, required=True)
    command.add_argument("--temperature", type=float, required=True)
    command.add_argument("--expected", type=int, default=164)
    command.add_argument("--command")
    command.set_defaults(func=cmd_suite_start)

    command = commands.add_parser("suite-run-start")
    command.add_argument("--state", required=True)
    command.add_argument("--index", type=int, required=True)
    command.add_argument("--tag", required=True)
    command.add_argument("--started-at")
    command.add_argument("--log")
    command.add_argument("--command", default="")
    command.set_defaults(func=cmd_suite_run_start)

    command = commands.add_parser("suite-run-finish")
    command.add_argument("--state", required=True)
    command.add_argument("--index", type=int, required=True)
    command.add_argument("--exit-code", type=int, required=True)
    command.add_argument("--summary")
    command.add_argument("--log")
    command.add_argument("--finished-at")
    command.add_argument("--tokens-generated", type=int)
    command.set_defaults(func=cmd_suite_run_finish)

    command = commands.add_parser("suite-aggregate")
    command.add_argument("--state", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--markdown", required=True)
    command.add_argument("--expected", type=int, default=164)
    command.set_defaults(func=cmd_suite_aggregate)

    command = commands.add_parser("suite-fail")
    command.add_argument("--state", required=True)
    command.add_argument("--status", default="interrupted")
    command.add_argument("--exit-code", type=int, default=1)
    command.add_argument("--message")
    command.set_defaults(func=cmd_suite_fail)

    command = commands.add_parser("count")
    command.add_argument("--samples", required=True)
    command.set_defaults(func=cmd_count)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
