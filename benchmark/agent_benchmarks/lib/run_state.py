#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_state(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"State file not found: {path}")
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid state file: {path}")
    return payload


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def parse_meta(pairs: list[str] | None) -> dict:
    meta = {}
    for pair in pairs or []:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise SystemExit(f"Invalid --meta value (expected KEY=VALUE): {pair}")
        meta[key] = value
    return meta


def read_report(path: str | None) -> dict | None:
    if not path:
        return None
    report_path = Path(path)
    if not report_path.is_file():
        return None
    try:
        payload = load_json(report_path)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def report_metrics(report: dict | None) -> dict:
    if report is None:
        return {}
    submitted = report.get("submitted_instances", report.get("evaluated_instances", 0))
    resolved = report.get("resolved_instances", 0)
    accuracy = report.get("accuracy")
    if accuracy is None:
        accuracy = resolved / submitted if submitted else 0.0
    return {
        "submitted": submitted,
        "completed": report.get("completed_instances", report.get("evaluated_instances", 0)),
        "resolved": resolved,
        "errors": report.get("error_instances", 0),
        "accuracy": accuracy,
    }


def refresh_counts(state: dict) -> None:
    runs = state.get("runs", [])
    state["runs_finished"] = sum(1 for run in runs if run.get("status") in ("completed", "failed"))
    state["runs_succeeded"] = sum(1 for run in runs if run.get("status") == "completed")


def find_run(state: dict, index: int) -> dict:
    for run in state.setdefault("runs", []):
        if run.get("index") == index:
            return run
    run = {"index": index}
    state["runs"].append(run)
    state["runs"].sort(key=lambda item: item.get("index", 0))
    return run


def state_dir_report_meta(report_path: str | None) -> dict:
    if not report_path:
        return {}
    state_path = Path(report_path).parent / "state.json"
    if not state_path.is_file():
        return {}
    try:
        payload = load_json(state_path)
    except (OSError, json.JSONDecodeError):
        return {}
    meta = payload.get("meta")
    return meta if isinstance(meta, dict) else {}


def duration_seconds(started_at: str | None, finished_at: str | None) -> int | None:
    if not started_at or not finished_at:
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0, int((end - start).total_seconds()))


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def cmd_start(args: argparse.Namespace) -> None:
    path = Path(args.state)
    meta = parse_meta(args.meta)
    if args.preserve and path.is_file():
        state = load_state(path)
        state.update(
            {
                "status": "running",
                "pid": args.pid,
                "finished_at": None,
                "runs_total": args.runs_total,
                "current_run": args.current_run,
                "message": args.message or "",
            }
        )
        state.setdefault("runs", [])
        state["meta"] = {**state.get("meta", {}), **meta}
    else:
        state = {
            "schema_version": 1,
            "name": args.name,
            "kind": args.kind,
            "status": "running",
            "pid": args.pid,
            "started_at": args.started_at or utc_now(),
            "updated_at": utc_now(),
            "finished_at": None,
            "runs_total": args.runs_total,
            "current_run": args.current_run,
            "runs_finished": 0,
            "runs_succeeded": 0,
            "exit_code": None,
            "message": args.message or "",
            "meta": meta,
            "runs": [],
        }
    refresh_counts(state)
    save_state(path, state)


def cmd_meta(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    state["meta"] = {**state.get("meta", {}), **parse_meta(args.meta)}
    if args.message is not None:
        state["message"] = args.message
    save_state(path, state)


def cmd_run_start(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    run = find_run(state, args.index)
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
    state["finished_at"] = None
    state["message"] = f"run {args.index}/{state.get('runs_total', args.index)}"
    refresh_counts(state)
    save_state(path, state)


def cmd_run_finish(args: argparse.Namespace) -> None:
    path = Path(args.state)
    state = load_state(path)
    run = find_run(state, args.index)
    report = read_report(args.report)
    metrics = report_metrics(report)
    run.update(
        {
            "status": "completed" if args.exit_code == 0 else "failed",
            "exit_code": args.exit_code,
            "finished_at": args.finished_at or utc_now(),
            "report": args.report,
            "log": args.log,
            "duration_seconds": duration_seconds(run.get("started_at"), args.finished_at or utc_now()),
        }
    )
    run.update(metrics)
    run_meta = state_dir_report_meta(args.report)
    if run_meta:
        run["meta"] = run_meta
        if run_meta.get("model") and not state.get("meta", {}).get("model"):
            state.setdefault("meta", {})["model"] = run_meta["model"]
    refresh_counts(state)
    save_state(path, state)


def cmd_finish(args: argparse.Namespace) -> None:
    path = Path(args.state)
    if not path.is_file():
        return
    state = load_state(path)
    state["status"] = args.status
    state["exit_code"] = args.exit_code
    state["finished_at"] = args.finished_at or utc_now()
    if args.message is not None:
        state["message"] = args.message
    report = read_report(args.report)
    if report is not None:
        state["result"] = report_metrics(report)
    refresh_counts(state)
    save_state(path, state)


def aggregate_records(state: dict) -> tuple[list[dict], dict, list[dict]]:
    records: list[dict] = []
    per_instance: dict[str, dict[str, int]] = {}
    runs = sorted(state.get("runs", []), key=lambda item: item.get("index", 0))
    for run in runs:
        report = read_report(run.get("report"))
        metrics = report_metrics(report)
        if report is not None:
            evaluated = set(report.get("submitted_ids") or [])
            if not evaluated:
                for key in (
                    "resolved_ids",
                    "unresolved_ids",
                    "error_ids",
                    "completed_ids",
                    "incomplete_ids",
                    "empty_patch_ids",
                ):
                    evaluated.update(report.get(key, []))
            resolved = set(report.get("resolved_ids") or [])
            for instance_id in evaluated:
                entry = per_instance.setdefault(instance_id, {"resolved": 0, "evaluated": 0})
                entry["evaluated"] += 1
                entry["resolved"] += int(instance_id in resolved)
        record = {
            "index": run.get("index"),
            "tag": run.get("tag"),
            "status": run.get("status"),
            "exit_code": run.get("exit_code"),
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "duration_seconds": run.get("duration_seconds")
            or duration_seconds(run.get("started_at"), run.get("finished_at")),
            "report": run.get("report"),
            "log": run.get("log"),
            "command": run.get("command", ""),
        }
        record.update(metrics)
        records.append(record)

    metrics_by_name = {}
    for name in ("accuracy", "resolved", "submitted", "completed", "errors"):
        values = [float(record[name]) for record in records if name in record]
        if values:
            metrics_by_name[name] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "values": values,
            }

    instances = [
        {
            "instance_id": instance_id,
            "resolved": entry["resolved"],
            "evaluated": entry["evaluated"],
            "rate": entry["resolved"] / entry["evaluated"] if entry["evaluated"] else 0.0,
        }
        for instance_id, entry in sorted(per_instance.items())
    ]
    return records, metrics_by_name, instances


def render_markdown(summary: dict) -> str:
    lines = [
        f"# Suite: {summary.get('name', '')}",
        "",
        f"- Kind: {summary.get('kind', '')}",
        f"- Model: {summary.get('model', 'unknown')}",
        f"- Dataset: {summary.get('dataset', 'unknown')}",
        f"- Runs: {summary.get('runs_total', 0)} requested, "
        f"{summary.get('runs_recorded', 0)} with results, "
        f"{summary.get('failed_runs', 0)} failed",
        "",
        "| Run | Status | Submitted | Resolved | Accuracy | Duration | Started | Finished |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]
    for record in summary.get("runs", []):
        accuracy = record.get("accuracy")
        accuracy_text = f"{accuracy * 100:.2f}%" if accuracy is not None else "-"
        lines.append(
            "| {index} | {status} | {submitted} | {resolved} | {accuracy} | {duration} | {started} | {finished} |".format(
                index=record.get("index", ""),
                status=record.get("status", ""),
                submitted=record.get("submitted", "-"),
                resolved=record.get("resolved", "-"),
                accuracy=accuracy_text,
                duration=format_duration(record.get("duration_seconds")),
                started=record.get("started_at", "-"),
                finished=record.get("finished_at", "-"),
            )
        )

    metrics = summary.get("metrics", {})
    accuracy = metrics.get("accuracy")
    if accuracy:
        lines.extend(
            [
                "",
                f"**Mean accuracy: {accuracy['mean'] * 100:.2f}% ± {accuracy['std'] * 100:.2f}%** "
                f"(min {accuracy['min'] * 100:.2f}%, max {accuracy['max'] * 100:.2f}%, "
                f"n={len(accuracy['values'])})",
            ]
        )
    resolved = metrics.get("resolved")
    if resolved:
        lines.append(
            f"**Mean resolved: {resolved['mean']:.2f} ± {resolved['std']:.2f}** "
            f"(min {resolved['min']:.0f}, max {resolved['max']:.0f})"
        )
    lines.append("")
    return "\n".join(lines)


def cmd_aggregate(args: argparse.Namespace) -> None:
    state = load_state(Path(args.state))
    records, metrics, instances = aggregate_records(state)
    summary = {
        "schema_version": 1,
        "name": state.get("name"),
        "kind": state.get("kind"),
        "model": state.get("meta", {}).get("model", ""),
        "dataset": state.get("meta", {}).get("dataset", ""),
        "started_at": state.get("started_at"),
        "finished_at": utc_now(),
        "runs_total": state.get("runs_total", len(records)),
        "runs_recorded": sum(1 for record in records if "accuracy" in record),
        "failed_runs": sum(1 for record in records if record.get("status") == "failed"),
        "metrics": metrics,
        "instances": instances,
        "runs": records,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown = render_markdown(summary)
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown)
    accuracy = metrics.get("accuracy")
    if accuracy:
        state["result"] = {
            "accuracy_mean": accuracy["mean"],
            "accuracy_std": accuracy["std"],
            "resolved_mean": metrics.get("resolved", {}).get("mean", 0.0),
            "submitted_mean": metrics.get("submitted", {}).get("mean", 0.0),
            "runs_recorded": summary["runs_recorded"],
        }
    save_state(Path(args.state), state)
    print(markdown, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track benchmark run state and aggregate repeated runs")
    commands = parser.add_subparsers(required=True)

    command = commands.add_parser("start")
    for name in ("state", "name", "kind"):
        command.add_argument(f"--{name}", required=True)
    command.add_argument("--runs-total", type=int, required=True)
    command.add_argument("--current-run", type=int, default=1)
    command.add_argument("--pid", type=int)
    command.add_argument("--started-at")
    command.add_argument("--message")
    command.add_argument("--meta", action="append")
    command.add_argument("--preserve", action="store_true")
    command.set_defaults(func=cmd_start)

    command = commands.add_parser("meta")
    command.add_argument("--state", required=True)
    command.add_argument("--meta", action="append")
    command.add_argument("--message")
    command.set_defaults(func=cmd_meta)

    command = commands.add_parser("run-start")
    for name in ("state", "tag"):
        command.add_argument(f"--{name}", required=True)
    command.add_argument("--index", type=int, required=True)
    command.add_argument("--started-at")
    command.add_argument("--log")
    command.add_argument("--command", default="")
    command.set_defaults(func=cmd_run_start)

    command = commands.add_parser("run-finish")
    command.add_argument("--state", required=True)
    command.add_argument("--index", type=int, required=True)
    command.add_argument("--exit-code", type=int, required=True)
    command.add_argument("--report")
    command.add_argument("--log")
    command.add_argument("--finished-at")
    command.set_defaults(func=cmd_run_finish)

    command = commands.add_parser("finish")
    command.add_argument("--state", required=True)
    command.add_argument("--status", required=True)
    command.add_argument("--exit-code", type=int, default=0)
    command.add_argument("--report")
    command.add_argument("--finished-at")
    command.add_argument("--message")
    command.set_defaults(func=cmd_finish)

    command = commands.add_parser("aggregate")
    command.add_argument("--state", required=True)
    command.add_argument("--output")
    command.add_argument("--markdown")
    command.set_defaults(func=cmd_aggregate)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
