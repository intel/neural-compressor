# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Summarize repeated evalscope runs produced by run_evalscope.sh.

Expected layout: <root>/<task>/run<i>/reports/**/<dataset>.json

With 3 or more runs, one highest and one lowest score are dropped and the
remaining ("kept") scores are averaged; otherwise all scores are averaged.
"""

import argparse
import glob
import json
import os
import re
import sys

TRIM_MIN_RUNS = 3


def load_score(run_dir):
    """Return (metric_name, score in percent) from the evalscope report of one run."""
    report_files = sorted(glob.glob(os.path.join(run_dir, "reports", "**", "*.json"), recursive=True))
    for path in report_files:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        metrics = report.get("metrics") if isinstance(report, dict) else None
        if not metrics:
            continue
        metric = metrics[0]
        primary = report.get("primary_metric_identity")
        if primary is not None:
            metric = next((m for m in metrics if m.get("identity") == primary), metric)
        name = metric.get("legacy_name") or metric.get("name")
        if not name and isinstance(metric.get("identity"), dict):
            name = metric["identity"].get("name")
        return name or "score", float(metric["score"]) * 100
    raise FileNotFoundError(f"No evalscope report with metrics found under {run_dir}/reports")


def run_index(run_dir):
    match = re.search(r"run(\d+)$", os.path.basename(os.path.normpath(run_dir)))
    return int(match.group(1)) if match else 0


def collect_task(root, task):
    run_dirs = sorted(glob.glob(os.path.join(root, task, "run*")), key=run_index)
    if not run_dirs:
        raise FileNotFoundError(f"No runs found for task '{task}' under {root}")
    metric_name = None
    scores = []
    for run_dir in run_dirs:
        metric_name, score = load_score(run_dir)
        scores.append(score)
    return metric_name, scores


def trim(scores):
    """Return indices of kept scores: drop one min and one max when there are 3 or more runs."""
    indices = list(range(len(scores)))
    if len(scores) < TRIM_MIN_RUNS:
        return indices
    lowest = min(indices, key=lambda i: scores[i])
    highest = max(reversed(indices), key=lambda i: scores[i])
    return [i for i in indices if i not in (lowest, highest)]


def mean(values):
    return sum(values) / len(values)


def fmt(value):
    return f"{value:.2f}"


def render_table(headers, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"

    out = [sep, line(headers), sep]
    out += [line(row) for row in rows[:-1]]
    if rows:
        if len(rows) > 1:
            out.append(sep)
        out.append(line(rows[-1]))
    out.append(sep)
    return "\n".join(out)


def task_table(root, task):
    metric_name, scores = collect_task(root, task)
    kept = trim(scores)
    rows = [
        [f"run{i + 1}", fmt(score), "kept" if i in kept else "dropped"] for i, score in enumerate(scores)
    ]
    rows.append([f"mean of kept ({len(kept)}/{len(scores)})", fmt(mean([scores[i] for i in kept])), ""])
    title = f"Result: {task} ({metric_name}, %)"
    return title + "\n" + render_table(["Run", "Score", "Status"], rows)


def summary_table(root, tasks):
    results = []
    for task in tasks:
        metric_name, scores = collect_task(root, task)
        kept = trim(scores)
        results.append((task, metric_name, scores, kept))

    max_kept = max(len(kept) for *_, kept in results)
    headers = ["Task", "Metric", "Runs"] + [f"Kept{i + 1}" for i in range(max_kept)] + ["Mean"]
    rows = []
    task_means = []
    for task, metric_name, scores, kept in results:
        kept_scores = [fmt(scores[i]) for i in kept]
        task_mean = mean([scores[i] for i in kept])
        task_means.append(task_mean)
        rows.append(
            [task, metric_name, f"{len(kept)}/{len(scores)}"]
            + kept_scores
            + [""] * (max_kept - len(kept))
            + [fmt(task_mean)]
        )
    rows.append(["Average (all tasks)", "", ""] + [""] * max_kept + [fmt(mean(task_means))])
    title = "Summary (scores in %, KeptN = scores kept after dropping max/min when runs >= 3)"
    return title + "\n" + render_table(headers, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Directory containing <task>/run<i> evalscope outputs.")
    parser.add_argument("--tasks", required=True, help="Comma-separated task names.")
    parser.add_argument("--summary", action="store_true", help="Print the overall summary table for all tasks.")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    try:
        if args.summary:
            print(summary_table(args.root, tasks))
        else:
            print("\n\n".join(task_table(args.root, task) for task in tasks))
    except (OSError, ValueError, KeyError) as e:
        print(f"[eval_summary] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
