"""Official VBench computations with explicit, task-specific input selection."""

import importlib
import importlib.metadata
import math
import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

from benchmark import METRICS, _lock, atomic_json, fingerprint, load_json, read_prepared, sha256, verify


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def evaluation_view(root, jobs):
    """Standard prompt-index names; preserve real duplicates, never fake samples."""
    folder = root / "evaluation" / "videos"
    folder.mkdir(parents=True, exist_ok=True)
    counts, prompts = {}, {}
    for job in jobs:
        prompt = job["prompt"]
        number = counts.get(prompt, 0)
        counts[prompt] = number + 1
        name = f"{prompt}-{number}.mp4"
        if "/" in name or "\0" in name or len(name.encode()) > 255:
            raise ValueError(f"Unsafe official evaluation filename: {name!r}")
        target = folder / name
        source = root / "videos" / job["filename"]
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"Conflicting evaluation link: {target}")
        elif target.exists():
            raise RuntimeError(f"Refusing existing non-link: {target}")
        else:
            target.symlink_to(source)
        prompts[str(target)] = prompt
    if {str(p) for p in folder.iterdir()} != set(prompts):
        raise RuntimeError("Unexpected files in evaluation video view")
    return folder, prompts


def evaluate(root, task, metrics):
    root = Path(root).resolve()
    with _lock(root / "locks" / "evaluation.lock"):
        out = root / "evaluation"
        # Publish a new summary only after the requested metrics succeed.
        (out / "completed.json").unlink(missing_ok=True)
        (out / "summary.json").unlink(missing_ok=True)
        manifest = verify(root, task)
        return _evaluate(root, task, metrics, manifest)


def _evaluate(root, task, metrics, manifest):
    import torch
    import vbench

    task_record, jobs = read_prepared(root, task)
    cfg = load_json(root / "configuration.json")
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    selected_metrics = list(metrics or METRICS[task])
    if not selected_metrics or len(set(selected_metrics)) != len(selected_metrics):
        raise ValueError("Select distinct metrics")
    if set(selected_metrics) - set(METRICS[task]):
        raise ValueError(f"Invalid {task} metrics: {selected_metrics}")
    evidence = dict(
        task=task,
        task_fingerprint=fingerprint(task_record),
        configuration_fingerprint=fingerprint(cfg),
        video_hashes={video["filename"]: video["video_sha256"] for video in manifest["videos"]},
        vbench_version=importlib.metadata.version("vbench"),
        torch_version=torch.__version__,
        samples_per_input=1,
        full_standard_protocol=False,
        read_frame=False,
        local=True,
        imaging_quality_preprocessing_mode="longer" if task == "t2v" else None,
        resolution="16-9" if task == "i2v" else None,
    )
    # Bind cached scores to code and source implementations, not filenames alone.
    evidence["adapter_sha256"] = sha256(Path(__file__))
    evidence["metric_source_sha256"] = {}
    for metric in METRICS[task]:
        package = "vbench2_beta_i2v" if metric.startswith("i2v_") else "vbench"
        spec = importlib.util.find_spec(f"{package}.{metric}")
        if spec is None or spec.origin is None:
            raise ImportError(f"Missing official metric: {metric}")
        evidence["metric_source_sha256"][metric] = sha256(Path(spec.origin))
    digest = fingerprint(evidence)
    previous_protocol = out / "protocol.json"
    if previous_protocol.exists() and load_json(previous_protocol).get("fingerprint") != digest:
        raise RuntimeError("Stale metric result: evaluation inputs or implementation changed; use a new run root")
    atomic_json(
        out / "protocol.json",
        dict(
            evidence,
            fingerprint=digest,
            aggregation=(
                "all 165 videos per metric" if task == "t2v" else "official dimension tags within selected 355 pairs"
            ),
        ),
    )
    summary = {}
    for metric in METRICS[task]:
        done = out / metric / "result.json"
        if done.exists():
            saved = load_json(done)
            if saved.get("fingerprint") != digest:
                raise RuntimeError(f"Stale metric result; use a new run root: {done}")
            summary[metric] = saved["score"]
    folder, prompts = evaluation_view(root, jobs)
    for metric in selected_metrics:
        if metric in summary:
            print("SKIP_METRIC", task, metric, summary[metric], flush=True)
            continue
        metric_out = out / metric
        metric_out.mkdir(parents=True, exist_ok=True)
        if task == "t2v":
            selected = jobs
        else:
            selected = [j for j in jobs if metric in j["dimensions"]]
        print("START_METRIC", task, metric, "videos", len(selected), flush=True)
        with torch.serialization.safe_globals([(OrderedDict, "typing.OrderedDict")]):
            if task == "t2v":
                from vbench import VBench

                evaluator = VBench(
                    torch.device("cuda"), str(Path(vbench.__file__).parent / "VBench_full_info.json"), str(metric_out)
                )
                evaluator.evaluate(
                    videos_path=str(folder),
                    name="results",
                    prompt_list=prompts,
                    dimension_list=[metric],
                    mode="custom_input",
                    local=True,
                    read_frame=False,
                    imaging_quality_preprocessing_mode="longer",
                )
                raw = load_json(metric_out / "results_eval_results.json")
                score, details = raw[metric]
            else:
                from vbench2_beta_i2v.utils import init_submodules

                full_info = [
                    dict(
                        prompt_en=j["prompt"],
                        dimension=j["dimensions"],
                        image_name=j["image_name"],
                        video_list=[str(root / "videos" / j["filename"])],
                    )
                    for j in selected
                ]
                meta = metric_out / "full_info.json"
                atomic_json(meta, full_info)
                # The upstream loader constructs this relative path from cwd.
                images = Path(task_record["i2v_images"])
                work = out / "reference_view"
                link = work / "vbench2_beta_i2v" / "data" / "crop" / "16-9"
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    if link.resolve() != images.resolve():
                        raise RuntimeError(f"Conflicting image link: {link}")
                elif link.exists():
                    raise RuntimeError(f"Refusing existing reference directory: {link}")
                else:
                    link.symlink_to(images, target_is_directory=True)
                package = "vbench2_beta_i2v" if metric.startswith("i2v_") else "vbench"
                compute = getattr(importlib.import_module(f"{package}.{metric}"), f"compute_{metric}")
                modules = init_submodules([metric], local=True, read_frame=False, resolution="16-9")
                with working_directory(work):
                    score, details = compute(str(meta), torch.device("cuda"), modules[metric])
        score = float(score)
        if not math.isfinite(score) or len(details) != len(selected):
            raise RuntimeError(f"Invalid metric result {metric}: {score}, {len(details)}/{len(selected)}")
        atomic_json(
            metric_out / "result.json", dict(fingerprint=digest, score=score, video_count=len(details), details=details)
        )
        summary[metric] = score
        print("DONE_METRIC", task, metric, score, len(details), flush=True)
        torch.cuda.empty_cache()
    atomic_json(out / "summary.json", summary)
    if set(summary) == set(METRICS[task]):
        atomic_json(out / "completed.json", dict(fingerprint=digest, metrics=len(summary), exit_code=0))
