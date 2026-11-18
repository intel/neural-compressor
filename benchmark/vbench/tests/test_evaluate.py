"""CPU-only evaluator contracts; placeholder bytes are not valid videos/images."""

import importlib.util
import sys
import tempfile
import unittest
from collections import Counter, OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_module("_evaluate_test_benchmark", "benchmark.py")
with patch.dict(sys.modules, {"benchmark": benchmark}):
    evaluator = load_module("_evaluate_test_adapter", "evaluate.py")


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "run"
        self.source = self.base / "metric.py"
        self.source.write_text("# fake official metric implementation\n", encoding="utf-8")
        self.torch = ModuleType("torch")
        self.torch.__version__ = "cpu-test"
        self.torch.device = Mock(return_value="mock-device")
        self.torch.serialization = SimpleNamespace(safe_globals=Mock(side_effect=lambda _: nullcontext()))
        self.torch.cuda = SimpleNamespace(empty_cache=Mock())
        self.vbench = ModuleType("vbench")
        self.vbench.__file__ = str(self.base / "vbench" / "__init__.py")
        self.enterContext(patch.dict(sys.modules, {"torch": self.torch, "vbench": self.vbench}))
        self.version = self.enterContext(
            patch.object(evaluator.importlib.metadata, "version", return_value="fake-vbench-1")
        )
        self.find_spec = self.enterContext(
            patch.object(evaluator.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(self.source)))
        )

    def prepare(self, task):
        images = None
        if task == "i2v":
            images = self.base / "reference-images"
            images.mkdir()
            for record in benchmark.load_json(REPO / "data" / "i2v_metadata.json"):
                if {"i2v_subject", "i2v_background"}.intersection(record["dimension"]):
                    (images / record["image_name"]).write_bytes(b"placeholder reference")
        jobs = benchmark.prepare(self.root, task, images)
        for job in jobs:
            (self.root / "videos" / job["filename"]).write_bytes(f"video-{job['index']}".encode())
        benchmark.atomic_json(self.root / "configuration.json", {"task": task, "precision": "bf16"})
        record = benchmark.load_json(self.root / "task.json")
        self.verify = self.enterContext(patch.object(evaluator, "verify"))
        self.read_prepared = self.enterContext(patch.object(evaluator, "read_prepared", return_value=(record, jobs)))
        return record, jobs

    def check_results(self, task, counts, scores):
        out = self.root / "evaluation"
        protocol = benchmark.load_json(out / "protocol.json")
        digest = protocol.pop("fingerprint")
        aggregation = protocol.pop("aggregation")
        self.assertEqual(digest, benchmark.fingerprint(protocol))
        self.assertEqual(protocol["samples_per_input"], 1)
        self.assertIs(protocol["full_standard_protocol"], False)
        self.assertEqual(protocol["vbench_version"], "fake-vbench-1")
        self.assertEqual(protocol["adapter_sha256"], benchmark.sha256(Path(evaluator.__file__)))
        self.assertEqual(
            protocol["metric_source_sha256"], dict.fromkeys(benchmark.METRICS[task], benchmark.sha256(self.source))
        )
        self.assertEqual(len(protocol["video_hashes"]), benchmark.EXPECTED_COUNTS[task])
        self.assertEqual(
            aggregation,
            "all 165 videos per metric" if task == "t2v" else "official dimension tags within selected 355 pairs",
        )
        self.assertEqual(benchmark.load_json(out / "summary.json"), scores)
        self.assertEqual(benchmark.load_json(out / "completed.json"), dict(fingerprint=digest, metrics=5, exit_code=0))
        for metric, count in counts.items():
            result = benchmark.load_json(out / metric / "result.json")
            self.assertEqual(result["fingerprint"], digest)
            self.assertEqual(result["score"], scores[metric])
            self.assertEqual(result["video_count"], count)
            self.assertEqual(len(result["details"]), count)
        return digest

    def check_cached_and_changed_video(self, task, compute):
        out = self.root / "evaluation"
        before = {p: p.read_bytes() for p in out.glob("*/result.json")}
        calls = compute.call_count
        evaluator.evaluate(self.root, task, None)
        self.assertEqual(compute.call_count, calls)
        self.assertEqual({p: p.read_bytes() for p in before}, before)
        old_digest = benchmark.load_json(out / "protocol.json")["fingerprint"]
        # Same size, different bytes: invalidation must use hashes, not sizes.
        video = self.root / "videos" / "0000.mp4"
        video.write_bytes(b"X" * video.stat().st_size)
        with self.assertRaisesRegex(RuntimeError, "Stale metric result"):
            evaluator.evaluate(self.root, task, None)
        self.assertEqual(compute.call_count, calls)
        self.assertEqual(benchmark.load_json(out / "protocol.json")["fingerprint"], old_digest)
        self.assertFalse((out / "completed.json").exists())
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_t2v_all_165_per_metric_duplicate_mapping_and_cache(self):
        record, jobs = self.prepare("t2v")
        # Current snapshot is unique; inject a duplicate into the prepared fixture
        # to prove the adapter preserves samples rather than deduplicating text.
        jobs[-1]["prompt"] = jobs[0]["prompt"]
        record["jobs_sha256"] = benchmark.fingerprint(jobs)
        benchmark.atomic_json(self.root / "jobs.json", jobs)
        benchmark.atomic_json(self.root / "task.json", record)
        scores = {m: (i + 1) / 10 for i, m in enumerate(benchmark.METRICS["t2v"])}
        expected = {}
        counts = Counter()
        for job in jobs:
            prompt = job["prompt"]
            target = self.root / "evaluation" / "videos" / f"{prompt}-{counts[prompt]}.mp4"
            expected[str(target)] = prompt
            counts[prompt] += 1
        self.assertEqual(len(counts), 164)
        self.assertEqual(counts[jobs[0]["prompt"]], 2)

        def official_evaluate(**kwargs):
            (metric,) = kwargs["dimension_list"]
            self.assertEqual(
                kwargs,
                dict(
                    videos_path=str(self.root / "evaluation" / "videos"),
                    name="results",
                    prompt_list=expected,
                    dimension_list=[metric],
                    mode="custom_input",
                    local=True,
                    read_frame=False,
                    imaging_quality_preprocessing_mode="longer",
                ),
            )
            self.assertEqual(len(kwargs["prompt_list"]), 165)
            self.assertEqual(Counter(kwargs["prompt_list"].values()), counts)
            for job, target in zip(jobs, expected):
                self.assertTrue(Path(target).is_symlink())
                self.assertEqual(Path(target).resolve(), self.root / "videos" / job["filename"])
            benchmark.atomic_json(
                self.root / "evaluation" / metric / "results_eval_results.json",
                {metric: [scores[metric], [{"video_path": p} for p in expected]]},
            )

        compute = Mock(side_effect=official_evaluate)
        self.vbench.VBench = Mock(return_value=SimpleNamespace(evaluate=compute))
        evaluator.evaluate(self.root, "t2v", None)
        self.assertEqual(compute.call_count, 5)
        for metric, call in zip(benchmark.METRICS["t2v"], self.vbench.VBench.call_args_list):
            self.assertEqual(
                call.args,
                (
                    "mock-device",
                    str(self.base / "vbench" / "VBench_full_info.json"),
                    str(self.root / "evaluation" / metric),
                ),
            )
        self.check_results("t2v", dict.fromkeys(scores, 165), scores)
        self.check_cached_and_changed_video("t2v", compute)
        self.verify.assert_called_with(self.root, "t2v")
        self.read_prepared.assert_called_with(self.root, "t2v")
        self.torch.serialization.safe_globals.assert_called_with([(OrderedDict, "typing.OrderedDict")])

    def test_i2v_dimension_counts_full_info_relative_references_and_cache(self):
        record, jobs = self.prepare("i2v")
        counts = dict(
            i2v_subject=246,
            subject_consistency=246,
            background_consistency=109,
            motion_smoothness=246,
            i2v_background=109,
        )
        scores = {m: (i + 1) / 10 for i, m in enumerate(benchmark.METRICS["i2v"])}
        utils = ModuleType("vbench2_beta_i2v.utils")
        utils.init_submodules = Mock(side_effect=lambda metrics, **kwargs: {metrics[0]: metrics[0]})
        modules = {"vbench2_beta_i2v": ModuleType("vbench2_beta_i2v"), "vbench2_beta_i2v.utils": utils}

        def official_compute(meta, device, metric):
            selected = [j for j in jobs if metric in j["dimensions"]]
            full_info = benchmark.load_json(Path(meta))
            self.assertEqual(len(full_info), counts[metric])
            self.assertEqual(device, "mock-device")
            self.assertEqual(Path.cwd(), self.root / "evaluation" / "reference_view")
            relative = Path("vbench2_beta_i2v/data/crop/16-9")
            self.assertTrue(relative.is_symlink())
            self.assertEqual(relative.resolve(), Path(record["i2v_images"]))
            self.assertEqual(
                full_info,
                [
                    dict(
                        prompt_en=j["prompt"],
                        dimension=j["dimensions"],
                        image_name=j["image_name"],
                        video_list=[str(self.root / "videos" / j["filename"])],
                    )
                    for j in selected
                ],
            )
            for entry in full_info:
                self.assertIn(metric, entry["dimension"])
                self.assertEqual((relative / entry["image_name"]).read_bytes(), b"placeholder reference")
                self.assertTrue(Path(entry["video_list"][0]).is_file())
            return scores[metric], [{"video_path": e["video_list"][0]} for e in full_info]

        compute = Mock(side_effect=official_compute)
        for metric in counts:
            package = "vbench2_beta_i2v" if metric.startswith("i2v_") else "vbench"
            module = ModuleType(f"{package}.{metric}")
            setattr(module, f"compute_{metric}", compute)
            modules[module.__name__] = module
        self.enterContext(patch.dict(sys.modules, modules))
        cwd = Path.cwd()
        evaluator.evaluate(self.root, "i2v", None)
        self.assertEqual(Path.cwd(), cwd)
        self.assertEqual(compute.call_count, 5)
        self.assertEqual([call.args[2] for call in compute.call_args_list], list(counts))
        self.assertEqual(utils.init_submodules.call_count, 5)
        for metric, call in zip(counts, utils.init_submodules.call_args_list):
            self.assertEqual(call.args, ([metric],))
            self.assertEqual(call.kwargs, dict(local=True, read_frame=False, resolution="16-9"))
        self.check_results("i2v", counts, scores)
        self.check_cached_and_changed_video("i2v", compute)
        self.assertEqual(Path.cwd(), cwd)

    def test_evaluation_view_rejects_collisions_and_unexpected_files(self):
        _, jobs = self.prepare("t2v")
        folder, prompts = evaluator.evaluation_view(self.root, jobs)
        self.assertEqual(len(prompts), 165)
        self.assertEqual(evaluator.evaluation_view(self.root, jobs), (folder, prompts))
        target = Path(next(iter(prompts)))
        target.unlink()
        target.symlink_to(self.root / "videos" / jobs[1]["filename"])
        with self.assertRaisesRegex(RuntimeError, "Conflicting evaluation link"):
            evaluator.evaluation_view(self.root, jobs)
        target.unlink()
        target.write_bytes(b"do not replace")
        with self.assertRaisesRegex(RuntimeError, "Refusing existing non-link"):
            evaluator.evaluation_view(self.root, jobs)
        self.assertEqual(target.read_bytes(), b"do not replace")
        target.unlink()
        evaluator.evaluation_view(self.root, jobs)
        (folder / "unexpected.mp4").write_bytes(b"extra")
        with self.assertRaisesRegex(RuntimeError, "Unexpected files"):
            evaluator.evaluation_view(self.root, jobs)


if __name__ == "__main__":
    unittest.main()
