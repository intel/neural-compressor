"""Chunk replay and original runner regressions: no Docker, GPU, or downloads."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_data", ROOT / "lib/benchmark_data.py")
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        base = Path(temp.name)
        (base / "tooling").mkdir()
        (base / "tooling/patch_replay.py").touch()
        for task in ("a", "ab", "b"):
            target = base / "tasks" / task
            (target / "tests").mkdir(parents=True)
            (target / "task.toml").write_text(
                f'[environment]\ndocker_image = "{v2.pro_image(task)}"\nbuild_timeout_sec = 1800\n'
                '[verifier]\nnetwork_mode = "public"\ntimeout_sec = 3000\n'
                '[agent]\nnetwork_mode = "no-network"\n'
            )
            (target / "tests/test.sh").write_text("official verifier\n")
        (base / "instances.csv").write_text("instance_id\na\nb\n")
        (base / "patches.json").write_text(
            json.dumps(
                [
                    {"instance_id": "a", "patch": "diff contents\n", "prefix": "ignored"},
                    {"instance_id": "b", "patch": "", "prefix": "ignored"},
                ]
            )
        )
        self.args = SimpleNamespace(
            tasks_root=base / "tasks",
            instances=base / "instances.csv",
            patches=base / "patches.json",
            output=base / "output",
            workers=2,
            pull_timeout=600,
            block_network=False,
        )

    def run_adapter(self, trials=None, returncode=0):
        if trials is None:
            trials = [self.trial("a", 1.0), self.trial("b", 0.0)]

        def harbor(command, **kwargs):
            self.command, self.kwargs = command, kwargs
            for index, trial in enumerate(trials):
                target = self.args.output / f"jobs/regrade/trial_{index}/result.json"
                target.parent.mkdir(parents=True)
                target.write_text(json.dumps(trial))
            return SimpleNamespace(returncode=returncode)

        with patch.object(v2.subprocess, "run", side_effect=harbor):
            status = v2.evaluate_pro(self.args)
        return status, json.loads((self.args.output / "eval_results.json").read_text())

    @staticmethod
    def trial(task, reward, exception=None):
        return {
            "task_name": f"swebench-pro/{task}",
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": exception,
        }

    def test_official_replay_cli_source_trials_and_rewards(self):
        status, report = self.run_adapter()
        self.assertEqual(status, 0)
        self.assertEqual(report["results"], {"a": True, "b": False})
        command = self.command
        self.assertEqual(command[:2], ["harbor", "run"])
        self.assertEqual(command[command.index("-e") + 1], "docker")
        self.assertEqual(command[command.index("-a") + 1], "patch_replay:PatchReplayAgent")
        self.assertEqual([command[i + 1] for i, flag in enumerate(command) if flag == "-i"], ["a", "b"])
        self.assertEqual(command[command.index("-n") + 1], "2")
        self.assertIn("--no-delete", command)  # Shell, not Harbor, owns image deletion.
        self.assertAlmostEqual(float(command[command.index("--environment-build-timeout-multiplier") + 1]), 1 / 3)
        source = self.args.output / "source_job"
        self.assertIn(f"source_job={source}", command)
        for task, content in (("a", "diff contents\n"), ("b", "")):
            trial = source / f"instance_{task}"
            self.assertEqual(json.loads((trial / "result.json").read_text()), {"task_name": task})
            self.assertEqual((trial / "agent/model.patch").read_text(), content)
        self.assertTrue(self.kwargs["env"]["PYTHONPATH"].startswith(str(self.args.tasks_root.parent / "tooling")))
        self.assertEqual(self.args.output.stat().st_mode & 0o777, 0o700)

    def test_block_network_uses_private_copies_only(self):
        self.args.block_network = True
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://test-proxy"}):
            status, _ = self.run_adapter()
        self.assertEqual(status, 0)
        self.assertNotIn("HTTPS_PROXY=http://test-proxy", self.command)
        copied = self.args.output / "tasks"
        self.assertEqual(self.command[self.command.index("-p") + 1], str(copied))
        self.assertEqual(sorted(p.name for p in copied.iterdir()), ["a", "b"])
        for task in ("a", "b"):
            original = (self.args.tasks_root / task / "task.toml").read_text()
            self.assertEqual(
                (copied / task / "task.toml").read_text(),
                original.replace('[verifier]\nnetwork_mode = "public"', '[verifier]\nnetwork_mode = "no-network"'),
            )
            self.assertEqual(
                (copied / task / "tests/test.sh").read_bytes(),
                (self.args.tasks_root / task / "tests/test.sh").read_bytes(),
            )

    def test_verifier_proxy_forwarding_only(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://test-proxy"}):
            status, _ = self.run_adapter()
        self.assertEqual(status, 0)
        self.assertIn("HTTPS_PROXY=http://test-proxy", self.command)
        self.assertNotIn("--ae", self.command)

    def test_missing_invalid_rewards_and_exceptions_are_errors(self):
        for reward, exception in ((None, None), ("1", None), (True, None), (2, None), (1, {"message": "secret"})):
            with self.subTest(reward=reward, exception=exception), tempfile.TemporaryDirectory() as output:
                self.args.output = Path(output)
                status, report = self.run_adapter([self.trial("a", reward, exception)])
                self.assertEqual(status, 1)
                self.assertEqual(report["error_ids"], ["a", "b"])
                self.assertEqual(report["total_instances"], 2)
                self.assertNotIn("secret", json.dumps(report))

    def test_subprocess_failure_is_reported_without_proxy_credentials(self):
        with patch.object(v2.subprocess, "run", side_effect=subprocess.TimeoutExpired(["secret"], 1)):
            self.assertEqual(v2.evaluate_pro(self.args), 1)
        report_text = (self.args.output / "eval_results.json").read_text()
        self.assertNotIn("secret", report_text)
        self.assertEqual(json.loads(report_text)["error_ids"], ["a", "b"])

    def test_harbor_exit_and_missing_results_fail(self):
        for trials, returncode in (([], 0), ([self.trial("a", 1)], 1), ([self.trial("a", 1), self.trial("b", 0)], 2)):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as output:
                self.args.output = Path(output)
                status, report = self.run_adapter(trials, returncode)
                self.assertEqual(status, 1)
                self.assertEqual(report["total_instances"], 2)

    def test_duplicate_unselected_and_malformed_results_fail(self):
        for bad in (self.trial("a", 0), self.trial("ab", 1), {}, [], {"task_name": None}):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as output:
                self.args.output = Path(output)
                status, _ = self.run_adapter([self.trial("a", 1), self.trial("b", 0), bad])
                self.assertEqual(status, 1)

    def test_stale_output_and_unmappable_pull_timeout_never_launch(self):
        (self.args.output / "jobs/regrade").mkdir(parents=True)
        with patch.object(v2.subprocess, "run") as run:
            self.assertEqual(v2.evaluate_pro(self.args), 1)
            run.assert_not_called()
        self.args.output = self.args.output.parent / "new-output"
        config = self.args.tasks_root / "b/task.toml"
        config.write_text(config.read_text().replace("1800", "1200"))
        with patch.object(v2.subprocess, "run") as run:
            self.assertEqual(v2.evaluate_pro(self.args), 1)
            run.assert_not_called()
        self.assertIn("--pull-timeout", (self.args.output / "eval_results.json").read_text())

    def test_failure_remains_in_denominator_and_retry_overrides_error(self):
        first = self.args.output.parent / "first.json"
        second = self.args.output.parent / "second.json"
        first.write_text(json.dumps(v2.build_pro_report({"a": True}, {"b": "timeout"})))
        second.write_text(json.dumps(v2.build_pro_report({"b": False})))
        listing = first.with_suffix(".txt")
        listing.write_text(f"{first}\n")
        args = SimpleNamespace(chunk_list=listing, report=first.parent / "report.json")
        v2.merge_pro_reports(args)
        self.assertEqual(json.loads(args.report.read_text())["accuracy"], 0.5)
        listing.write_text(f"{first}\n{second}\n")
        v2.merge_pro_reports(args)
        report = json.loads(args.report.read_text())
        self.assertEqual(report["error_ids"], [])
        self.assertEqual(report["unresolved_ids"], ["b"])

    def test_invalid_inputs_never_launch(self):
        for content in ("instance_id\na\na\n", "instance_id\na*\n", "instance_id\n../a\n"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as output:
                self.args.output = Path(output)
                self.args.instances.write_text(content)
                with patch.object(v2.subprocess, "run") as run:
                    with self.assertRaises(ValueError):
                        v2.evaluate_pro(self.args)
                    run.assert_not_called()
        self.args.instances.write_text("instance_id\na\nb\n")
        for records in (
            [],
            [{"instance_id": "ab", "patch": "x"}],
            [{"instance_id": "a", "patch": "x"}] * 2,
            [None, None],
            [{}, {}],
            ["a", "b"],
            [{"instance_id": [], "patch": "x"}] * 2,
            [{"instance_id": "a", "patch": None}, {"instance_id": "b", "patch": ""}],
        ):
            with tempfile.TemporaryDirectory() as output:
                self.args.output = Path(output)
                self.args.patches.write_text(json.dumps(records))
                with patch.object(v2.subprocess, "run") as run:
                    with self.assertRaises(ValueError):
                        v2.evaluate_pro(self.args)
                    run.assert_not_called()


class DataCommandTests(unittest.TestCase):
    def test_cli_helpers_with_no_return_value_exit_successfully(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source, output = base / "preds.json", base / "patches.json"
            source.write_text(json.dumps({"a": {"model_patch": ""}}))
            commands = [
                ["normalize-pro", "--source", str(source), "--output", str(output), "--prefix", "test"],
                ["pending-predictions", "--preds", str(source), "--claimed", str(base / "missing.txt")],
            ]
            for command in commands:
                with self.subTest(command=command):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "lib/benchmark_data.py"), *command], capture_output=True, text=True
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())[0]["instance_id"], "a")

    def test_evaluation_cli_propagates_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "instances.csv").write_text("instance_id\na\n")
            (base / "patches.json").write_text(json.dumps([{"instance_id": "a", "patch": ""}]))
            # Missing official task files must fail before Docker is ever launched.
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "lib/benchmark_data.py"),
                    "evaluate-pro",
                    "--tasks-root",
                    str(base / "missing-tasks"),
                    "--instances",
                    str(base / "instances.csv"),
                    "--patches",
                    str(base / "patches.json"),
                    "--output",
                    str(base / "output"),
                    "--workers",
                    "1",
                    "--pull-timeout",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((base / "output/eval_results.json").read_text())
            self.assertEqual(report["error_ids"], ["a"])
            self.assertEqual(report["total_instances"], 1)

    def test_selection_writes_csv_and_images_without_json_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            csv_path = base / "instances.csv"
            loader = Mock(return_value=[{"instance_id": "a"}, {"instance_id": "b"}])
            dataframe = Mock()
            with patch.dict(
                sys.modules,
                {"datasets": SimpleNamespace(load_dataset=loader), "pandas": SimpleNamespace(DataFrame=dataframe)},
            ):
                v2.select_pro(SimpleNamespace(csv=csv_path, images=base / "images.txt", slice="1:2"))
            loader.assert_called_once_with("ScaleAI/SWE-bench_Pro", "default", split="test")
            dataframe.assert_called_once_with([{"instance_id": "b", "image_name": v2.pro_image("b")}])
            dataframe.return_value.to_csv.assert_called_once_with(csv_path, index=False)
            self.assertEqual((base / "images.txt").read_text(), v2.pro_image("b") + "\n")
            self.assertFalse(csv_path.with_suffix(".json").exists())

    def test_empty_selection_writes_no_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataframe = Mock()
            with patch.dict(
                sys.modules,
                {
                    "datasets": SimpleNamespace(load_dataset=Mock(return_value=[{"instance_id": "a"}])),
                    "pandas": SimpleNamespace(DataFrame=dataframe),
                },
            ):
                with self.assertRaisesRegex(RuntimeError, "selected no instances"):
                    v2.select_pro(SimpleNamespace(csv=base / "instances.csv", images=base / "images.txt", slice="1:2"))
            dataframe.assert_not_called()
            self.assertEqual(list(base.iterdir()), [])


class OriginalRunnerTests(unittest.TestCase):
    def test_native_hf_subset_and_slice(self):
        source = (ROOT / "run_swebenchpro.sh").read_text()
        self.assertIn("--subset ScaleAI/SWE-bench_Pro", source)
        self.assertIn('--slice "${FULL_SLICE}"', source)
        self.assertNotIn("instances.json", source)
        self.assertIn("v2_selection.txt", source)

    def test_patch_native_loader_and_dataset_scoped_image_mapping(self):
        source = (ROOT / "patches/swebench_pro_image.patch").read_text()
        loader_hunk = source.split("@@ -212,", 1)[1].split("\n", 1)[1]
        self.assertIn("     instances = list(load_dataset(dataset_path, split=split))", loader_hunk)
        self.assertIn("     instances = filter_instances(instances,", loader_hunk)
        self.assertNotIn("json.loads", loader_hunk)
        self.assertNotIn("swebench_pro_version", source)
        self.assertNotIn("def get_swebench_docker_image_name", source)
        # Execute the actual mapping added by the patch, without importing mini-SWE-agent.
        mapping = "\n".join(line[5:] for line in loader_hunk.splitlines() if line.startswith("+"))
        for dataset in ("ScaleAI/SWE-bench_Pro", "princeton-nlp/SWE-Bench_Verified", "another/dataset"):
            with self.subTest(dataset=dataset):
                instances = [
                    {"instance_id": "instance_shared__123"},
                    {"instance_id": "instance_other__456", "image_name": "custom/image:tag"},
                    {"instance_id": "instance_null__789", "image_name": None},
                ]
                exec(mapping, {"dataset_path": dataset, "instances": instances})
                self.assertEqual(instances[1]["image_name"], "custom/image:tag")
                if dataset == "ScaleAI/SWE-bench_Pro":
                    self.assertEqual(instances[0]["image_name"], v2.pro_image(instances[0]["instance_id"]))
                    self.assertEqual(instances[2]["image_name"], v2.pro_image(instances[2]["instance_id"]))
                else:
                    self.assertNotIn("image_name", instances[0])
                    self.assertIsNone(instances[2]["image_name"])

    def test_mini_agent_patch_is_well_formed(self):
        result = subprocess.run(
            ["git", "apply", "--numstat", str(ROOT / "patches/swebench_pro_image.patch")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("src/minisweagent/environments/docker.py", result.stdout)
        self.assertIn("src/minisweagent/run/extra/swebench.py", result.stdout)

    def test_original_cli_options_retained(self):
        result = subprocess.run(["bash", str(ROOT / "run_swebenchpro.sh"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "host port served-name num-tasks slice workers eval-workers step-limit pull-timeout "
            "command-timeout eval-chunk-size poll-interval health-interval health-failures tag "
            "skip-eval block-network keep-images"
        ).split():
            self.assertIn(f"--{option}", result.stdout)

    def test_async_chunks_and_image_policy_retained(self):
        source = (ROOT / "run_swebenchpro.sh").read_text()
        for fragment in (
            "GEN_PID=$!",
            "EVAL_PID=$!",
            "MONITOR_PID=$!",
            "pending-predictions",
            "prepare-pro-chunk",
            "merge-pro-reports",
            "remove_chunk_images",
            "docker image rm",
            "KEEP_IMAGES",
            "EVAL_CHUNK_SIZE",
            "claimed_ids.txt",
            'wait "${EVAL_PID}"',
            'wait "${GEN_PID}"',
            "evaluate-pro",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("swe_bench_pro_eval.py", source)

    def test_cleanup_noops_succeed_under_errexit(self):
        source = (ROOT / "run_swebenchpro.sh").read_text()
        functions = "\n".join(
            source[source.index(f"{name}() {{") :].split("\n}", 1)[0] + "\n}"
            for name in ("remove_chunk_images", "stop_vllm_monitor")
        )
        with tempfile.TemporaryDirectory() as directory:
            image_list = Path(directory) / "images.txt"
            image_list.write_text("unrelated/image:tag\n")
            script = (
                "set -euo pipefail\n" + functions + "\n"
                'KEEP_IMAGES=true\nremove_chunk_images "$1"\n'
                'KEEP_IMAGES=false\nremove_chunk_images "$1"\n'
                'remove_chunk_images "$1.missing"\n'
                'MONITOR_PID=""\nstop_vllm_monitor\nstop_vllm_monitor\necho survived\n'
            )
            result = subprocess.run(["bash", "-s", "--", str(image_list)], input=script, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "survived")


if __name__ == "__main__":
    unittest.main()
