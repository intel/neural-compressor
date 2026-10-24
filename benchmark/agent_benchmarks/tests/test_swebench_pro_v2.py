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
        for task in ("a", "ab", "b"):
            target = base / "tasks" / task
            (target / "tests").mkdir(parents=True)
            (target / "task.toml").write_text(
                f'[environment]\ndocker_image = "{v2.pro_image(task)}"\n'
                '[verifier]\ntimeout_sec = 3000\n')
            (target / "tests/test.sh").write_text("official verifier\n")
        (base / "instances.csv").write_text("instance_id\na\nb\n")
        (base / "patches.json").write_text(json.dumps([
            {"instance_id": "a", "patch": "diff contents\n", "prefix": "ignored"},
            {"instance_id": "b", "patch": "", "prefix": "ignored"}]))
        self.args = SimpleNamespace(tasks_root=base / "tasks", instances=base / "instances.csv",
                                    patches=base / "patches.json", output=base / "output",
                                    workers=2, pull_timeout=600, block_network=False)

    def docker(self, command, **kwargs):
        self.commands.append((command, kwargs))
        stdout, returncode = "", 0
        if command[1] == "exec" and command[-1].startswith("if [ -d /app"):
            stdout = self.cwd + "\n"
        if command[1] == "exec" and "apply" in command:
            returncode = self.apply_status
        if command[-2:] == ["bash", "/tests/test.sh"]:
            returncode = self.verifier_status
        if command[1] == "cp" and command[-2].endswith(":/logs/verifier/."):
            if self.reward is not None:
                (Path(command[-1]) / "reward.txt").write_text(self.reward)
        return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

    def run_adapter(self):
        self.commands = []
        self.cwd = getattr(self, "cwd", "/app")
        self.reward = getattr(self, "reward", "1")
        self.apply_status = getattr(self, "apply_status", 0)
        self.verifier_status = getattr(self, "verifier_status", 0)
        with patch.object(v2.subprocess, "run", side_effect=self.docker):
            status = v2.evaluate_pro(self.args)
        return status, json.loads((self.args.output / "eval_results.json").read_text())

    def test_fresh_docker_replay_and_official_verifier(self):
        status, report = self.run_adapter()
        self.assertEqual(status, 0)
        self.assertEqual(report["resolved_ids"], ["a", "b"])
        starts = [c for c, _ in self.commands if c[1] == "run"]
        self.assertEqual(len(starts), 2)
        self.assertNotEqual(starts[0][starts[0].index("--name") + 1], starts[1][starts[1].index("--name") + 1])
        self.assertTrue(all("--entrypoint" in c for c in starts))
        self.assertTrue(all(kw["timeout"] == 600 for c, kw in self.commands if c[1] == "run"))
        self.assertEqual(len([c for c, _ in self.commands if "apply" in c]), 1)  # Empty patch still verified.
        self.assertEqual(len([c for c, kw in self.commands if c[-2:] == ["bash", "/tests/test.sh"] and kw["timeout"] == 3000]), 2)
        self.assertEqual(len([c for c, _ in self.commands if c[1:3] == ["rm", "-f"]]), 2)

    def test_testbed_and_block_network(self):
        self.cwd = "/testbed"
        self.args.block_network = True
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://test-proxy"}):
            status, _ = self.run_adapter()
        self.assertEqual(status, 0)
        for command, _ in self.commands:
            if command[1] == "run":
                self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertNotIn("HTTPS_PROXY=http://test-proxy", command)
        self.assertTrue(any(c[1:4] == ["exec", "-w", "/testbed"] for c, _ in self.commands))

    def test_invalid_submission_is_unresolved(self):
        self.apply_status = 1
        status, report = self.run_adapter()
        self.assertEqual(status, 0)
        self.assertEqual(report["unresolved_ids"], ["a"])
        self.assertEqual(report["error_ids"], [])

    def test_missing_reward_is_infrastructure_error(self):
        self.reward = None
        status, report = self.run_adapter()
        self.assertEqual(status, 1)
        self.assertEqual(report["error_ids"], ["a", "b"])
        self.assertEqual(report["total_instances"], 2)
        self.assertEqual(report["evaluated_instances"], 0)
        self.assertEqual(len([c for c, _ in self.commands if c[1:3] == ["rm", "-f"]]), 2)

    def test_failed_official_tests_are_unresolved_not_infrastructure_errors(self):
        self.reward = "0"
        self.verifier_status = 1
        status, report = self.run_adapter()
        self.assertEqual(status, 0)
        self.assertEqual(report["unresolved_ids"], ["a", "b"])
        self.assertEqual(report["error_ids"], [])

    def test_inconsistent_reward_is_infrastructure_error(self):
        for reward, returncode in (("1", 1), ("0", 0), ("invalid", 0)):
            with self.subTest(reward=reward, returncode=returncode), tempfile.TemporaryDirectory() as output:
                self.args.output = Path(output)
                self.reward, self.verifier_status = reward, returncode
                status, report = self.run_adapter()
                self.assertEqual(status, 1)
                self.assertEqual(report["error_ids"], ["a", "b"])

    def test_verifier_timeout_is_reported_without_proxy_credentials(self):
        self.commands = []
        self.cwd, self.reward, self.apply_status = "/app", "1", 0

        def timed_out(command, **kwargs):
            if command[-2:] == ["bash", "/tests/test.sh"]:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return self.docker(command, **kwargs)

        with patch.dict(os.environ, {"HTTPS_PROXY": "http://user:secret@proxy"}), \
                patch.object(v2.subprocess, "run", side_effect=timed_out):
            self.assertEqual(v2.evaluate_pro(self.args), 1)
        report_text = (self.args.output / "eval_results.json").read_text()
        self.assertNotIn("secret", report_text)
        self.assertEqual(json.loads(report_text)["error_ids"], ["a", "b"])
        self.assertEqual(len([c for c, _ in self.commands if c[1:3] == ["rm", "-f"]]), 2)

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
        for records in ([], [{"instance_id": "ab", "patch": "x"}],
                [{"instance_id": "a", "patch": "x"}] * 2,
                [None, None], [{}, {}], ["a", "b"],
                [{"instance_id": [], "patch": "x"}] * 2,
                [{"instance_id": "a", "patch": None}, {"instance_id": "b", "patch": ""}]):
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
                    result = subprocess.run([sys.executable, str(ROOT / "lib/benchmark_data.py"), *command],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())[0]["instance_id"], "a")

    def test_evaluation_cli_propagates_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "instances.csv").write_text("instance_id\na\n")
            (base / "patches.json").write_text(json.dumps([{"instance_id": "a", "patch": ""}]))
            # Missing official task files must fail before Docker is ever launched.
            result = subprocess.run(
                [sys.executable, str(ROOT / "lib/benchmark_data.py"), "evaluate-pro",
                 "--tasks-root", str(base / "missing-tasks"), "--instances", str(base / "instances.csv"),
                 "--patches", str(base / "patches.json"), "--output", str(base / "output"),
                 "--workers", "1", "--pull-timeout", "1"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((base / "output/eval_results.json").read_text())
            self.assertEqual(report["error_ids"], ["a"])
            self.assertEqual(report["total_instances"], 1)

    def test_changed_selection_preserves_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            csv_path, snapshot = base / "instances.csv", base / "instances.json"
            csv_path.write_text("original CSV\n")
            snapshot.write_text(json.dumps([{"instance_id": "old"}]))
            loader = Mock(return_value=[{"instance_id": "a"}, {"instance_id": "b"}])
            dataframe = Mock()
            with patch.dict(sys.modules, {"datasets": SimpleNamespace(load_dataset=loader),
                                         "pandas": SimpleNamespace(DataFrame=dataframe)}):
                with self.assertRaisesRegex(RuntimeError, "Selected V2 data changed"):
                    v2.select_pro(SimpleNamespace(csv=csv_path, images=base / "images.txt", slice="1:2"))
            loader.assert_called_once_with("ScaleAI/SWE-bench_Pro", "default", split="test")
            dataframe.assert_not_called()
            self.assertEqual(csv_path.read_text(), "original CSV\n")
            self.assertEqual(json.loads(snapshot.read_text()), [{"instance_id": "old"}])


class OriginalRunnerTests(unittest.TestCase):
    def test_mini_agent_patch_is_well_formed(self):
        result = subprocess.run(["git", "apply", "--numstat", str(ROOT / "patches/swebench_pro_image.patch")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("src/minisweagent/environments/docker.py", result.stdout)
        self.assertIn("src/minisweagent/run/extra/swebench.py", result.stdout)

    def test_original_cli_options_retained(self):
        result = subprocess.run(["bash", str(ROOT / "run_swebenchpro.sh"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("host port served-name num-tasks slice workers eval-workers step-limit pull-timeout "
                       "command-timeout eval-chunk-size poll-interval health-interval health-failures tag "
                       "skip-eval block-network keep-images").split():
            self.assertIn(f"--{option}", result.stdout)

    def test_async_chunks_and_image_policy_retained(self):
        source = (ROOT / "run_swebenchpro.sh").read_text()
        for fragment in ('GEN_PID=$!', 'EVAL_PID=$!', 'MONITOR_PID=$!', 'pending-predictions',
                         'prepare-pro-chunk', 'merge-pro-reports', 'remove_chunk_images',
                         'docker image rm', 'KEEP_IMAGES', 'EVAL_CHUNK_SIZE', 'claimed_ids.txt',
                         'wait "${EVAL_PID}"', 'wait "${GEN_PID}"', 'evaluate-pro'):
            self.assertIn(fragment, source)
        self.assertNotIn("swe_bench_pro_eval.py", source)

    def test_cleanup_noops_succeed_under_errexit(self):
        source = (ROOT / "run_swebenchpro.sh").read_text()
        functions = "\n".join(source[source.index(f"{name}() {{"):].split("\n}", 1)[0] + "\n}"
                              for name in ("remove_chunk_images", "stop_vllm_monitor"))
        with tempfile.TemporaryDirectory() as directory:
            image_list = Path(directory) / "images.txt"
            image_list.write_text("unrelated/image:tag\n")
            script = ('set -euo pipefail\n' + functions + '\n'
                      'KEEP_IMAGES=true\nremove_chunk_images "$1"\n'
                      'KEEP_IMAGES=false\nremove_chunk_images "$1"\n'
                      'remove_chunk_images "$1.missing"\n'
                      'MONITOR_PID=""\nstop_vllm_monitor\nstop_vllm_monitor\necho survived\n')
            result = subprocess.run(["bash", "-s", "--", str(image_list)], input=script,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "survived")


if __name__ == "__main__":
    unittest.main()
