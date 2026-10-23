"""Dependency-light regression tests for pinned V2 selection and scoring."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "swebench_pro_v2", Path(__file__).resolve().parents[1] / "lib/swebench_pro_v2.py"
)
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for index in range(642):
            task = self.root / "tasks" / f"instance_{index:03}"
            task.mkdir(parents=True)
            (task / "task.toml").touch()
        (self.root / "hard51_ids.txt").write_text("\n".join(f"instance_{i:03}" for i in range(51)))

    def test_full_and_hard_counts(self):
        self.assertEqual(len(v2.select_tasks(self.root, "full")), 642)
        self.assertEqual(len(v2.select_tasks(self.root, "hard")), 51)

    def test_selection_is_sorted_and_sliced(self):
        self.assertEqual(
            [p.name for p in v2.select_tasks(self.root, "full", slice_spec="2:4")], ["instance_002", "instance_003"]
        )
        self.assertEqual(len(v2.select_tasks(self.root, "full", num_tasks="3")), 3)
        self.assertEqual(len(v2.select_tasks(self.root, "hard", slice_spec="50:")), 1)

    def test_invalid_selections(self):
        for kwargs in (
            {"num_tasks": "0"},
            {"slice_spec": "642:"},
            {"slice_spec": "-1:3"},
            {"num_tasks": "2", "slice_spec": "0:2"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                v2.select_tasks(self.root, "full", **kwargs)

    def test_v1_tree_rejected(self):
        task = self.root / "tasks/extra"
        task.mkdir()
        (task / "task.toml").touch()
        with self.assertRaisesRegex(ValueError, "642-task"):
            v2.select_tasks(self.root, "full")

    def test_invalid_hard_list_rejected(self):
        (self.root / "hard51_ids.txt").write_text("unknown\n")
        with self.assertRaisesRegex(ValueError, "HARD-51"):
            v2.select_tasks(self.root, "hard")

    def test_slice_end_is_clamped(self):
        self.assertEqual(len(v2.select_tasks(self.root, "full", slice_spec="640:700")), 2)

    def test_prepare_preserves_official_tasks(self):
        import yaml

        tooling = self.root / "tooling/configs"
        tooling.mkdir(parents=True)
        (tooling / "mini_toolcall.yaml").write_text("agent: {step_limit: 0}\nmodel: {model_kwargs: {}}\n")
        task = self.root / "tasks/instance_000"
        (task / "instruction.md").write_text("Official task text\n")
        (task / "tests").mkdir()
        (task / "tests/test.sh").write_text("official verifier\n")
        output = self.root / "output"
        v2.prepare(
            SimpleNamespace(
                root=self.root,
                output=output,
                subset="full",
                num_tasks="1",
                slice="",
                step_limit=8,
                max_tokens=1024,
                base_url="http://172.17.0.1:8897/v1",
                commit="pinned",
                agent_version="2.4.6",
            )
        )
        copied = output / "tasks/instance_000"
        for filename in ("instruction.md", "task.toml", "tests/test.sh"):
            self.assertEqual((task / filename).read_bytes(), (copied / filename).read_bytes())
        config = yaml.safe_load((output / "agent.yaml").read_text())
        self.assertEqual(config["agent"]["step_limit"], 8)
        self.assertEqual(config["model"]["model_kwargs"]["max_tokens"], 1024)
        self.assertEqual(json.loads((output / "selection.json").read_text())["dataset_version"], "2.0.0")

    def test_endpoint_credentials_rejected(self):
        with self.assertRaisesRegex(ValueError, "without credentials"):
            v2.prepare(SimpleNamespace(base_url="http://user:secret@localhost/v1"))


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / "selection.json").write_text(
            json.dumps({"benchmark": "swebench_pro_v2", "instance_ids": ["a", "b", "c", "d"]})
        )

    def trial(self, task, reward=None, exception=None, directory=None, job="regrade"):
        trial = self.output / "jobs" / job / (directory or task)
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": f"swebench-pro/{task}",
                    "verifier_result": {"rewards": {"reward": reward}},
                    "exception_info": exception,
                }
            )
        )

    def test_only_fresh_regrade_counts_and_errors_keep_denominator(self):
        self.trial("a", 1)
        self.trial("b", 0)
        self.trial("c", 1, {"exception_message": "timeout"})
        self.trial("d", 1, job="generation")
        report = v2.build_report(self.output)
        self.assertEqual(report["accuracy"], 0.25)
        self.assertEqual(report["resolved_ids"], ["a"])
        self.assertEqual(report["error_ids"], ["c", "d"])
        self.assertEqual(report["evaluated_instances"], 2)
        self.assertEqual(report["total_instances"], 4)

    def test_missing_verifier_reward_is_error(self):
        self.trial("a")
        self.assertIn("a", v2.build_report(self.output)["error_ids"])

    def test_duplicate_trial_rejected(self):
        self.trial("a", 1)
        self.trial("a", 0, directory="retry")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            v2.build_report(self.output)

    def test_unselected_trial_rejected(self):
        self.trial("unknown", 1)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            v2.build_report(self.output)

    def test_no_regrade_results_cannot_inflate_score(self):
        report = v2.build_report(self.output)
        self.assertEqual(report["accuracy"], 0)
        self.assertEqual(report["error_instances"], 4)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / "selection.json").write_text(json.dumps({"instance_ids": ["a"]}))
        self.trial = self.output / "jobs/generation/instance_a__trial"
        (self.trial / "agent").mkdir(parents=True)
        (self.trial / "result.json").write_text(json.dumps({"task_name": "swebench-pro/a"}))
        (self.trial / "agent/model.patch").touch()
        self.args = SimpleNamespace(output=self.output)

    def trajectory(self, status):
        (self.trial / "agent/mini-swe-agent.trajectory.json").write_text(json.dumps({"info": {"exit_status": status}}))

    def test_empty_patch_is_valid_when_agent_finishes(self):
        for status in ("Submitted", "LimitsExceeded"):
            with self.subTest(status=status):
                self.trajectory(status)
                v2.check_generation(self.args)

    def test_internal_agent_exception_not_hidden_by_harbor_success(self):
        self.trajectory("ModuleNotFoundError")
        with self.assertRaises(SystemExit):
            v2.check_generation(self.args)
        errors = json.loads((self.output / "generation_errors.json").read_text())
        self.assertIn("ModuleNotFoundError", errors["a"]["exception_message"])

    def test_missing_native_trajectory_is_not_success(self):
        with self.assertRaises(SystemExit):
            v2.check_generation(self.args)

    def test_missing_patch_is_not_an_empty_submission(self):
        self.trajectory("Submitted")
        (self.trial / "agent/model.patch").unlink()
        with self.assertRaises(SystemExit):
            v2.check_generation(self.args)

    def test_missing_trial_is_not_success(self):
        (self.trial / "result.json").unlink()
        with self.assertRaises(SystemExit):
            v2.check_generation(self.args)

    def test_harbor_exception_rejected(self):
        (self.trial / "result.json").write_text(
            json.dumps({"task_name": "swebench-pro/a", "exception_info": {"exception_message": "setup failed"}})
        )
        with self.assertRaises(SystemExit):
            v2.check_generation(self.args)


class RunnerCliTests(unittest.TestCase):
    def run_script(self, *args):
        script = Path(__file__).resolve().parents[1] / "run_swebenchpro.sh"
        return subprocess.run(["bash", str(script), *args], capture_output=True, text=True)

    def test_help_needs_no_server_or_checkout(self):
        result = self.run_script("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("642", result.stdout)
        self.assertIn("--agent-base-url", result.stdout)

    def test_old_v1_options_fail_explicitly(self):
        for option in ("--block-network", "--eval-chunk-size", "--keep-images"):
            with self.subTest(option=option):
                result = self.run_script(option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unknown V2 option", result.stderr)

    def test_invalid_selection_fails_before_server_access(self):
        for args in (
            ("--num-tasks", "0"),
            ("--slice", "bad"),
            ("--num-tasks", "2", "--slice", "0:2"),
            ("--subset", "v1"),
        ):
            with self.subTest(args=args):
                self.assertNotEqual(self.run_script(*args).returncode, 0)


if __name__ == "__main__":
    unittest.main()
