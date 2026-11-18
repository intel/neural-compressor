"""CPU tests of the shared driver's orchestration contract."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("vbench_driver_test", SCRIPTS / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class DriverTests(unittest.TestCase):
    def test_all_formats_both_tasks(self):
        for task, cls in run.MODEL_CLASSES.items():
            for dtype, precision in (
                (None, "bf16"),
                ("mx_fp", "mxfp8"),
                ("nv_fp", "nvfp4"),
                ("nvfp4_v2", "nvfp4_e5m3"),
            ):
                with self.subTest(task=task, precision=precision), tempfile.TemporaryDirectory() as tmp:
                    model = Path(tmp) / "model"
                    (model / "transformer").mkdir(parents=True)
                    (model / "model_index.json").write_text(json.dumps({"_class_name": cls}))
                    if dtype:
                        (model / "transformer/quantization_config.json").write_text(json.dumps({"data_type": dtype}))
                    output = Path(tmp) / "output"
                    with patch.object(run, "validate_source"), patch.object(
                        run, "validate_model"
                    ) as validate, patch.object(run.subprocess, "run") as invoke:
                        run.main(
                            [
                                f"--model_path={model}",
                                f"--output_dir={output}",
                                "--omni_source=/source",
                                "--gpus=2,5",
                                "--i2v_images=/images",
                            ]
                        )
                    validate.assert_called_once_with(model, task, precision)
                    calls = invoke.call_args_list
                    self.assertEqual(len(calls), 2)
                    launch = calls[0].args[0]
                    self.assertEqual(launch[2:5], [task, str(model), precision])
                    self.assertIn("2,5", launch)
                    self.assertEqual(launch[-1], "/images" if task == "i2v" else "2,5")
                    self.assertEqual(calls[1].kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")

    def test_evaluate_without_model_or_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "configuration.json").write_text(json.dumps({"task": "t2v", "model": {"path": "/checkpoint"}}))
            with patch.object(run.subprocess, "run") as invoke:
                run.main(["--stage=evaluate", f"--output_dir={tmp}", "--metrics=subject_consistency", "--gpus=3"])
            self.assertEqual(invoke.call_count, 1)
            self.assertEqual(invoke.call_args.args[0][-2:], ["--metrics", "subject_consistency"])

    def test_failed_generation_never_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "model_index.json").write_text('{"_class_name":"WanPipeline"}')
            with patch.object(run, "validate_source"), patch.object(run, "validate_model"), patch.object(
                run.subprocess, "run", side_effect=run.subprocess.CalledProcessError(1, "worker")
            ) as invoke:
                with self.assertRaises(run.subprocess.CalledProcessError):
                    run.main([f"--model_path={model}", "--omni_source=/source", "--gpus=0"])
            self.assertEqual(invoke.call_count, 1)

    def test_duplicate_devices_rejected_before_launch(self):
        with patch.object(run.subprocess, "run") as invoke:
            with self.assertRaises(SystemExit):
                run.main(["--model_path=/model", "--gpus=0,00"])
            invoke.assert_not_called()
