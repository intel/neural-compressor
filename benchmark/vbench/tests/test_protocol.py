"""CPU-only protocol and shell-helper contracts; stdlib unittest discovery."""

import importlib.util
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
spec = importlib.util.spec_from_file_location("wan_protocol_test", SCRIPTS / "benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def read_json(path, **kwargs):
    return json.loads(path.read_text(encoding="utf-8"), **kwargs)


class ProtocolTests(unittest.TestCase):
    def test_import_is_lazy_without_site_packages(self):
        code = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('bench', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert not {'torch', 'numpy', 'vllm', 'vllm_omni', 'diffusers', 'vbench'} & sys.modules.keys()
assert len(module.build_jobs('t2v')) == 165
"""
        subprocess.run(
            [sys.executable, "-S", "-c", code, str(SCRIPTS / "benchmark.py")],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_t2v_counts_order_and_groups(self):
        records = read_json(REPO / "data/t2v_prompts.json")
        jobs = benchmark.build_jobs("t2v")
        self.assertEqual(len(jobs), 165)
        self.assertEqual(Counter(j["group"] for j in jobs), {"subject_consistency": 72, "overall_consistency": 93})
        self.assertEqual([j["prompt"] for j in jobs], [r["prompt"] for r in records])
        self.assertEqual([j["index"] for j in jobs], list(range(165)))
        self.assertEqual([j["filename"] for j in jobs], [f"{i:04d}.mp4" for i in range(165)])
        self.assertTrue(all((j["width"], j["height"]) == (1280, 720) for j in jobs))

    def test_i2v_official_union_and_tag_counts_without_fake_images(self):
        records = read_json(REPO / "data/i2v_metadata.json")
        self.assertEqual(len(records), 1118)
        selected = [r for r in records if {"i2v_subject", "i2v_background"} & set(r["dimension"])]
        self.assertEqual(len(selected), 355)
        expected = dict(
            i2v_subject=246,
            subject_consistency=246,
            background_consistency=109,
            motion_smoothness=246,
            i2v_background=109,
        )
        self.assertEqual({m: sum(m in r["dimension"] for r in selected) for m in expected}, expected)
        self.assertEqual(read_json(REPO / "results/i2v.json")["protocol"]["metric_counts"], expected)
        # Metadata counts are not evidence of valid reference images; do not mock image verification.
        with self.assertRaisesRegex(ValueError, "requires --i2v-images"):
            benchmark.build_jobs("i2v")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                benchmark.build_jobs("i2v", Path(directory))

    def test_metrics_and_published_protocol_match_runner(self):
        for task in ("t2v", "i2v"):
            result = read_json(REPO / "results" / f"{task}.json")
            protocol = result["protocol"]
            self.assertEqual(protocol["video_count"], benchmark.EXPECTED_COUNTS[task])
            self.assertEqual(protocol["videos_per_prompt"], 1)
            self.assertIs(protocol["full_standard_protocol"], False)
            self.assertEqual(protocol["vllm_omni_commit"], benchmark.SOURCE_COMMIT)
            self.assertEqual(set(protocol["metric_counts"]), set(benchmark.METRICS[task]))
            for model in result["models"].values():
                self.assertEqual(set(model["scores"]), set(benchmark.METRICS[task]))
            aliases = {
                "cache": "cache_backend",
                "cpu_offload": "enable_cpu_offload",
                "vae_tiling": "vae_use_tiling",
                "vae_slicing": "vae_use_slicing",
            }
            for key, value in protocol["generation"].items():
                self.assertEqual(value, benchmark.SETTINGS[task][aliases.get(key, key)], (task, key))
            if task == "t2v":
                self.assertEqual(protocol["metric_counts"], dict.fromkeys(benchmark.METRICS[task], 165))
            else:
                self.assertEqual(protocol["metadata_sha256"], benchmark.sha256(REPO / "data/i2v_metadata.json"))

    def test_fixed_settings_and_full_official_negative(self):
        negative = (
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
            "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
            "杂乱的背景，三条腿，背景人很多，倒着走"
        )
        self.assertEqual(benchmark.NEGATIVE_PROMPT, negative)
        self.assertEqual(benchmark.NEGATIVE, negative)
        common = dict(
            frames=81,
            fps=16,
            steps=40,
            negative_prompt=negative,
            attention="TRTLLM_ATTN",
            enforce_eager=True,
            cache_backend=None,
            enable_cpu_offload=False,
            enable_layerwise_offload=False,
            vae_use_tiling=False,
            vae_use_slicing=False,
            samples_per_prompt=1,
        )
        self.assertEqual(
            benchmark.SETTINGS,
            {
                "t2v": dict(
                    common,
                    width=1280,
                    height=720,
                    seed=42,
                    guidance_scale=4.0,
                    guidance_scale_2=3.0,
                    boundary_ratio=0.875,
                    flow_shift=5.0,
                    linear_backend="flashinfer_cutlass",
                    init_timeout=3600,
                    stage_init_timeout=3600,
                    enable_diffusion_pipeline_profiler=True,
                ),
                "i2v": dict(
                    common,
                    width=832,
                    height=480,
                    seed=0,
                    guidance_scale=3.5,
                    guidance_scale_2=3.5,
                    boundary_ratio=0.9,
                    flow_shift=3.0,
                    dimension_multiple=16,
                    image_resize="LANCZOS (official Omni example)",
                    init_timeout=7200,
                    stage_init_timeout=7200,
                    enable_diffusion_pipeline_profiler=False,
                ),
            },
        )

    def test_exact_published_scores_and_mean_absolute_percentage_points(self):
        # Literal decimal snapshots, not rounded display values or relative-percent errors.
        expected = {
            "t2v": {
                "bf16": "0.7218889263822397 0.6220035941311808 0.9801938021871339 0.23738084271775953 0.9515503867376934",
                "mxfp8": "0.7222442780911054 0.6210655156410102 0.9805028541362358 0.2380925087766214 0.9521171957626939",
                "nvfp4": "0.7193358409209418 0.6096992107954893 0.9809034917964539 0.23840135157559858 0.9474571227000067",
            },
            "i2v": {
                "bf16": "0.9775844285967266 0.9526026905976175 0.969535617653383 0.9845980721048039 0.9910735762508909",
                "mxfp8": "0.9756404829846045 0.9519102743258927 0.9688302976275803 0.9852716223834217 0.9899654466194023",
                "nvfp4": "0.9617349679147202 0.9363069311132639 0.9617780772917861 0.9832274752402079 0.982622336524479",
            },
        }
        means = {
            "t2v": {"mxfp8": "0.05761914464001140", "nvfp4": "0.41361862603670300"},
            "i2v": {"mxfp8": "0.1024672363951200", "nvfp4": "0.9944919423792960"},
        }
        for task, models in expected.items():
            result = read_json(REPO / "results" / f"{task}.json", parse_float=Decimal)
            self.assertEqual(set(result["models"]), set(models))
            metrics = benchmark.METRICS[task]
            baseline = result["models"]["bf16"]["scores"]
            for precision, values in models.items():
                scores = result["models"][precision]["scores"]
                self.assertEqual(scores, dict(zip(metrics, map(Decimal, values.split()))))
                if precision != "bf16":
                    mean = sum(abs(scores[m] - baseline[m]) * 100 for m in metrics) / len(metrics)
                    self.assertEqual(mean, Decimal(means[task][precision]))

    def test_prepare_is_idempotent_and_rejects_mismatched_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark.prepare(root, "t2v")
            before = (root / "jobs.json").read_bytes()
            benchmark.prepare(root, "t2v")
            self.assertEqual((root / "jobs.json").read_bytes(), before)
            record = read_json(root / "task.json")
            record["settings"]["seed"] = 999
            benchmark.atomic_json(root / "task.json", record)
            with self.assertRaisesRegex(RuntimeError, "mismatched preparation"):
                benchmark.prepare(root, "t2v")
            self.assertEqual((root / "jobs.json").read_bytes(), before)


class HelperTests(unittest.TestCase):
    def run_script(self, name, *args, **kwargs):
        env = dict(os.environ, PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        env.update(kwargs.pop("env", {}))
        return subprocess.run(
            ["bash", str(SCRIPTS / name), *map(str, args)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            **kwargs,
        )

    def test_shell_syntax_and_help(self):
        for name in ("setup_env.sh", "download_i2v.sh", "launch_workers.sh"):
            subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)
            self.assertEqual(self.run_script(name, "--help").returncode, 0)

    def test_launcher_rejects_duplicates_and_invalid_lists_without_creating_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            for gpus in ("0,0", "0,00", "", "0,", ",1", "-1", "0, 1", "GPU-uuid"):
                result = self.run_script("launch_workers.sh", "t2v", directory, "bf16", root, directory, gpus)
                self.assertNotEqual(result.returncode, 0, gpus)
                self.assertFalse(root.exists())

    def test_download_retains_zip_and_reports_resolved_images(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            with zipfile.ZipFile(dest / "crop.zip", "w") as archive:
                archive.writestr("crop/16-9/reference.jpg", b"fixture, not an image validation")
            result = self.run_script("download_i2v.sh", dest)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str((dest / "crop/16-9").resolve()))
            self.assertTrue((dest / "crop.zip").is_file())
            self.assertTrue((dest / "crop/16-9/reference.jpg").is_file())

    def test_download_uses_selected_interpreter_and_part_then_retains_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gdown.py").write_text(textwrap.dedent("""\
                import sys, zipfile
                from pathlib import Path
                assert sys.argv[1] == '1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ'
                assert sys.argv[2] == '-O'
                target = Path(sys.argv[3])
                assert target.name == 'crop.zip.part'
                with zipfile.ZipFile(target, 'w') as archive:
                    archive.writestr('crop/16-9/reference.jpg', b'fixture')
                """))
            dest = root / "external-inputs"
            result = self.run_script("download_i2v.sh", dest, env={"PYTHONPATH": str(root)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((dest / "crop.zip").is_file())
            self.assertFalse((dest / "crop.zip.part").exists())
            self.assertEqual(result.stdout.strip(), str((dest / "crop/16-9").resolve()))

    def test_setup_defaults_to_check_and_never_resets_mismatched_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            fake_bin = root / "bin"
            fake_bin.mkdir()
            git = fake_bin / "git"
            git.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
                import sys
                from pathlib import Path
                source = sys.argv[2]
                arguments = sys.argv[3:]
                if arguments == ['rev-parse', '--show-toplevel']:
                    print(source)
                elif arguments == ['rev-parse', 'HEAD']:
                    print('wrong-commit')
                else:
                    Path(source, 'unexpected-git-action').touch()
                    sys.exit(99)
                """))
            git.chmod(0o755)
            env_dir = root / "must-not-be-created"
            for mode in ([], ["--install"]):
                result = self.run_script(
                    "setup_env.sh",
                    *mode,
                    env_dir,
                    source,
                    env={"PATH": str(fake_bin) + os.pathsep + os.environ["PATH"]},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("commit mismatch", result.stderr)
                self.assertFalse(env_dir.exists())
                self.assertFalse((source / "unexpected-git-action").exists())
            missing = root / "missing-source"
            result = self.run_script("setup_env.sh", env_dir, missing)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("use --install", result.stderr)
            self.assertFalse(missing.exists())
            self.assertFalse(env_dir.exists())

    def test_download_rejects_zip_slip_and_symlink_members_before_extraction(self):
        for unsafe in ("../escape", "/absolute", "crop/../../escape", r"..\escape", "C:/escape", "link"):
            with self.subTest(member=unsafe), tempfile.TemporaryDirectory() as directory:
                dest = Path(directory)
                with zipfile.ZipFile(dest / "crop.zip", "w") as archive:
                    archive.writestr("crop/16-9/should-not-be-written.jpg", b"fixture")
                    if unsafe == "link":
                        member = zipfile.ZipInfo("link")
                        member.create_system = 3
                        member.external_attr = (stat.S_IFLNK | 0o777) << 16
                        archive.writestr(member, "../outside")
                    else:
                        archive.writestr(unsafe, b"fixture")
                result = self.run_script("download_i2v.sh", dest)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((dest / "crop").exists())

    def test_download_rejects_existing_directory_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest, outside = root / "inputs", root / "outside"
            dest.mkdir()
            outside.mkdir()
            (dest / "crop").symlink_to(outside, target_is_directory=True)
            with zipfile.ZipFile(dest / "crop.zip", "w") as archive:
                archive.writestr("crop/16-9/escape.jpg", b"fixture")
            result = self.run_script("download_i2v.sh", dest)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(outside.iterdir()), [])

    def test_launcher_waits_all_workers_records_exits_and_verifies_only_success(self):
        # Copy only the helper into a temporary harness; never load inference packages.
        stub = textwrap.dedent("""\
            import json, os, sys
            from pathlib import Path
            def atomic_json(path, value):
                Path(path).write_text(json.dumps(value))
            if __name__ == '__main__':
                command = sys.argv[1]
                root = Path(sys.argv[sys.argv.index('--root') + 1])
                with (root / 'calls.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(command=command, args=sys.argv[2:],
                        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        attention=os.environ.get('DIFFUSION_ATTENTION_BACKEND'))) + '\\n')
                if command == 'generate':
                    shard = sys.argv[sys.argv.index('--shard-index') + 1]
                    (root / ('shard-' + shard)).write_text('finished')
                    sys.exit(7 if os.environ.get('FAIL_SHARD') == shard else 0)
            """)
        for failure in ("", "0"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                helper = directory / "launch_workers.sh"
                helper.write_text((SCRIPTS / helper.name).read_text())
                (directory / "benchmark.py").write_text(stub)
                root = directory / "run"
                result = subprocess.run(
                    ["bash", str(helper), "t2v", str(directory), "nvfp4", str(root), str(directory), "2,5"],
                    env=dict(os.environ, PYTHON=sys.executable, FAIL_SHARD=failure, PYTHONDONTWRITEBYTECODE="1"),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 1 if failure else 0, result.stderr)
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                self.assertEqual(sum(c["command"] == "prepare" for c in calls), 1)
                self.assertEqual(sum(c["command"] == "verify" for c in calls), 0 if failure else 1)
                self.assertNotIn("evaluate", [c["command"] for c in calls])
                self.assertTrue(all(c["attention"] == "TRTLLM_ATTN" for c in calls))
                report = read_json(next((root / "logs").glob("workers-*.json")))
                self.assertEqual([w["gpu"] for w in report["workers"]], ["2", "5"])
                self.assertEqual([w["shard_index"] for w in report["workers"]], [0, 1])
                self.assertEqual([w["returncode"] for w in report["workers"]], [7 if failure else 0, 0])
                for worker in report["workers"]:
                    command = worker["command"]
                    self.assertEqual(command[command.index("--num-shards") + 1], "2")
                    self.assertTrue(Path(worker["log"]).is_file())
                    self.assertTrue((root / f"shard-{worker['shard_index']}").is_file())

    def test_launcher_signal_cleans_up_its_worker_and_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            helper = directory / "launch_workers.sh"
            helper.write_text((SCRIPTS / helper.name).read_text())
            child_code = textwrap.dedent("""\
                import os, signal, sys
                from pathlib import Path
                root = Path(sys.argv[1])
                def stop(number, frame):
                    (root / 'descendant-stopped').write_text(str(number))
                    with (root.parent / 'ready').open('w') as ready:
                        ready.write('stopped\\n')
                    sys.exit(0)
                signal.signal(signal.SIGTERM, stop)
                with (root.parent / 'ready').open('w') as ready:
                    ready.write(str(os.getpid()) + '\\n')
                signal.pause()
                """)
            stub = textwrap.dedent("""\
                import json, subprocess, sys, signal
                from pathlib import Path
                def atomic_json(path, value):
                    Path(path).write_text(json.dumps(value))
                if __name__ == '__main__' and sys.argv[1] == 'generate':
                    root = Path(sys.argv[sys.argv.index('--root') + 1])
                    subprocess.Popen([sys.executable, '-c', CHILD_CODE, str(root)])
                    signal.pause()
                """).replace("CHILD_CODE", repr(child_code))
            (directory / "benchmark.py").write_text(stub)
            fifo = directory / "ready"
            os.mkfifo(fifo)
            fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
            launcher = None
            child_pid = None
            root = directory / "run"
            try:
                launcher = subprocess.Popen(
                    ["bash", str(helper), "t2v", str(directory), "bf16", str(root), str(directory), "0"],
                    env=dict(os.environ, PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                with selectors.DefaultSelector() as selector:
                    selector.register(fd, selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=10), "Worker descendant did not become ready")
                child_pid = int(os.read(fd, 100).strip())
                launcher.send_signal(signal.SIGTERM)
                stdout, stderr = launcher.communicate(timeout=20)
                self.assertEqual(launcher.returncode, 143, (stdout, stderr))
                with selectors.DefaultSelector() as selector:
                    selector.register(fd, selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=5), "Descendant did not receive cleanup signal")
                self.assertEqual(os.read(fd, 100).strip(), b"stopped")
                report = read_json(next((root / "logs").glob("workers-*.json")))
                self.assertEqual(report["status"], "interrupted")
                self.assertEqual(report["signal"], signal.SIGTERM)
                self.assertEqual(report["workers"][0]["returncode"], -signal.SIGTERM)
                self.assertEqual((root / "descendant-stopped").read_text(), str(signal.SIGTERM))
            finally:
                os.close(fd)
                if launcher is not None and launcher.poll() is None:
                    launcher.kill()
                    launcher.communicate(timeout=5)
                # Only test-created PIDs, in case the assertion failed before cleanup.
                if child_pid is not None and not (root / "descendant-stopped").exists():
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
