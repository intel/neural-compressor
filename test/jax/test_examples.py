import os
import subprocess
from pathlib import Path

import pytest

MODELS_PATH = os.environ.get("MODELS_PATH", "/tf_dataset/jax")
REPO_ROOT_PATH = f"{os.path.dirname(__file__)}/../.."
EXAMPLES_PATH = f"{REPO_ROOT_PATH}/examples/jax/keras"
# fmt: off
EXAMPLES = [
    # Helloworld ------------------------
    {
        "filepath": Path(f"{EXAMPLES_PATH}/helloworld.py")
    },
    # Simple model ----------------------
    {
        "filepath": Path(f"{EXAMPLES_PATH}/simple_model/simple_config.py")
    },
    {
        "filepath": Path(f"{EXAMPLES_PATH}/simple_model/composable_config.py")
    },
    {
        "filepath": Path(f"{EXAMPLES_PATH}/simple_model/external_config.py"),
        "args": [
            ["--quant_config_file", f"{EXAMPLES_PATH}/simple_model/configs/static_config.json"],
            ["--quant_config_file", f"{EXAMPLES_PATH}/simple_model/configs/dynamic_config.json"],
            ["--quant_config_file", f"{EXAMPLES_PATH}/simple_model/configs/composable_config.json"],
        ]
    },
    {
        "filepath": Path(f"{EXAMPLES_PATH}/simple_model/model_saving.py")
    },
    # Vit -------------------------------
    {
        "filepath": Path(f"{EXAMPLES_PATH}/vit/quantization.py"),
        "args": [
            [
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "fp8_e4m3"
            ],
            [
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "fp8_e5m2"
            ],
            [
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "int8"
            ],
        ]
    },
    {
        "filepath": Path(f"{EXAMPLES_PATH}/vit/prepare_static.py"),
        "args": [
            [
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "fp8_e4m3",
                "--quantized_path", "./vit_quantized_fp8_e4m3.keras"
            ],
            [
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "fp8_e5m2",
                "--quantized_path", "./vit_quantized_fp8_e5m2.keras"
            ],
            [
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "int8",
                "--quantized_path", "./vit_quantized_int8.keras"
            ],
        ]
    },
    {
        "filepath": Path(f"{EXAMPLES_PATH}/vit/use_static.py"),
        "args": [
            ["--quantized_path", "./vit_quantized_fp8_e4m3.keras"],
            ["--quantized_path", "./vit_quantized_fp8_e5m2.keras"],
            ["--quantized_path", "./vit_quantized_int8.keras"],
        ]
    },
    # Gemma -----------------------------
    {
        "filepath": Path(f"{EXAMPLES_PATH}/gemma/quantization.py"),
        "args": [
            [
                "--model_path", f"{MODELS_PATH}/gemma3_instruct_270m",
                "--precision", "fp8_e4m3"
            ],
            [
                "--model_path", f"{MODELS_PATH}/gemma3_instruct_270m",
                "--precision", "fp8_e5m2"
            ],
        ]
    },
    {
        "filepath": Path(f"{EXAMPLES_PATH}/gemma/prepare_static.py"),
        "args": [
            [
                "--model_path", f"{MODELS_PATH}/gemma3_instruct_270m",
                "--precision", "fp8_e4m3",
                "--quantized_path", "./gemma3_instruct_270m_quantized_fp8_e4m3.keras"
            ],
            [
                "--model_path", f"{MODELS_PATH}/gemma3_instruct_270m",
                "--precision", "fp8_e5m2",
                "--quantized_path", "./gemma3_instruct_270m_quantized_fp8_e5m2.keras"
            ],
        ]
    },
    {
        "filepath": Path(f"{EXAMPLES_PATH}/gemma/use_static.py"),
        "args": [
            ["--quantized_path", "./gemma3_instruct_270m_quantized_fp8_e4m3.keras"],
            ["--quantized_path", "./gemma3_instruct_270m_quantized_fp8_e5m2.keras"],
        ]
    },
]
# fmt: on


@pytest.mark.parametrize(
    "example",
    [example for example in EXAMPLES],
    ids=[f"{example['filepath'].parent.name}/{example['filepath'].name}" for example in EXAMPLES],
)
def test_example(example):
    if "args" in example:
        for args in example["args"]:
            process = subprocess.run(args=["python", example["filepath"], *args], cwd=example["filepath"].parent)
            assert process.returncode == 0, f"Example failed with args: {args}"
    else:
        process = subprocess.run(args=["python", example["filepath"]], cwd=example["filepath"].parent)
        assert process.returncode == 0, "No args example failed"
