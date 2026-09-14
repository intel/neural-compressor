import os
import subprocess
import tempfile
from pathlib import Path

import pytest

TMP = tempfile.TemporaryDirectory()
TMP_PATH = Path(TMP.name)
MODELS_PATH = os.environ.get("MODELS_PATH", "/tf_dataset/jax")
REPO_ROOT_PATH = Path(os.path.dirname(__file__)).resolve().parents[1]
EXAMPLES_PATH = f"{REPO_ROOT_PATH}/examples/jax/keras"
# fmt: off
EXAMPLES = [
    # Helloworld ------------------------
    {
        "test_case": "helloworld",
        "args": [
            [ Path(f"{EXAMPLES_PATH}/helloworld.py") ]
        ]
    },
    # Simple model ----------------------
    {
        "test_case": "simple_model/simple_config.py",
        "args": [
            [ Path(f"{EXAMPLES_PATH}/simple_model/simple_config.py") ]
        ]
    },
    {
        "test_case": "simple_model/composable_config.py",
        "args": [
            [ Path(f"{EXAMPLES_PATH}/simple_model/composable_config.py") ]
        ]
    },
    {
        "test_case": "simple_model/external_config.py",
        "args": [
            [
                Path(f"{EXAMPLES_PATH}/simple_model/external_config.py"),
                "--quant_config_file", f"{EXAMPLES_PATH}/simple_model/configs/static_config.json"
            ],
            [
                Path(f"{EXAMPLES_PATH}/simple_model/external_config.py"),
                "--quant_config_file", f"{EXAMPLES_PATH}/simple_model/configs/dynamic_config.json"
            ],
            [
                Path(f"{EXAMPLES_PATH}/simple_model/external_config.py"),
                "--quant_config_file", f"{EXAMPLES_PATH}/simple_model/configs/composable_config.json"
            ],
        ]
    },
    {
        "test_case": "simple_model/model_saving.py",
        "args": [
            [ Path(f"{EXAMPLES_PATH}/simple_model/model_saving.py") ]
        ]
    },
    # Vit -------------------------------
    {
        "test_case": "vit/quantization.py",
        "args": [
            [
                Path(f"{EXAMPLES_PATH}/vit/quantization.py"),
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "fp8_e5m2"
            ],
            [
                Path(f"{EXAMPLES_PATH}/vit/quantization.py"),
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "int8"
            ],
        ],
    },
    {
        "test_case": "vit/prepare_static.py-use_static.py",
        "args": [
            [
                Path(f"{EXAMPLES_PATH}/vit/prepare_static.py"),
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "fp8_e4m3",
                "--quantized_path", f"{TMP_PATH}/vit_quantized.keras"
            ],
            [
                Path(f"{EXAMPLES_PATH}/vit/use_static.py"),
                "--quantized_path", f"{TMP_PATH}/vit_quantized.keras"
            ],
            [
                Path(f"{EXAMPLES_PATH}/vit/prepare_static.py"),
                "--model_path", f"{MODELS_PATH}/vit_base_patch16_224_imagenet",
                "--precision", "int8",
                "--quantized_path", f"{TMP_PATH}/vit_quantized"
            ],
            [
                Path(f"{EXAMPLES_PATH}/vit/use_static.py"),
                "--quantized_path", f"{TMP_PATH}/vit_quantized"
            ],
        ]
    },
    # Gemma -----------------------------
    {
        "test_case": "gemma/quantization.py",
        "args": [
            [
                Path(f"{EXAMPLES_PATH}/gemma/quantization.py"),
                "--model_path", f"{MODELS_PATH}/gemma3_instruct_270m",
                "--precision", "fp8_e4m3"
            ]
        ]
    },
    {
        "test_case": "gemma/prepare_static.py-use_static.py",
        "args": [
            [
                Path(f"{EXAMPLES_PATH}/gemma/prepare_static.py"),
                "--model_path", f"{MODELS_PATH}/gemma3_instruct_270m",
                "--precision", "fp8_e5m2",
                "--quantized_path", f"{TMP_PATH}/gemma3_instruct_270m_quantized"
            ],
            [
                Path(f"{EXAMPLES_PATH}/gemma/use_static.py"),
                "--quantized_path", f"{TMP_PATH}/gemma3_instruct_270m_quantized"
            ]
        ]
    },
]
# fmt: on


@pytest.mark.parametrize(
    "example", [example for example in EXAMPLES], ids=[example["test_case"] for example in EXAMPLES]
)
def test_example(example):
    for args in example["args"]:
        process = subprocess.run(args=["python", *args], cwd=args[0].parent)
        assert process.returncode == 0, f"Example failed with args: {args}"
