import contextlib
import copy
import os

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM

from neural_compressor.torch.algorithms.weight_only.hqq.auto_accelerator import auto_detect_accelerator
from neural_compressor.torch.quantization import HQQConfig, get_default_hqq_config, quantize


def override_envs(**kwargs):
    """Decorator to temporarily override environment variables before entering a function.

    # Example Usage:
    @override_envs(CUDA_VISIBLE_DEVICES="")
    def my_function():
        print("Environment variable MY_VAR:", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))

    # The decorator temporarily overrides MY_VAR for the duration of my_function
    my_function()

    # Outside the decorated function, MY_VAR is back to its original value
    print("Outside function MY_VAR:", os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))
    """

    def decorator(func):
        @contextlib.wraps(func)
        def wrapper(*args, **kwds):
            # Save the current environment variables
            original_envs = {key: os.environ.get(key) for key in kwargs}

            try:
                # Override environment variables with the provided values
                os.environ.update(kwargs)
                result = func(*args, **kwds)
            finally:
                # Revert environment variables to their original values
                for key, value in original_envs.items():
                    if value is not None:
                        os.environ[key] = value
                    else:
                        os.environ.pop(key, None)

            return result

        return wrapper

    return decorator


class TestHQQCPU:
    @override_envs(CUDA_VISIBLE_DEVICES="")
    def test_hqq_cpu(self):
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        example_inputs = torch.tensor(
            [[10, 20, 30, 40, 50, 60]], dtype=torch.long, device=auto_detect_accelerator().current_device()
        )
        if auto_detect_accelerator().name == "cpu":
            from neural_compressor.torch.algorithms.weight_only.hqq.config import hqq_global_option

            hqq_global_option.use_half = False
        quant_config = get_default_hqq_config()
        print(f"Current accelerator {auto_detect_accelerator().current_device()}")
        model = quantize(model, quant_config)
        q_label = model(example_inputs)[0]
        print(q_label)
