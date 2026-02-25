import os

# Ensure that the HPU is in lazy mode and weight sharing is disabled
os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
os.environ.setdefault("PT_HPU_WEIGHT_SHARING", "0")
# Ensure that only 3x PyTorch part of INC is imported
os.environ.setdefault("INC_PT_ONLY", "1")


def pytest_sessionstart():
    import habana_frameworks.torch.core as htcore
    import torch

    htcore.hpu_set_inference_env()
    torch.manual_seed(0)


def pytest_sessionfinish():
    import habana_frameworks.torch.core as htcore

    htcore.hpu_teardown_inference_env()
