import os
# Ensure that the HPU is in lazy mode and weight sharing is disabled
os.environ.setdefault("PT_HPU_LAZY_MODE", "1")
os.environ.setdefault("PT_HPU_WEIGHT_SHARING", "0")
# Ensure that only 3x of INC is imported
os.environ.setdefault("INC_3X_ONLY", "1")


def pytest_sessionstart():
    import torch
    import habana_frameworks.torch.core as htcore

    htcore.hpu_set_inference_env()
    torch.manual_seed(0)


def pytest_sessionfinish():
    import habana_frameworks.torch.core as htcore
    htcore.hpu_teardown_inference_env()
