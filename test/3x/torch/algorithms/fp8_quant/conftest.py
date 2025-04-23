# Called once at the beginning of the test session
def pytest_sessionstart():
    import os
    import habana_frameworks.torch.core as htcore
    import torch

    htcore.hpu_set_env()
    os.environ.setdefault("INC_3X_ONLY", "1")

    # Use reproducible results
    torch.use_deterministic_algorithms(True)

    # Fix the seed - just in case
    torch.manual_seed(0)
