# Called once at the beginning of the test session
def pytest_sessionstart():
    import habana_frameworks.torch.core as htcore
    import torch

    htcore.hpu_set_env()

    # Use reproducible results
    torch.use_deterministic_algorithms(True)

    # Fix the seed - just in case
    torch.manual_seed(0)
