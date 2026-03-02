import pytest


# Called once at the beginning of the test session
def pytest_sessionstart():
    import os

    import habana_frameworks.torch.core as htcore
    import torch

    htcore.hpu_set_env()

    # Use reproducible results
    torch.use_deterministic_algorithms(True)
    # Ensure that only 3x PyTorch part of INC is imported
    os.environ.setdefault("INC_PT_ONLY", "1")

    # Fix the seed - just in case
    torch.manual_seed(0)


@pytest.fixture
def inc_output_handler():
    """Fixture to handle the creation and deletion of output directory for INC measure or scale files."""
    from pathlib import Path
    from shutil import rmtree

    inc_output_dir_path = Path(__file__).parent.resolve() / Path("inc_output")
    inc_dump_stats_path = inc_output_dir_path / Path("measure")
    yield str(inc_dump_stats_path)  # return the path to the output directory to the test

    # Cleanup the output directory after tests
    if inc_output_dir_path.exists():
        rmtree(str(inc_output_dir_path))
