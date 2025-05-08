import os

os.environ.setdefault("PT_HPU_LAZY_MODE", "1")

from .tester import run_accuracy_test, TestVector

__all__ = [
    "run_accuracy_test",
    "TestVector",
]
