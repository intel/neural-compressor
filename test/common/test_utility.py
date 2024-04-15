"""Tests for common components.

!!! Please do not import any framework-specific modules in this file. !!!
* Note, we may need to add some auto check mechanisms to ensure this.

These tests aim to assess the fundamental functionalities of common utils and enhance code coverage.
All tests will be included for each framework CI.
"""

import unittest

from neural_compressor_ort.common import options
from neural_compressor_ort.common.utils import (
    CpuInfo,
    LazyImport,
    set_random_seed,
    set_resume_from,
    set_tensorboard,
    set_workspace,
    singleton,
)


class TestOptions(unittest.TestCase):
    def test_set_random_seed(self):
        seed = 12345
        set_random_seed(seed)
        self.assertEqual(options.random_seed, seed)

        # non int type
        seed = "12345"
        with self.assertRaises(AssertionError):
            set_random_seed(seed)

    def test_set_workspace(self):
        workspace = "/path/to/workspace"
        set_workspace(workspace)
        self.assertEqual(options.workspace, workspace)

        # non String type
        workspace = 12345
        with self.assertRaises(AssertionError):
            set_workspace(workspace)

    def test_set_resume_from(self):
        resume_from = "/path/to/resume"
        set_resume_from(resume_from)
        self.assertEqual(options.resume_from, resume_from)

        # non String type
        resume_from = 12345
        with self.assertRaises(AssertionError):
            set_resume_from(resume_from)

    def test_set_tensorboard(self):
        tensorboard = True
        set_tensorboard(tensorboard)
        self.assertEqual(options.tensorboard, tensorboard)

        # non bool type
        tensorboard = 123
        with self.assertRaises(AssertionError):
            set_tensorboard(tensorboard)


class TestCPUInfo(unittest.TestCase):
    def test_cpu_info(self):
        cpu_info = CpuInfo()
        assert cpu_info.cores_per_socket > 0, "CPU count should be greater than 0"
        assert isinstance(cpu_info.bf16, bool), "bf16 should be a boolean"
        assert isinstance(cpu_info.vnni, bool), "avx512 should be a boolean"


class TestLazyImport(unittest.TestCase):
    def test_lazy_import(self):
        # Test import
        pydantic = LazyImport("pydantic")
        assert pydantic.__name__ == "pydantic", "pydantic should be imported"

    def test_lazy_import_error(self):
        # Test import error
        with self.assertRaises(ImportError):
            non_existent_module = LazyImport("non_existent_module")
            non_existent_module.non_existent_function()


class TestSingletonDecorator:
    def test_singleton_decorator(self):
        @singleton
        class TestSingleton:
            def __init__(self):
                self.value = 0

        instance = TestSingleton()
        instance.value = 1
        instance2 = TestSingleton()
        assert instance2.value == 1, "Singleton should return the same instance"


if __name__ == "__main__":
    unittest.main()
