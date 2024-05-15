"""Tests for utility components."""

import unittest

from onnx_neural_compressor import utility


class TestOptions(unittest.TestCase):

    def test_set_random_seed(self):
        seed = 12345
        utility.set_random_seed(seed)
        self.assertEqual(utility.options.random_seed, seed)

        # non int type
        seed = "12345"
        with self.assertRaises(AssertionError):
            utility.set_random_seed(seed)

    def test_set_workspace(self):
        workspace = "/path/to/workspace"
        utility.set_workspace(workspace)
        self.assertEqual(utility.options.workspace, workspace)

        # non String type
        workspace = 12345
        with self.assertRaises(AssertionError):
            utility.set_workspace(workspace)

    def test_set_resume_from(self):
        resume_from = "/path/to/resume"
        utility.set_resume_from(resume_from)
        self.assertEqual(utility.options.resume_from, resume_from)

        # non String type
        resume_from = 12345
        with self.assertRaises(AssertionError):
            utility.set_resume_from(resume_from)


class TestCPUInfo(unittest.TestCase):

    def test_cpu_info(self):
        cpu_info = utility.CpuInfo()
        assert cpu_info.cores_per_socket > 0, "CPU count should be greater than 0"
        assert isinstance(cpu_info.bf16, bool), "bf16 should be a boolean"
        assert isinstance(cpu_info.vnni, bool), "avx512 should be a boolean"


class TestLazyImport(unittest.TestCase):

    def test_lazy_import(self):
        # Test import
        pydantic = utility.LazyImport("pydantic")
        assert pydantic.__name__ == "pydantic", "pydantic should be imported"

    def test_lazy_import_error(self):
        # Test import error
        with self.assertRaises(ImportError):
            non_existent_module = utility.LazyImport("non_existent_module")
            non_existent_module.non_existent_function()


class TestSingletonDecorator:

    def test_singleton_decorator(self):

        @utility.singleton
        class TestSingleton:

            def __init__(self):
                self.value = 0

        instance = TestSingleton()
        instance.value = 1
        instance2 = TestSingleton()
        assert instance2.value == 1, "Singleton should return the same instance"


if __name__ == "__main__":
    unittest.main()
