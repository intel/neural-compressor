"""Tests for common components.

!!! Please do not import any framework-specific modules in this file. !!!
* Note, we may need to add some auto check mechanisms to ensure this.

These tests aim to assess the fundamental functionalities of common utils and enhance code coverage.
All tests will be included for each framework CI.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

import neural_compressor.common.utils.utility as inc_utils
from neural_compressor.common import options
from neural_compressor.common.utils import (
    CpuInfo,
    LazyImport,
    Mode,
    default_tuning_logger,
    dump_elapsed_time,
    log_process,
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

    def test_lazy_import_access_attr(self):
        module_name = "neural_compressor.common.utils.utility"
        lazy_import = LazyImport(module_name)
        self.assertIsNone(lazy_import.module)

        with patch("importlib.import_module") as mock_import_module:
            module = MagicMock()
            mock_import_module.return_value = module

            # Test accessing attributes
            attribute = lazy_import.attribute_name
            self.assertEqual(attribute, module.attribute_name)
            mock_import_module.assert_called_once_with(module_name)

            # Test calling functions
            function = lazy_import.function_name()
            self.assertEqual(function, module.function_name())
            mock_import_module.assert_called_with(module_name)

        self.assertIsNotNone(lazy_import.module)


class TestUtils(unittest.TestCase):
    def test_dump_elapsed_time(self):
        @dump_elapsed_time("test function")
        def test_function():
            time.sleep(1)
            return True

        with patch("neural_compressor.common.utils.utility.logger") as mock_logger:
            test_function()
            mock_logger.info.assert_called()
            # Extract the actual log message
            log_message = mock_logger.info.call_args[0][0]
            # Check that the log message contains the expected parts
            self.assertIn("test function elapsed time:", log_message)


class TestLogProcess(unittest.TestCase):
    def test_log_process_wrapper(self):
        @log_process(mode=Mode.QUANTIZE)
        def test_function():
            return True

        with patch.object(default_tuning_logger, "execution_start") as mock_start_log, patch.object(
            default_tuning_logger, "execution_end"
        ) as mock_end_log:
            test_function()
            mock_start_log.assert_called_with(mode=Mode.QUANTIZE, stacklevel=4)
            mock_end_log.assert_called_with(mode=Mode.QUANTIZE, stacklevel=4)

    def test_inner_wrapper(self):
        def test_function():
            return True

        with patch.object(default_tuning_logger, "execution_start") as mock_start_log, patch.object(
            default_tuning_logger, "execution_end"
        ) as mock_end_log:
            inner_wrapper = log_process(mode=Mode.QUANTIZE)(test_function)
            inner_wrapper()
            mock_start_log.assert_called_with(mode=Mode.QUANTIZE, stacklevel=4)
            mock_end_log.assert_called_with(mode=Mode.QUANTIZE, stacklevel=4)


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


class TestCallCounter(unittest.TestCase):
    def test_call_counter(self):
        # empty dict
        inc_utils.FUNC_CALL_COUNTS.clear()

        @inc_utils.call_counter
        def add(a, b):
            return a + b

        # Initial count should be 0
        self.assertEqual(inc_utils.FUNC_CALL_COUNTS["add"], 0)

        # Call the function multiple times
        add(1, 2)
        add(3, 4)
        add(5, 6)

        # Count should be incremented accordingly
        self.assertEqual(inc_utils.FUNC_CALL_COUNTS["add"], 3)


if __name__ == "__main__":
    unittest.main()
