"""Tests for common components.

!!! Please do not import any framework-specific modules in this file. !!!
* Note, we may need to add some auto check mechanisms to ensure this.

These tests aim to assess the fundamental functionalities of common utils and enhance code coverage.
All tests will be included for each framework CI.
"""

import unittest

from neural_compressor.common import options
from neural_compressor.common.utils import (
    dump_elapsed_time,
    set_random_seed,
    set_resume_from,
    set_tensorboard,
    set_workspace,
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

    @dump_elapsed_time("test_set_workspace")
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


if __name__ == "__main__":
    unittest.main()
