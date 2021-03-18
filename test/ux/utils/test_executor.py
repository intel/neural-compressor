# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Executor test."""

import os
import shutil
import unittest

from lpot.ux.utils.executor import Executor


class TestExecutor(unittest.TestCase):
    """Executor tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Executor tests constructor."""
        super().__init__(*args, **kwargs)
        self.executor = Executor(
            workspace_path="tmp_workdir",
            subject="test",
            data={"id": "abc", "some_key": "some_value"},
            send_response=False,
            log_name="my_log",
            additional_log_names=["additional_log1", "additional_log2"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down environment for test."""
        shutil.rmtree("tmp_workdir", ignore_errors=True)

    def test_workdir_property(self) -> None:
        """Test if workdir property returns correct path."""
        self.assertEqual(self.executor.workdir, "tmp_workdir")

    def test_request_id_property(self) -> None:
        """Test if request_id property returns correct value."""
        self.assertEqual(self.executor.request_id, "abc")

    def test_log_name_property(self) -> None:
        """Test if log_name property returns correct value."""
        self.assertEqual(self.executor.log_name, "my_log")

    def test_additional_log_names_property(self) -> None:
        """Test if additional_log_names property returns correct value."""
        self.assertIs(type(self.executor.additional_log_names), list)
        self.assertEqual(
            self.executor.additional_log_names,
            ["additional_log1", "additional_log2"],
        )

    def test_is_not_multi_commands(self) -> None:
        """Test if execution type is recognized correctly."""
        result = self.executor.is_multi_commands(["echo", "Hello world!"])
        self.assertFalse(result)

    def test_is_multi_commands(self) -> None:
        """Test if multi command execution is recognized correctly."""
        result = self.executor.is_multi_commands(
            [
                ["echo", "Hello"],
                ["echo", "world!"],
            ],
        )
        self.assertTrue(result)

    def test_process_call(self) -> None:
        """Test if multi command execution is recognized correctly."""
        print_phrase = "Hello world!"
        proc = self.executor.call(["echo", print_phrase])
        self.assertTrue(proc.is_ok)

        logs = self.executor.additional_log_names
        if self.executor.log_name is not None:
            logs.append(f"{self.executor.log_name}.txt")
        for log in logs:
            with open(os.path.join(self.executor.workdir, log), "r") as log_file:
                self.assertEqual(log_file.readline().rstrip("\n"), print_phrase)


if __name__ == "__main__":
    unittest.main()
