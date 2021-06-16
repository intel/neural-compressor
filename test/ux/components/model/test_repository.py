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
"""Test Edge."""

import unittest

from lpot.ux.components.model.repository import ModelRepository


class TestModelRepository(unittest.TestCase):
    """Test ModelRepository class."""

    def test_onnx_is_model_path(self) -> None:
        """Test if onnx file is recognized correctly."""
        path = "/home/user/model.onnx"
        result = ModelRepository.is_model_path(path)
        self.assertTrue(result)

    def test_ckpt_is_model_path(self) -> None:
        """Test if ckpt file is recognized correctly."""
        path = "/home/user/model.onnx.ckpt"
        result = ModelRepository.is_model_path(path)
        self.assertFalse(result)

    def test_mp3_is_model_path(self) -> None:
        """Test if mp3 file is recognized correctly."""
        path = "/home/user/favourite_song.mp3"
        result = ModelRepository.is_model_path(path)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
