# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""Download config test."""

import unittest
from typing import List
from unittest.mock import MagicMock, patch

from neural_compressor.ux.components.model_zoo.list_models import get_available_models


@patch("sys.argv", ["inc_bench.py", "-p5000"])
class TestListingModels(unittest.TestCase):
    """DownloadConfig tests."""

    @patch("neural_compressor.ux.components.model_zoo.list_models.get_installed_frameworks")
    def test_get_available_tf_models(self, installed_frameworks_mock: MagicMock) -> None:
        """Test download_config."""
        installed_frameworks_mock.return_value = {
            "TensorFlow": "2.7",
        }

        actual: List[dict] = get_available_models()

        expected: List[dict] = [
            {
                "framework": "TensorFlow",
                "domain": "Image Recognition",
                "model": "inception_v3",
            },
            {
                "framework": "TensorFlow",
                "domain": "Image Recognition",
                "model": "inception_v4",
            },
            {
                "framework": "TensorFlow",
                "domain": "Image Recognition",
                "model": "mobilenetv1",
            },
            {
                "framework": "TensorFlow",
                "domain": "Image Recognition",
                "model": "resnet50_v1_5",
            },
            {
                "framework": "TensorFlow",
                "domain": "Image Recognition",
                "model": "resnet101",
            },
            {
                "framework": "TensorFlow",
                "domain": "Object Detection",
                "model": "faster_rcnn_inception_resnet_v2",
            },
            {
                "framework": "TensorFlow",
                "domain": "Object Detection",
                "model": "faster_rcnn_resnet101",
            },
            {
                "framework": "TensorFlow",
                "domain": "Object Detection",
                "model": "mask_rcnn_inception_v2",
            },
            {
                "framework": "TensorFlow",
                "domain": "Object Detection",
                "model": "ssd_mobilenet_v1",
            },
            {
                "framework": "TensorFlow",
                "domain": "Object Detection",
                "model": "ssd_resnet50_v1",
            },
        ]

        self.assertListEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
