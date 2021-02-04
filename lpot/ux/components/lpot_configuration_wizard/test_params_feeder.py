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
"""Parameter feeder test module."""

import unittest

from lpot.ux.components.lpot_configuration_wizard.params_feeder import Feeder


class TestParamsFeeder(unittest.TestCase):
    """Parameter feeder test class."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize parameter feeder test class."""
        super().__init__(*args, **kwargs)

    def test_frameworks(self) -> None:
        """Test frameworks."""
        feeder = Feeder(
            {
                "param": "framework",
                "config": {},
            },
        )
        assert feeder.feed() == {"framework": ["pytorch", "tensorflow"]}

    def test_tensorflow_domains(self) -> None:
        """Test TensorFlow domains."""
        feeder = Feeder(
            {
                "param": "domain",
                "config": {
                    "framework": "tensorflow",
                },
            },
        )
        assert feeder.feed() == {
            "domain": [
                "image_recognition",
                "nlp",
                "object_detection",
                "recommendation",
            ],
        }

    def test_pytorch_domains(self) -> None:
        """Test Pytorch domains."""
        feeder = Feeder(
            {
                "param": "domain",
                "config": {
                    "framework": "pytorch",
                },
            },
        )
        assert feeder.feed() == {
            "domain": [
                "image_recognition",
                "nlp",
                "object_detection",
                "recommendation",
            ],
        }

    def test_tf_image_recognition_models(self) -> None:
        """Test TensorFlow image recognition models."""
        feeder = Feeder(
            {
                "param": "model",
                "config": {
                    "framework": "tensorflow",
                    "domain": "image_recognition",
                },
            },
        )
        assert feeder.feed() == {
            "model": [
                "inception_v1_slim",
                "inception_v2_slim",
                "inception_v3",
                "inception_v3_slim",
                "inception_v4",
                "inception_v4_slim",
                "mobilenetv1",
                "resnet_v1_50_slim",
                "resnet_v1_101_slim",
                "resnet_v1_152_slim",
                "resnet50_v1_5",
                "resnet50v1.0",
                "resnet101",
                "resnetv2_50_slim",
                "resnetv2_101_slim",
                "resnetv2_152_slim",
                "vgg16_slim",
                "vgg19_slim",
            ],
        }

    def test_tf_object_detection_models(self) -> None:
        """Test TensorFlow object detection models."""
        feeder = Feeder(
            {
                "param": "model",
                "config": {
                    "framework": "tensorflow",
                    "domain": "object_detection",
                },
            },
        )
        assert feeder.feed() == {
            "model": [
                "faster_rcnn_inception_resnet_v2",
                "faster_rcnn_resnet101",
                "mask_rcnn_inception_v2",
                "ssd_mobilenet_v1",
                "ssd_resnet50_v1",
            ],
        }

    def test_tf_recommendation_models(self) -> None:
        """Test TensorFlow recommendation models."""
        feeder = Feeder(
            {
                "param": "model",
                "config": {
                    "framework": "tensorflow",
                    "domain": "recommendation",
                },
            },
        )
        assert feeder.feed() == {"model": ["wide_deep_large_ds"]}

    def test_tf_nlp_models(self) -> None:
        """Test TensorFlow NLP models."""
        feeder = Feeder(
            {
                "param": "model",
                "config": {
                    "framework": "tensorflow",
                    "domain": "nlp",
                },
            },
        )
        assert feeder.feed() == {"model": ["bert"]}

    def test_pt_image_recognition_models(self) -> None:
        """Test Pytorch image recognition models."""
        feeder = Feeder(
            {
                "param": "model",
                "config": {
                    "framework": "pytorch",
                    "domain": "image_recognition",
                },
            },
        )
        assert feeder.feed() == {
            "model": [
                "inception_v3",
                "mobilenet_v2",
                "peleenet",
                "resnest50",
                "resnet18",
                "resnet18_qat",
                "resnet50",
                "resnet50_gpu",
                "resnet50_ipex",
                "resnet50_qat",
                "resnet101_qat",
                "resnext101_32x8d",
                "se_resnext50_32x4d",
            ],
        }

    def test_pt_object_detection_models(self) -> None:
        """Test Pytorch object detection models."""
        feeder = Feeder(
            {
                "param": "model",
                "config": {
                    "framework": "pytorch",
                    "domain": "object_detection",
                },
            },
        )
        assert feeder.feed() == {"model": ["yolo_v3"]}

    def test_pt_recommendation_models(self) -> None:
        """Test Pytorch recommendation models."""
        feeder = Feeder(
            {
                "param": "model",
                "config": {
                    "framework": "pytorch",
                    "domain": "recommendation",
                },
            },
        )
        assert feeder.feed() == {"model": ["dlrm"]}

    def test_pt_nlp_models(self) -> None:
        """Test Pytorch NLP models."""
        feeder = Feeder(
            {
                "param": "model",
                "config": {
                    "framework": "pytorch",
                    "domain": "nlp",
                },
            },
        )
        assert feeder.feed() == {"model": ["blendcnn"]}
