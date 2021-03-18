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
"""Utils test."""

import inspect
import os
import unittest

from lpot.ux.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    NotFoundException,
)
from lpot.ux.utils.utils import (
    check_module,
    get_dataset_path,
    get_file_extension,
    get_framework_from_path,
    get_predefined_config_path,
    is_dataset_file,
    is_development_env,
    is_hidden,
    is_model_file,
    load_dataloader_config,
    load_help_lpot_params,
    load_model_config,
    load_transforms_config,
    verify_file_path,
)


class TestUtils(unittest.TestCase):
    """Value parser tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)

    def test_is_hidden(self) -> None:
        """Test if path is correctly recognized as hidden."""
        path = ".ssh"
        result = is_hidden(path)
        self.assertEqual(result, True)

    def test_is_not_hidden(self) -> None:
        """Test if path is correctly recognized as not hidden."""
        path = "home"
        result = is_hidden(path)
        self.assertEqual(result, False)

    def test_get_dataset_path(self) -> None:
        """Test getting dataset path."""
        framework = "tensorflow"
        domain = "image_recognition"
        result = get_dataset_path(framework, domain)
        self.assertEqual(result, "examples/test/dataset/imagenet")

    def test_get_dataset_path_unknown_framework(self) -> None:
        """Test getting dataset path failure."""
        framework = "unknown"
        domain = "image_recognition"
        with self.assertRaises(Exception):
            get_dataset_path(framework, domain)

    def test_get_dataset_path_unknown_domain(self) -> None:
        """Test getting dataset path failure."""
        framework = "tensorflow"
        domain = "domain"
        with self.assertRaises(Exception):
            get_dataset_path(framework, domain)

    def test_get_tensorflow_framework_from_path(self) -> None:
        """Test getting framework name from path."""
        path = "/home/user/model.pb"
        result = get_framework_from_path(path)
        self.assertEqual(result, "tensorflow")

    def test_get_onnx_framework_from_path(self) -> None:
        """Test getting framework name from path."""
        path = "/home/user/model.onnx"
        result = get_framework_from_path(path)
        self.assertEqual(result, "onnxrt")

    def test_get_unknown_framework_from_path(self) -> None:
        """Test getting framework name from path."""
        path = "/home/user/model.some_extension"
        result = get_framework_from_path(path)
        self.assertIsNone(result)

    def test_get_file_extension(self) -> None:
        """Test getting file extension from path."""
        path = "/home/user/file.ext"
        result = get_file_extension(path)
        self.assertEqual(result, "ext")

    def test_get_file_with_dots_extension(self) -> None:
        """Test getting file extension from path."""
        path = "/home/user/file.name.ext2"
        result = get_file_extension(path)
        self.assertEqual(result, "ext2")

    def test_get_file_without_extension(self) -> None:
        """Test getting file extension from path."""
        path = "/home/user/file"
        result = get_file_extension(path)
        self.assertEqual(result, "")

    def test_pb_is_model_file(self) -> None:
        """Test if pb file is recognized correctly."""
        path = "/home/user/model.pb"
        result = is_model_file(path)
        self.assertEqual(result, True)

    def test_onnx_is_model_file(self) -> None:
        """Test if onnx file is recognized correctly."""
        path = "/home/user/model.onnx"
        result = is_model_file(path)
        self.assertEqual(result, True)

    def test_ckpt_is_model_file(self) -> None:
        """Test if ckpt file is recognized correctly."""
        path = "/home/user/model.onnx.ckpt"
        result = is_model_file(path)
        self.assertEqual(result, False)

    def test_mp3_is_model_file(self) -> None:
        """Test if mp3 file is recognized correctly."""
        path = "/home/user/favourite_song.mp3"
        result = is_model_file(path)
        self.assertEqual(result, False)

    def test_record_is_dataset_file(self) -> None:
        """Test if record is a valid dataset file."""
        path = "/home/user/dataset.record"
        result = is_dataset_file(path)
        self.assertEqual(result, True)

    def test_pbtxt_is_dataset_file(self) -> None:
        """Test if record is a valid dataset file."""
        path = "/home/user/dataset.pbtxt"
        result = is_dataset_file(path)
        self.assertEqual(result, False)

    def test_get_predefined_tf_image_recognition_config_path(self) -> None:
        """Test getting predefined config path for TF image recognition models."""
        self._assert_predefined_config_path(
            framework="tensorflow",
            domain="image_recognition",
        )

    def test_get_predefined_tf_object_detection_config_path(self) -> None:
        """Test getting predefined config path for TF object detection models."""
        self._assert_predefined_config_path(
            framework="tensorflow",
            domain="object_detection",
        )

    def test_get_predefined_tf_nlp_config_path(self) -> None:
        """Test getting predefined config path for TF NLP models."""
        self._assert_predefined_config_path(
            framework="tensorflow",
            domain="nlp",
        )

    def test_get_predefined_tf_recommendation_config_path(self) -> None:
        """Test getting predefined config path for TF recommendation models."""
        self._assert_predefined_config_path(
            framework="tensorflow",
            domain="recommendation",
        )

    def test_get_predefined_onnx_image_recognition_config_path(self) -> None:
        """Test getting predefined config path for onnx image recognition models."""
        self._assert_predefined_config_path(
            framework="onnxrt",
            domain="image_recognition",
        )

    def test_get_predefined_onnx_nlp_config_path(self) -> None:
        """Test getting predefined config path for onnx NLP models."""
        self._assert_predefined_config_path(
            framework="onnxrt",
            domain="nlp",
        )

    def test_get_predefined_config_path_framework_failure(self) -> None:
        """Test getting predefined config path for onnx NLP models."""
        with self.assertRaises(Exception):
            get_predefined_config_path(
                framework="onnx",
                domain="image_recognition",
            )

    def test_get_predefined_config_path_domain_failure(self) -> None:
        """Test getting predefined config path for onnx NLP models."""
        with self.assertRaises(Exception):
            get_predefined_config_path(
                framework="onnxrt",
                domain="object_detection",
            )

    def test_check_module(self) -> None:
        """Test checking existing module."""
        check_module("os")

    def test_check_non_existing_module(self) -> None:
        """Test checking non existing module."""
        with self.assertRaises(ClientErrorException):
            check_module("non_existing_module")

    def test_load_model_config(self) -> None:
        """Test getting models config."""
        result = load_model_config()
        self.assertIs(type(result), dict)
        self.assertIsNot(result, {})

    def test_load_dataloader_config(self) -> None:
        """Test getting dataloaders config."""
        result = load_dataloader_config()
        self.assertIs(type(result), list)
        self.assertIsNot(result, [])

    def test_load_transforms_config(self) -> None:
        """Test getting transforms config."""
        result = load_transforms_config()
        self.assertIs(type(result), list)
        self.assertIsNot(result, [])

    def test_load_metrics_help_lpot_params(self) -> None:
        """Test getting lpot metrics tooltips."""
        result = load_help_lpot_params("metrics")
        self.assertIs(type(result), dict)
        self.assertIsNot(result, {})

    def test_load_objectives_help_lpot_params(self) -> None:
        """Test getting lpot objectives tooltips."""
        result = load_help_lpot_params("objectives")
        self.assertIs(type(result), dict)
        self.assertIsNot(result, {})

    def test_load_strategies_help_lpot_params(self) -> None:
        """Test getting lpot strategies tooltips."""
        result = load_help_lpot_params("strategies")
        self.assertIs(type(result), dict)
        self.assertIsNot(result, {})

    def test_load_non_existing_help_lpot_params(self) -> None:
        """Test getting lpot strategies tooltips."""
        with self.assertRaises(FileNotFoundError):
            load_help_lpot_params("unknown_param")

    def verify_file_path(self) -> None:
        """Check if path can be accessed."""
        path = "/data"
        verify_file_path(path)

    def verify_root_path(self) -> None:
        """Check if path can be accessed."""
        path = "/"
        with self.assertRaises(AccessDeniedException):
            verify_file_path(path)

    def verify_hidden_path(self) -> None:
        """Check if path can be accessed."""
        path = "/home/user/.ssh/secret_key"
        with self.assertRaises(AccessDeniedException):
            verify_file_path(path)

    def verify_non_existing_path(self) -> None:
        """Check if path can be accessed."""
        path = "/some/non/existing/path"
        with self.assertRaises(NotFoundException):
            verify_file_path(path)

    def verify_restricted_path(self) -> None:
        """Check if path can be accessed."""
        path = "/usr"
        with self.assertRaises(NotFoundException):
            verify_file_path(path)

    def _assert_predefined_config_path(self, framework: str, domain: str) -> None:
        """Assert predefined config path."""
        result = get_predefined_config_path(framework, domain)
        expected = os.path.join(
            os.path.abspath(
                os.path.dirname(
                    inspect.getfile(get_predefined_config_path),
                ),
            ),
            "configs",
            "predefined_configs",
            f"{framework}",
            f"{domain}.yaml",
        )
        self.assertEqual(result, expected)
        self.assertEqual(os.path.isfile(result), True)

    def test_is_development_env(self) -> None:
        """Check if development env is activated."""
        os.environ.update({"LPOT_MODE": "development"})
        is_develop = is_development_env()
        self.assertTrue(is_develop)

    def test_is_production_env(self) -> None:
        """Check if production env is activated."""
        os.environ.update({"LPOT_MODE": "production"})
        is_develop = is_development_env()
        self.assertFalse(is_develop)

    def test_is_empty_lpot_mode_env(self) -> None:
        """Check if development env is activated."""
        if os.environ.get("LPOT_MODE", None) is not None:
            del os.environ["LPOT_MODE"]
        is_develop = is_development_env()
        self.assertFalse(is_develop)


if __name__ == "__main__":
    unittest.main()
