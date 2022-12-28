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
from typing import Any, List
from unittest.mock import MagicMock, patch

from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.utils.consts import DomainFlavours, Domains, Frameworks
from neural_compressor.ux.utils.exceptions import (
    AccessDeniedException,
    ClientErrorException,
    InternalException,
    NotFoundException,
)
from neural_compressor.ux.utils.utils import (
    check_module,
    get_dataset_path,
    get_file_extension,
    get_framework_from_path,
    get_metrics_dict,
    get_predefined_config_path,
    is_dataset_file,
    is_development_env,
    is_hidden,
    load_dataloader_config,
    load_help_nc_params,
    load_model_config,
    load_model_wise_params,
    load_transforms_config,
    normalize_string,
    parse_to_string_list,
    parse_version,
    release_tag,
    verify_file_path,
)

fake_metrics: dict = {
    "topk": {},
    "COCOmAP": {},
    "MSE": {},
    "RMSE": {},
    "MAE": {},
    "metric1": {},
}


class TestUtils(unittest.TestCase):
    """Value parser tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)

    def test_is_hidden(self) -> None:
        """Test if path is correctly recognized as hidden."""
        path = ".ssh"
        result = is_hidden(path)
        self.assertTrue(result)

    def test_is_not_hidden(self) -> None:
        """Test if path is correctly recognized as not hidden."""
        path = "home"
        result = is_hidden(path)
        self.assertFalse(result)

    def test_get_dataset_path(self) -> None:
        """Test getting dataset path."""
        framework = Frameworks.TF.value
        domain = Domains.IMAGE_RECOGNITION.value
        result = get_dataset_path(framework, domain)
        self.assertEqual(result, "examples/test/dataset/imagenet")

    def test_get_dataset_path_unknown_framework(self) -> None:
        """Test getting dataset path failure."""
        framework = "unknown"
        domain = Domains.IMAGE_RECOGNITION.value
        with self.assertRaises(Exception):
            get_dataset_path(framework, domain)

    def test_get_dataset_path_unknown_domain(self) -> None:
        """Test getting dataset path failure."""
        framework = Frameworks.TF.value
        domain = "domain"
        with self.assertRaises(Exception):
            get_dataset_path(framework, domain)

    @patch("neural_compressor.ux.components.model.tensorflow.frozen_pb.get_model_type")
    def test_get_tensorflow_framework_from_path(self, mocked_get_model_type: MagicMock) -> None:
        """Test getting framework name from path."""
        mocked_get_model_type.return_value = "frozen_pb"
        path = "/home/user/model.pb"
        result = get_framework_from_path(path)
        self.assertEqual(result, Frameworks.TF.value)
        mocked_get_model_type.assert_called_with(path)

    def test_get_onnx_framework_from_path(self) -> None:
        """Test getting framework name from path."""
        path = "/home/user/model.onnx"
        result = get_framework_from_path(path)
        self.assertEqual(result, Frameworks.ONNX.value)

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

    def test_record_is_dataset_file(self) -> None:
        """Test if record is a valid dataset file."""
        path = "/home/user/dataset.record"
        result = is_dataset_file(path)
        self.assertTrue(result)

    def test_pbtxt_is_dataset_file(self) -> None:
        """Test if record is a valid dataset file."""
        path = "/home/user/dataset.pbtxt"
        result = is_dataset_file(path)
        self.assertFalse(result)

    def test_get_predefined_tf_image_recognition_config_path(self) -> None:
        """Test getting predefined config path for TF image recognition models."""
        self._assert_predefined_config_path(
            framework=Frameworks.TF.value,
            domain=Domains.IMAGE_RECOGNITION.value,
            domain_flavour=DomainFlavours.NONE.value,
            expected_filename="image_recognition.yaml",
        )

    def test_get_predefined_tf_object_detection_config_path(self) -> None:
        """Test getting predefined config path for TF object detection models."""
        self._assert_predefined_config_path(
            framework=Frameworks.TF.value,
            domain=Domains.OBJECT_DETECTION.value,
            domain_flavour=DomainFlavours.NONE.value,
            expected_filename="object_detection.yaml",
        )

    def test_get_predefined_tf_object_detection_ssd_config_path(self) -> None:
        """Test getting predefined config path for TF object detection models."""
        self._assert_predefined_config_path(
            framework=Frameworks.TF.value,
            domain=Domains.OBJECT_DETECTION.value,
            domain_flavour=DomainFlavours.SSD.value,
            expected_filename="object_detection_ssd.yaml",
        )

    def test_get_predefined_tf_object_detection_unknown_flavour_config_path(self) -> None:
        """Test getting predefined config path for TF object detection models."""
        self._assert_predefined_config_path(
            framework=Frameworks.TF.value,
            domain=Domains.OBJECT_DETECTION.value,
            domain_flavour="foo",
            expected_filename="object_detection.yaml",
        )

    def test_get_predefined_tf_nlp_config_path(self) -> None:
        """Test getting predefined config path for TF NLP models."""
        self._assert_predefined_config_path(
            framework=Frameworks.TF.value,
            domain=Domains.NLP.value,
            domain_flavour=DomainFlavours.NONE.value,
            expected_filename="nlp.yaml",
        )

    def test_get_predefined_tf_recommendation_config_path(self) -> None:
        """Test getting predefined config path for TF recommendation models."""
        self._assert_predefined_config_path(
            framework=Frameworks.TF.value,
            domain=Domains.RECOMMENDATION.value,
            domain_flavour=DomainFlavours.NONE.value,
            expected_filename="recommendation.yaml",
        )

    def test_get_predefined_onnx_image_recognition_config_path(self) -> None:
        """Test getting predefined config path for onnx image recognition models."""
        self._assert_predefined_config_path(
            framework=Frameworks.ONNX.value,
            domain=Domains.IMAGE_RECOGNITION.value,
            domain_flavour=DomainFlavours.NONE.value,
            expected_filename="image_recognition.yaml",
        )

    def test_get_predefined_onnx_object_detection_config_path(self) -> None:
        """Test getting predefined config path for onnx object detection models."""
        self._assert_predefined_config_path(
            framework=Frameworks.ONNX.value,
            domain=Domains.OBJECT_DETECTION.value,
            domain_flavour=DomainFlavours.NONE.value,
            expected_filename="object_detection.yaml",
        )

    def test_get_predefined_onnx_nlp_config_path(self) -> None:
        """Test getting predefined config path for onnx NLP models."""
        self._assert_predefined_config_path(
            framework=Frameworks.ONNX.value,
            domain=Domains.NLP.value,
            domain_flavour=DomainFlavours.NONE.value,
            expected_filename="nlp.yaml",
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
                framework=Frameworks.ONNX.value,
                domain="recommendation",
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

    def test_load_metrics_help_nc_params(self) -> None:
        """Test getting neural_compressor metrics tooltips."""
        result = load_help_nc_params("metrics")
        self.assertIs(type(result), dict)
        self.assertIsNot(result, {})

    def test_load_objectives_help_nc_params(self) -> None:
        """Test getting neural_compressor objectives tooltips."""
        result = load_help_nc_params("objectives")
        self.assertIs(type(result), dict)
        self.assertIsNot(result, {})

    def test_load_strategies_help_nc_params(self) -> None:
        """Test getting neural_compressor strategies tooltips."""
        result = load_help_nc_params("strategies")
        self.assertIs(type(result), dict)
        self.assertIsNot(result, {})

    def test_load_non_existing_help_nc_params(self) -> None:
        """Test getting neural_compressor strategies tooltips."""
        with self.assertRaises(FileNotFoundError):
            load_help_nc_params("unknown_param")

    def test_load_common_model_wise_params(self) -> None:
        """Test getting common model wise parameters."""
        framework = "non existing framework"
        result = load_model_wise_params(framework)

        expected = {
            "model_wise": {
                "weight": {
                    "granularity": ["per_channel", "per_tensor"],
                    "scheme": ["asym", "sym"],
                    "dtype": ["int8", "uint8", "fp32", "bf16", "fp16"],
                    "algorithm": ["minmax"],
                    "bit": 7.0,
                },
                "activation": {
                    "granularity": ["per_channel", "per_tensor"],
                    "scheme": ["asym", "sym"],
                    "dtype": ["int8", "uint8", "fp32", "bf16", "fp16"],
                    "algorithm": ["minmax", "kl"],
                },
            },
        }

        self.assertDictEqual(result, expected)

    def test_load_pytorch_model_wise_params(self) -> None:
        """Test getting pytorch model wise parameters."""
        framework = "pytorch"
        result = load_model_wise_params(framework)

        expected = {
            "model_wise": {
                "weight": {
                    "granularity": ["per_channel", "per_tensor"],
                    "scheme": ["asym", "sym", "asym_float"],
                    "dtype": ["int8", "uint8", "fp32", "bf16", "fp16"],
                    "algorithm": ["minmax"],
                    "bit": 7.0,
                },
                "activation": {
                    "granularity": ["per_channel", "per_tensor"],
                    "scheme": ["asym", "sym"],
                    "dtype": ["int8", "uint8", "fp32", "bf16", "fp16"],
                    "algorithm": ["minmax", "kl", "placeholder"],
                    "compute_dtype": ["int8", "uint8", "fp32", "bf16", "None"],
                },
            },
        }

        self.assertDictEqual(result, expected)

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

    def _assert_predefined_config_path(
        self,
        framework: str,
        domain: str,
        domain_flavour: str,
        expected_filename: str,
    ) -> None:
        """Assert predefined config path."""
        result = get_predefined_config_path(framework, domain, domain_flavour)
        names_mapper = NamesMapper(MappingDirection.ToCore)
        mapped_framework = names_mapper.map_name("framework", framework)
        expected = os.path.join(
            os.path.abspath(
                os.path.dirname(
                    inspect.getfile(get_predefined_config_path),
                ),
            ),
            "configs",
            "predefined_configs",
            f"{mapped_framework}",
            expected_filename,
        )
        self.assertEqual(result, expected)
        self.assertEqual(os.path.isfile(result), True)

    def test_is_development_env(self) -> None:
        """Check if development env is activated."""
        os.environ.update({"NC_MODE": "development"})
        is_develop = is_development_env()
        self.assertTrue(is_develop)

    def test_is_production_env(self) -> None:
        """Check if production env is activated."""
        os.environ.update({"NC_MODE": "production"})
        is_develop = is_development_env()
        self.assertFalse(is_develop)

    def test_is_empty_nc_mode_env(self) -> None:
        """Check if development env is activated."""
        if os.environ.get("NC_MODE", None) is not None:
            del os.environ["NC_MODE"]
        is_develop = is_development_env()
        self.assertFalse(is_develop)

    @patch("neural_compressor.ux.utils.utils.nc_version", "3.14.15")
    def test_release_tag(self) -> None:
        """Test release tag building."""
        self.assertEqual("v3.14.15", release_tag())

    @patch("neural_compressor.ux.utils.utils.nc_version", "42.12dev20200102.foo")
    def test_release_tag_for_dev_version(self) -> None:
        """Test release tag building."""
        self.assertEqual("v42.12", release_tag())

    @patch("neural_compressor.ux.utils.utils.nc_version", "")
    def test_release_tag_for_empty(self) -> None:
        """Test release tag building."""
        with self.assertRaisesRegexp(ValueError, "Unable to parse version "):
            release_tag()

    @patch("neural_compressor.ux.utils.utils.nc_version", "foo.bar.baz")
    def test_release_tag_for_invalid_version(self) -> None:
        """Test release tag building."""
        with self.assertRaisesRegexp(ValueError, "Unable to parse version foo.bar.ba"):
            release_tag()

    @patch(
        "neural_compressor.ux.utils.utils.registry_metrics",
        {"tensorflow": fake_metrics},
    )
    @patch(
        "neural_compressor.ux.utils.utils.WORKSPACE_LOCATION",
        os.path.join("foo", "bar", "workdir"),
    )
    def test_get_tensorflow_metrics_dict(self) -> None:
        """Test getting metrics dict."""
        self.maxDiff = None

        expected = {
            "tensorflow": [
                {
                    "name": "topk",
                    "help": "",
                    "params": [
                        {
                            "name": "k",
                            "help": "Number of top elements to look at for computing accuracy",
                            "value": [1, 5],
                        },
                    ],
                },
                {
                    "name": "COCOmAP",
                    "help": "",
                    "params": [
                        {
                            "name": "anno_path",
                            "help": "annotation path",
                            "value": os.path.join("foo", "bar", "workdir", "label_map.yaml"),
                            "label": "annotation path",
                        },
                    ],
                },
                {
                    "name": "MSE",
                    "help": "",
                    "params": [
                        {
                            "name": "compare_label",
                            "help": "",
                            "value": True,
                        },
                    ],
                },
                {
                    "name": "RMSE",
                    "help": "",
                    "params": [
                        {
                            "name": "compare_label",
                            "help": "",
                            "value": True,
                        },
                    ],
                },
                {
                    "name": "MAE",
                    "help": "",
                    "params": [
                        {
                            "name": "compare_label",
                            "help": "",
                            "value": True,
                        },
                    ],
                },
                {
                    "name": "metric1",
                    "help": "",
                    "value": None,
                },
                {
                    "name": "custom",
                    "help": "Create your own metric. After you choose this option code "
                    "template will be generated on the server",
                    "value": None,
                },
            ],
        }

        actual = get_metrics_dict()

        self.assertEqual(expected, actual)

    @patch(
        "neural_compressor.ux.utils.utils.registry_metrics",
        {"onnxrt_qlinearops": fake_metrics},
    )
    @patch(
        "neural_compressor.ux.utils.utils.WORKSPACE_LOCATION",
        os.path.join("foo", "bar", "workdir"),
    )
    def test_get_onnx_metrics_dict(self) -> None:
        """Test getting metrics dict."""
        expected = {
            "onnxrt": [
                {
                    "name": "topk",
                    "help": "",
                    "params": [
                        {
                            "name": "k",
                            "help": "Number of top elements to look at for computing accuracy",
                            "value": [1, 5],
                        },
                    ],
                },
                {
                    "name": "COCOmAP",
                    "help": "",
                    "params": [
                        {
                            "name": "anno_path",
                            "help": "annotation path",
                            "value": os.path.join("foo", "bar", "workdir", "label_map.yaml"),
                            "label": "annotation path",
                        },
                    ],
                },
                {
                    "name": "MSE",
                    "help": "",
                    "params": [
                        {
                            "name": "compare_label",
                            "help": "",
                            "value": True,
                        },
                    ],
                },
                {
                    "name": "RMSE",
                    "help": "",
                    "params": [
                        {
                            "name": "compare_label",
                            "help": "",
                            "value": True,
                        },
                    ],
                },
                {
                    "name": "MAE",
                    "help": "",
                    "params": [
                        {
                            "name": "compare_label",
                            "help": "",
                            "value": True,
                        },
                    ],
                },
                {
                    "name": "metric1",
                    "help": "",
                    "value": None,
                },
                {
                    "name": "custom",
                    "help": "Create your own metric. After you choose this option code "
                    "template will be generated on the server",
                    "value": None,
                },
            ],
        }

        actual = get_metrics_dict()

        self.assertEqual(expected, actual)

    @patch(
        "neural_compressor.ux.utils.utils.registry_metrics",
        {
            "tensorflow": {"topk": None, "COCOmAP": None},
        },
    )
    @patch(
        "neural_compressor.ux.utils.utils.WORKSPACE_LOCATION",
        os.path.join("foo", "bar", "workdir"),
    )
    def test_get_metrics_with_label(self) -> None:
        """Test that get_metrics can correctly set label map file location."""
        expected = {
            "tensorflow": [
                {
                    "name": "topk",
                    "help": "",
                    "params": [
                        {
                            "name": "k",
                            "help": "Number of top elements to look at for computing accuracy",
                            "value": [1, 5],
                        },
                    ],
                },
                {
                    "name": "COCOmAP",
                    "help": "",
                    "params": [
                        {
                            "name": "anno_path",
                            "help": "annotation path",
                            "value": os.path.join("foo", "bar", "workdir", "label_map.yaml"),
                            "label": "annotation path",
                        },
                    ],
                },
                {
                    "name": "custom",
                    "help": "Create your own metric. After you choose this option code "
                    "template will be generated on the server",
                    "value": None,
                },
            ],
        }

        actual = get_metrics_dict()

        self.assertEqual(expected, actual)

    def test_parse_none_nodes(self) -> None:
        """Test parsing None to empty list."""
        nodes = None
        result = parse_to_string_list(nodes)
        self.assertEqual(result, [])

    def test_parse_string_nodes(self) -> None:
        """Test parsing string to empty list."""
        nodes = "input"
        result = parse_to_string_list(nodes)
        self.assertEqual(result, ["input"])

    def test_parse_to_string_list_empty_list(self) -> None:
        """Test parsing list to list."""
        nodes: List[Any] = []
        result = parse_to_string_list(nodes)
        self.assertEqual(result, [])

    def test_parse_to_string_list_list(self) -> None:
        """Test parsing nodes list to list."""
        nodes = ["input_1", "input_2"]
        result = parse_to_string_list(nodes)
        self.assertEqual(result, ["input_1", "input_2"])

    def test_normalize_string(self) -> None:
        """Test string normalization."""
        string = "/Some string with extra / characters: éèà'çëöāãñę"
        result = normalize_string(string)
        self.assertEqual(result, "some_string_with_extra_characters_eeaceoaane")

    def test_parse_version(self) -> None:
        """Test parsing version from string."""
        string_version = "1.2.3+cpu"
        result = parse_version(string_version)
        self.assertEqual(result, "1.2.3")

    def test_parse_version_exception(self) -> None:
        """Test parsing version from string."""
        string_version = "ModuleNotFoundError: No module named 'module'"
        with self.assertRaisesRegex(InternalException, "Could not parse version."):
            parse_version(string_version)

    def test_parse_version_none(self) -> None:
        """Test parsing version from None."""
        string_version = None
        with self.assertRaisesRegex(InternalException, "Missing version to parse."):
            parse_version(string_version)

    def test_parse_version_without_patch(self) -> None:
        """Test parsing version from string."""
        string_version = "4.5"
        result = parse_version(string_version)
        self.assertEqual(result, "4.5")

    def test_parse_version_without_minor(self) -> None:
        """Test parsing version from string."""
        string_version = "6"
        result = parse_version(string_version)
        self.assertEqual(result, "6")


if __name__ == "__main__":
    unittest.main()
