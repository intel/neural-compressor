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
"""Configuration type parser test."""


import unittest
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

from neural_compressor.ux.components.configuration_wizard.configuration_parser import ConfigurationParser
from neural_compressor.ux.utils.exceptions import ClientErrorException


@patch("sys.argv", ["neural_compressor_bench.py", "-p5000"])
class TestParser(unittest.TestCase):
    """Main test class for parser."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)
        self.parser = ConfigurationParser()

    def _assert_parses_to_expected_float(self, value: Any, expected: float) -> None:
        self._assert_parses_to_expected_type(value, expected, float)

    def _assert_parses_float_failure(self, value: Any) -> None:
        self._assert_parses_failure(value, float)

    def _assert_parses_to_expected_int(self, value: Any, expected: int) -> None:
        self._assert_parses_to_expected_type(value, expected, int)

    def _assert_parses_int_failure(self, value: Any) -> None:
        self._assert_parses_failure(value, int)

    def _assert_parses_to_expected_bool(self, value: Any, expected: bool) -> None:
        self._assert_parses_to_expected_type(value, expected, bool)

    def _assert_parses_bool_failure(self, value: Any) -> None:
        self._assert_parses_failure(value, bool)

    def _assert_parses_to_expected_list(
        self,
        value: Any,
        element_type: type,
        expected: list,
    ) -> None:
        self._assert_parses_to_expected_list_type(value, expected, [element_type])

    def _assert_parses_to_expected_type(
        self,
        value: Any,
        expected: Any,
        required_type: Union[type, List[type]],
    ) -> None:
        parsed_value = self.parser.parse_value(value, required_type)
        self.assertEqual(parsed_value, expected)
        self.assertIs(type(parsed_value), required_type)

    def _assert_parses_failure(
        self,
        value: Any,
        required_type: Union[type, List[type]],
    ) -> None:
        """Test failed parsing."""
        with self.assertRaises(ClientErrorException):
            self.parser.parse_value(value, required_type)

    def _assert_parses_to_expected_list_type(
        self,
        value: Any,
        expected: Any,
        required_type: List[type],
    ) -> None:
        parsed_value = self.parser.parse_value(value, required_type)
        self.assertEqual(parsed_value, expected)
        self.assertIs(type(parsed_value), list)
        for element in parsed_value:
            self.assertIs(type(element), required_type[0])


class TestValueParser(TestParser):
    """Value parser tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)

    def test_float_to_float(self) -> None:
        """Test float value parsing."""
        self._assert_parses_to_expected_float(0.1, 0.1)

    def test_int_to_float(self) -> None:
        """Test int value parsing."""
        self._assert_parses_to_expected_float(1, 1.0)

    def test_string_to_float(self) -> None:
        """Test parsing string to float."""
        self._assert_parses_to_expected_float("0.1", 0.1)

    def test_string_with_whitespace_to_float(self) -> None:
        """Test parsing string with whitespace to float."""
        self._assert_parses_to_expected_float(" \t 0.1 ", 0.1)

    def test_float_parsing_failure(self) -> None:
        """Test parsing failure."""
        self._assert_parses_float_failure("abc")

    def test_float_to_int(self) -> None:
        """Test float value parsing to int."""
        self._assert_parses_to_expected_int(11.1, 11)

    def test_int_to_int(self) -> None:
        """Test int value parsing."""
        self._assert_parses_to_expected_int(11, 11)

    def test_string_to_int(self) -> None:
        """Test parsing string to int."""
        self._assert_parses_to_expected_int("11", 11)

    def test_string_with_whitespace_to_int(self) -> None:
        """Test parsing string with whitespace to int."""
        self._assert_parses_to_expected_int(" \t \t 11 \t \t ", 11)

    def test_int_parsing_failure(self) -> None:
        """Test parsing failure."""
        self._assert_parses_int_failure("abc")

    def test_float_to_list(self) -> None:
        """Test float value parsing to float list."""
        self._assert_parses_to_expected_list(11.1, float, [11.1])

    def test_int_to_float_list(self) -> None:
        """Test int value parsing to float list."""
        self._assert_parses_to_expected_list(11, float, [11])

    def test_int_to_list(self) -> None:
        """Test int value parsing to int list."""
        self._assert_parses_to_expected_list(11, int, [11])

    def test_string_to_list(self) -> None:
        """Test parsing string to int list."""
        self._assert_parses_to_expected_list("11", int, [11])

    def test_string_with_brackets_to_list(self) -> None:
        """Test parsing string to list."""
        self._assert_parses_to_expected_list("[3, 14, 15, 9]", int, [3, 14, 15, 9])

    def test_string_to_bool_false_lowercase(self) -> None:
        """Test parsing 'false' to boolean."""
        self._assert_parses_to_expected_bool("false", False)

    def test_string_to_bool_false_capitalized(self) -> None:
        """Test parsing 'False' to boolean."""
        self._assert_parses_to_expected_bool("False", False)

    def test_string_to_bool_false_mixed_case(self) -> None:
        """Test parsing 'FaLsE' to boolean."""
        self._assert_parses_to_expected_bool("FaLsE", False)

    def test_string_to_bool_false_mixed_case_with_whitespace(self) -> None:
        """Test parsing 'FaLsE' to boolean."""
        self._assert_parses_to_expected_bool("   FaLsE \t", False)

    def test_string_to_bool_true_lowercase(self) -> None:
        """Test parsing 'true' to boolean."""
        self._assert_parses_to_expected_bool("true", True)

    def test_string_to_bool_true_capitalized(self) -> None:
        """Test parsing 'True' to boolean."""
        self._assert_parses_to_expected_bool("True", True)

    def test_string_to_bool_true_mixed_case(self) -> None:
        """Test parsing 'TrUe' to boolean."""
        self._assert_parses_to_expected_bool("TrUe", True)

    def test_string_to_bool_true_mixed_case_with_whitespace(self) -> None:
        """Test parsing 'TrUe' to boolean."""
        self._assert_parses_to_expected_bool("\t TrUe ", True)

    def test_bool_parsing_failure(self) -> None:
        """Test parsing failure."""
        self._assert_parses_bool_failure("\nsi")


class TestTransformParser(TestParser):
    """Transform parser tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)

    def test_parses_AlignImageChannel(self) -> None:
        """Test parsing of AlignImageChannel parameters."""
        self._assert_parses_single_transform_params(
            "AlignImageChannel",
            {"dim": "123"},
            {"dim": 123},
        )

    def test_parses_default_BilinearImagenet(self) -> None:
        """Test parsing of default BilinearImagenet parameters."""
        self._assert_parses_single_transform_params(
            "BilinearImagenet",
            {
                "height": 224,
                "width": 224,
                "central_fraction": 0.875,
                "mean_value": "[0.0, 0.0, 0.0]",
                "scale": 1,
            },
            {
                "height": 224,
                "width": 224,
                "central_fraction": 0.875,
                "mean_value": [0.0, 0.0, 0.0],
                "scale": 1,
            },
        )

    def test_parses_BilinearImagenet(self) -> None:
        """Test parsing of BilinearImagenet parameters."""
        self._assert_parses_single_transform_params(
            "BilinearImagenet",
            {
                "height": "1",
                "width": "2",
                "central_fraction": "3.14",
                "mean_value": "[4.5]",
                "scale": "6.7",
            },
            {
                "height": 1,
                "width": 2,
                "central_fraction": 3.14,
                "mean_value": [4.5],
                "scale": 6.7,
            },
        )

    def test_parses_Cast(self) -> None:
        """Test parsing of Cast parameters."""
        self._assert_parses_single_transform_params(
            "Cast",
            {"dtype": "float32"},
            {"dtype": "float32"},
        )

    def test_parses_empty_data(self) -> None:
        """Test parsing data without transforms."""
        parsed = self.parser.parse({})

        self.assertIsNone(parsed.get("transform", None))

    def _build_input_with_transform(self, name: str, params: Optional[dict]) -> dict:
        """Build fake data with single Transform."""
        return {
            "domain": "image_recognition",
            "framework": "tensorflow",
            "id": "f435c2a528c38176d1bb829e5cff3690",
            "model_path": "/foo/bar/baz.pb",
            "inputs": ["input_tensor"],
            "outputs": ["softmax_tensor"],
            "transform": [{"name": name, "params": params}],
            "quantization": {
                "dataset_path": "/foo/bar/baz.pb",
                "dataloader": {"name": "ImageRecord", "params": {"root": ""}},
                "accuracy_goal": 0.01,
                "sampling_size": 100,
                "strategy": "basic",
                "approach": "post_training_static_quant",
                "objective": "",
                "timeout": 0,
                "max_trials": 100,
                "random_seed": 9527,
            },
            "evaluation": {
                "dataset_path": "/foo/bat/baz",
                "dataloader": {"name": "ImageRecord", "params": {"root": ""}},
                "metric": "topk",
                "metric_param": 1,
                "batch_size": 1,
                "cores_per_instance": "1",
                "instances": "1",
                "inter_nr_of_threads": "",
                "intra_nr_of_threads": "",
                "iterations": -1,
                "warmup": 10,
                "kmp_blocktime": 1,
            },
            "workspace_path": "/foo",
            "tuning": True,
        }

    def _build_input_with_multiple_transforms(
        self,
        transform_list: List[Dict[str, Union[str, Optional[dict]]]],
    ) -> dict:
        """Build fake data with multiple transforms."""
        return {
            "domain": "image_recognition",
            "framework": "tensorflow",
            "id": "f435c2a528c38176d1bb829e5cff3690",
            "model_path": "/foo/bar/baz.pb",
            "inputs": ["input_tensor"],
            "outputs": ["softmax_tensor"],
            "transform": transform_list,
            "quantization": {
                "dataset_path": "/foo/bar/baz.pb",
                "dataloader": {"name": "ImageRecord", "params": {"root": ""}},
                "accuracy_goal": 0.01,
                "sampling_size": 100,
                "strategy": "basic",
                "approach": "post_training_static_quant",
                "objective": "",
                "timeout": 0,
                "max_trials": 100,
                "random_seed": 9527,
            },
            "evaluation": {
                "dataset_path": "/foo/bat/baz",
                "dataloader": {"name": "ImageRecord", "params": {"root": ""}},
                "metric": "topk",
                "metric_param": 1,
                "batch_size": 1,
                "cores_per_instance": "1",
                "instances": "1",
                "inter_nr_of_threads": "",
                "intra_nr_of_threads": "",
                "iterations": -1,
                "warmup": 10,
                "kmp_blocktime": 1,
            },
            "workspace_path": "/foo",
            "tuning": True,
        }

    def _assert_parses_single_transform_params(
        self,
        name: str,
        input_params: Optional[dict],
        expected_params: Optional[dict],
    ) -> None:
        parsed = self.parser.parse(self._build_input_with_transform(name, input_params))
        transforms = parsed.get("transform", None)
        self.assertIsNotNone(transforms)
        matching_transforms = list(
            filter(
                lambda e: name == e.get("name", ""),
                transforms,
            ),
        )
        self.assertEqual(1, len(matching_transforms))
        parsed_params = matching_transforms[0].get("params", {})
        self.assertEqual(parsed_params, expected_params)

    def _assert_parses_multiple_transforms(
        self,
        input_list: List[Dict[str, Union[str, Optional[dict]]]],
        expected_list: List[Dict[str, Union[str, Optional[dict]]]],
    ) -> None:
        parsed = self.parser.parse(
            self._build_input_with_multiple_transforms(input_list),
        )
        transforms = parsed.get("transform", None)
        self.assertIsNotNone(transforms)
        self.assertCountEqual(transforms, expected_list)
        self.assertListEqual(transforms, expected_list)


class TestDataloaderParser(TestParser):
    """Dataloader parser tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)

    def test_parses_dummy_dataloader(self) -> None:
        """Test parsing dummy dataloader."""
        self._assert_parses_dataloader_params(
            "dummy",
            {
                "dtype": "float32",
                "high": 127,
                "label": "False",
                "low": -128,
                "shape": "[224,224]",
            },
            {
                "dtype": "float32",
                "high": [127],
                "label": False,
                "low": [-128],
                "shape": [224, 224],
            },
        )

    def test_parses_dataloader_shape_without_brackets(self) -> None:
        """Test parsing dummy dataloader."""
        self._assert_parses_dataloader_params(
            "dummy",
            {
                "dtype": "float32",
                "high": 127,
                "label": "False",
                "low": -128,
                "shape": "224,224",
            },
            {
                "dtype": "float32",
                "high": [127],
                "label": False,
                "low": [-128],
                "shape": [224, 224],
            },
        )

    def test_parses_multidim_dummy_dataloader(self) -> None:
        """Test parsing multidim dummy dataloader."""
        self._assert_parses_dataloader_params(
            "dummy",
            {
                "shape": "[[128, 3, 224, 224], [128, 1, 1, 1]]",
            },
            {
                "shape": [[128, 3, 224, 224], [128, 1, 1, 1]],
            },
        )

    def test_parses_evaluation_performance_dataloader_batch_size(self) -> None:
        """Test parsing evaluation.performance.batch_size."""
        parsed = self.parser.parse(
            self._build_input_with_dataloader("foo", {}),
        )

        self.assertEqual(234, parsed.get("evaluation", {}).get("batch_size"))

    def _build_input_with_dataloader(self, name: str, params: Optional[dict]) -> dict:
        """Build fake data with dataloader."""
        return {
            "domain": "image_recognition",
            "framework": "tensorflow",
            "id": "f435c2a528c38176d1bb829e5cff3690",
            "model_path": "/foo/bar/baz.pb",
            "inputs": ["input_tensor"],
            "outputs": ["softmax_tensor"],
            "transform": [],
            "quantization": {
                "dataset_path": "/foo/bar/baz.pb",
                "dataloader": {"name": name, "params": params},
                "accuracy_goal": 0.01,
                "sampling_size": 100,
                "strategy": "basic",
                "approach": "post_training_static_quant",
                "objective": "",
                "timeout": 0,
                "max_trials": 100,
                "random_seed": 9527,
            },
            "evaluation": {
                "dataset_path": "/foo/bat/baz",
                "dataloader": {"name": name, "params": params},
                "metric": "topk",
                "metric_param": 1,
                "batch_size": "234",
                "cores_per_instance": "1",
                "instances": "1",
                "inter_nr_of_threads": "",
                "intra_nr_of_threads": "",
                "iterations": -1,
                "warmup": 10,
                "kmp_blocktime": 1,
            },
            "workspace_path": "/foo",
            "tuning": True,
        }

    def _assert_parses_dataloader_params(
        self,
        name: str,
        input_params: Optional[dict],
        expected_params: Optional[dict],
    ) -> None:
        parsed = self.parser.parse(
            self._build_input_with_dataloader(name, input_params),
        )
        q_dataloader_params = (
            parsed.get("quantization", {}).get("dataloader", {}).get("params", None)
        )
        eval_dataloader_params = (
            parsed.get("evaluation", {}).get("dataloader", None).get("params", None)
        )

        self.assertIsNotNone(q_dataloader_params)
        self.assertIsNotNone(eval_dataloader_params)

        self.assertEqual(q_dataloader_params, expected_params)
        self.assertEqual(eval_dataloader_params, expected_params)


class TestEvaluationParser(TestParser):
    """Test evaluation part of parser."""

    def test_non_positive_cores_per_instance_fails(self) -> None:
        """Test parsing evaluation with non positive cores per instance."""
        with self.assertRaisesRegex(
            ClientErrorException,
            "At least one core per instance must be used.",
        ):
            self.parser.parse(
                {
                    "evaluation": {
                        "cores_per_instance": 0,
                    },
                },
            )

    @patch("neural_compressor.ux.components.configuration_wizard.configuration_parser.HWInfo")
    def test_too_big_num_cores_fails(self, mocked_hw_info: MagicMock) -> None:
        """Test parsing evaluation with non positive cores per instance."""
        mocked_hw_info.return_value.cores = 4
        with self.assertRaisesRegex(
            ClientErrorException,
            "Requested 10 cores per instance, while only 4 available.",
        ):
            self.parser.parse(
                {
                    "evaluation": {
                        "cores_per_instance": 10,
                    },
                },
            )

    def test_non_positive_instances_fails(self) -> None:
        """Test parsing evaluation with non positive cores per instance."""
        with self.assertRaisesRegex(
            ClientErrorException,
            "At least one instance must be used.",
        ):
            self.parser.parse(
                {
                    "evaluation": {
                        "cores_per_instance": 1,
                        "instances": 0,
                    },
                },
            )

    @patch("neural_compressor.ux.components.configuration_wizard.configuration_parser.HWInfo")
    def test_too_big_instances_fails(self, mocked_hw_info: MagicMock) -> None:
        """Test parsing evaluation with non positive cores per instance."""
        mocked_hw_info.return_value.cores = 12
        with self.assertRaisesRegex(
            ClientErrorException,
            "Attempted to use 4 instances, while only 3 allowed.",
        ):
            self.parser.parse(
                {
                    "evaluation": {
                        "cores_per_instance": 4,
                        "instances": 4,
                    },
                },
            )


if __name__ == "__main__":
    unittest.main()
