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
"""Parameters feeder test."""

import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

from lpot.ux.components.configuration_wizard.params_feeder import Feeder
from lpot.ux.utils.exceptions import ClientErrorException


class TestParamsFeeder(unittest.TestCase):
    """Main test class for params feeder."""

    def test_feed_without_param_failes(self) -> None:
        """Test that calling feed when param is not set fails."""
        feeder = Feeder(data={})
        with self.assertRaisesRegex(ClientErrorException, "Parameter not defined."):
            feeder.feed()

    def test_feed_with_unknown_param_fails(self) -> None:
        """Test that calling feed with not supported param fails."""
        feeder = Feeder(data={"param": "foo"})
        with self.assertRaisesRegex(
            ClientErrorException,
            "Could not found method for foo parameter.",
        ):
            feeder.feed()

    @patch("lpot.ux.components.configuration_wizard.params_feeder.Feeder.get_frameworks")
    def test_feed_for_framework_works(self, mocked_get_frameworks: MagicMock) -> None:
        """Test that calling feed with not supported param fails."""
        frameworks = [
            {"name": "framework1"},
            {"name": "framework2"},
            {"name": "framework3"},
        ]
        mocked_get_frameworks.return_value = frameworks
        expected = {"framework": frameworks}

        feeder = Feeder(data={"param": "framework"})
        actual = feeder.feed()

        self.assertEqual(expected, actual)

    @patch("lpot.ux.components.configuration_wizard.params_feeder.ModelRepository.get_frameworks")
    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_model_config")
    def test_get_frameworks(
        self,
        mocked_load_model_config: MagicMock,
        mocked_get_frameworks: MagicMock,
    ) -> None:
        """Test get_frameworks function."""
        mocked_load_model_config.return_value = {
            "__help__framework_foo": "framework_foo is in known frameworks, so should be inculded",
            "framework_foo": {
                "__help__domain1": "help text for framework_foo/domain1",
                "domain1": {},
            },
            "__help__framework_bar": "framework_bar is not known, so should be ignored",
            "framework_bar": {
                "__help__domain1": "help text for framework_bar/domain1",
                "domain1": {},
            },
            "__help__framework_baz": "framework_baz is in known frameworks, so should be inculded",
            "framework_baz": {
                "__help__domain1": "help text for framework_baz/domain1",
                "domain1": {},
            },
        }
        mocked_get_frameworks.return_value = [
            "framework_baz",
            "framework_foo",
        ]
        expected = [
            {
                "name": "framework_foo",
                "help": "framework_foo is in known frameworks, so should be inculded",
            },
            {
                "name": "framework_baz",
                "help": "framework_baz is in known frameworks, so should be inculded",
            },
        ]

        actual = Feeder.get_frameworks()

        self.assertEqual(expected, actual)

    def test_get_domains_fails_without_framework(self) -> None:
        """Test that get_domains fails when no config given."""
        feeder = Feeder(data={})
        with self.assertRaisesRegex(ClientErrorException, "Framework not set."):
            feeder.get_domains()

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_model_config")
    def test_get_domains(
        self,
        mocked_load_model_config: MagicMock,
    ) -> None:
        """Test get_domains method."""
        mocked_load_model_config.return_value = {
            "__help__framework_foo": "help for framework_foo",
            "framework_foo": {
                "__help__domain1": "help text for framework_foo/domain1",
                "domain1": {},
                "__help__domain2": "help text for framework_foo/domain2",
                "domain2": {},
                "__help__domain3": "help text for framework_foo/domain3",
                "domain3": {},
            },
            "__help__framework_bar": "help for framework_bar",
            "framework_bar": {
                "__help__domain1": "help text for framework_bar/domain1",
                "domain1": {},
            },
            "__help__framework_baz": "help for framework_baz",
            "framework_baz": {
                "__help__domain1": "help text for framework_baz/domain1",
                "domain1": {},
            },
        }

        expected = [
            {
                "name": "domain1",
                "help": "help text for framework_foo/domain1",
            },
            {
                "name": "domain2",
                "help": "help text for framework_foo/domain2",
            },
            {
                "name": "domain3",
                "help": "help text for framework_foo/domain3",
            },
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                },
            },
        )
        actual = feeder.get_domains()

        mocked_load_model_config.assert_called_once()
        self.assertEqual(expected, actual)

    def test_get_models_fails_without_framework(self) -> None:
        """Test that get_models fails when no config given."""
        feeder = Feeder(data={})
        with self.assertRaisesRegex(ClientErrorException, "Framework not set."):
            feeder.get_models()

    def test_get_models_fails_without_domain(self) -> None:
        """Test that get_models fails when no config given."""
        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                },
            },
        )
        with self.assertRaisesRegex(ClientErrorException, "Domain not set."):
            feeder.get_models()

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_model_config")
    def test_get_models(
        self,
        mocked_load_model_config: MagicMock,
    ) -> None:
        """Test get_models method."""
        mocked_load_model_config.return_value = {
            "__help__framework_foo": "help for framework_foo",
            "framework_foo": {
                "__help__domain1": "help text for framework_foo/domain1",
                "domain1": {},
                "__help__domain2": "help text for framework_foo/domain2",
                "domain2": {
                    "__help__model1": "help for model 1",
                    "model1": {},
                    "__help__model2": "help for model 2",
                    "model2": {},
                    "__help__model3": "help for model 3",
                    "model3": {},
                },
                "__help__domain3": "help text for framework_foo/domain3",
                "domain3": {},
            },
            "__help__framework_bar": "help for framework_bar",
            "framework_bar": {
                "__help__domain1": "help text for framework_bar/domain1",
                "domain1": {},
            },
            "__help__framework_baz": "help for framework_baz",
            "framework_baz": {
                "__help__domain1": "help text for framework_baz/domain1",
                "domain1": {},
            },
        }

        expected = [
            {
                "name": "model1",
                "help": "help for model 1",
            },
            {
                "name": "model2",
                "help": "help for model 2",
            },
            {
                "name": "model3",
                "help": "help for model 3",
            },
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                    "domain": "domain2",
                },
            },
        )
        actual = feeder.get_models()

        mocked_load_model_config.assert_called_once()
        self.assertEqual(expected, actual)

    def test_get_dataloaders_fails_without_framework(self) -> None:
        """Test that get_dataloaders fails when no config given."""
        feeder = Feeder(data={})
        with self.assertRaisesRegex(ClientErrorException, "Framework not set."):
            feeder.get_dataloaders()

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_dataloader_config")
    def test_get_dataloaders_for_unknown_framework(
        self,
        mocked_load_dataloader_config: MagicMock,
    ) -> None:
        """Test that get_dataloaders works when unknown framework requested."""
        mocked_load_dataloader_config.return_value = [
            {
                "name": "framework_foo",
                "help": None,
                "params": {},
            },
            {
                "name": "framework_bar",
                "help": None,
                "params": {},
            },
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_unknown",
                },
            },
        )
        actual = feeder.get_dataloaders()

        mocked_load_dataloader_config.assert_called_once()
        self.assertEqual([], actual)

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_dataloader_config")
    def test_get_dataloaders(
        self,
        mocked_load_dataloader_config: MagicMock,
    ) -> None:
        """Test that get_dataloaders works."""
        params = [
            {"name": "param1"},
            {"name": "param2"},
            {"name": "param3"},
        ]
        mocked_load_dataloader_config.return_value = [
            {
                "name": "framework_foo",
                "help": None,
                "params": params,
            },
            {
                "name": "framework_bar",
                "help": None,
                "params": {},
            },
        ]

        expected = params

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                },
            },
        )
        actual = feeder.get_dataloaders()

        mocked_load_dataloader_config.assert_called_once()
        self.assertEqual(expected, actual)

    def test_get_transforms_fails_without_framework(self) -> None:
        """Test that get_transforms fails when no config given."""
        feeder = Feeder(data={})
        with self.assertRaisesRegex(ClientErrorException, "Framework not set."):
            feeder.get_transforms()

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_transforms_config")
    def test_get_transforms_for_unknown_framework(
        self,
        mocked_load_transforms_config: MagicMock,
    ) -> None:
        """Test that get_transforms works when unknown framework requested."""
        mocked_load_transforms_config.return_value = [
            {
                "name": "framework_foo",
                "help": None,
                "params": [
                    {"name": "transform1"},
                    {"name": "transform2"},
                    {"name": "transform3"},
                ],
            },
            {
                "name": "framework_bar",
                "help": None,
                "params": [
                    {"name": "transform4"},
                    {"name": "transform5"},
                    {"name": "transform6"},
                ],
            },
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_unknown",
                },
            },
        )
        actual = feeder.get_transforms()

        mocked_load_transforms_config.assert_called_once()
        self.assertEqual([], actual)

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_transforms_config")
    def test_get_transforms_without_domain(
        self,
        mocked_load_transforms_config: MagicMock,
    ) -> None:
        """Test that get_transforms works when no domain requested."""
        mocked_load_transforms_config.return_value = [
            {
                "name": "framework_foo",
                "help": None,
                "params": [
                    {"name": "transform1"},
                    {"name": "transform2"},
                    {"name": "transform3"},
                ],
            },
            {
                "name": "framework_bar",
                "help": None,
                "params": [
                    {"name": "transform4"},
                    {"name": "transform5"},
                    {"name": "transform6"},
                ],
            },
        ]
        expected = [
            {"name": "transform1"},
            {"name": "transform2"},
            {"name": "transform3"},
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                },
            },
        )
        actual = feeder.get_transforms()

        mocked_load_transforms_config.assert_called_once()
        self.assertEqual(expected, actual)

    @patch("lpot.ux.utils.utils.load_transforms_filter_config")
    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_transforms_config")
    def test_get_transforms_with_domain_not_in_filters(
        self,
        mocked_load_transforms_config: MagicMock,
        mocked_load_transforms_filter_config: MagicMock,
    ) -> None:
        """Test that get_transforms works."""
        mocked_load_transforms_config.return_value = [
            {
                "name": "framework_foo",
                "help": None,
                "params": [
                    {"name": "transform1"},
                    {"name": "transform2"},
                    {"name": "transform3"},
                ],
            },
            {
                "name": "framework_bar",
                "help": None,
                "params": [
                    {"name": "transform4"},
                    {"name": "transform5"},
                    {"name": "transform6"},
                ],
            },
        ]
        mocked_load_transforms_filter_config.return_value = {
            "framework_foo": {
                "domain_foo": [
                    "transform1",
                    "transform2",
                    "transform3",
                    "transform4",
                ],
                "domain_bar": [
                    "transform2",
                    "transform3",
                    "transform4",
                ],
            },
        }

        expected = [
            {"name": "transform1"},
            {"name": "transform2"},
            {"name": "transform3"},
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                    "domain": "domain_UNKNOWN",
                },
            },
        )
        actual = feeder.get_transforms()

        mocked_load_transforms_config.assert_called_once()
        mocked_load_transforms_filter_config.assert_called_once()
        self.assertEqual(expected, actual)

    @patch("lpot.ux.utils.utils.load_transforms_filter_config")
    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_transforms_config")
    def test_get_transforms(
        self,
        mocked_load_transforms_config: MagicMock,
        mocked_load_transforms_filter_config: MagicMock,
    ) -> None:
        """Test that get_transforms works."""
        mocked_load_transforms_config.return_value = [
            {
                "name": "framework_foo",
                "help": None,
                "params": [
                    {"name": "transform1"},
                    {"name": "transform2"},
                    {"name": "transform3"},
                ],
            },
            {
                "name": "framework_bar",
                "help": None,
                "params": [
                    {"name": "transform4"},
                    {"name": "transform5"},
                    {"name": "transform6"},
                ],
            },
        ]
        mocked_load_transforms_filter_config.return_value = {
            "framework_foo": {
                "domain_foo": [
                    "transform1",
                    "transform2",
                    "transform3",
                    "transform4",
                ],
                "domain_bar": [
                    "transform2",
                    "transform3",
                    "transform4",
                ],
            },
        }

        expected = [
            {"name": "transform2"},
            {"name": "transform3"},
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                    "domain": "domain_bar",
                },
            },
        )
        actual = feeder.get_transforms()

        mocked_load_transforms_config.assert_called_once()
        mocked_load_transforms_filter_config.assert_called_once()
        self.assertEqual(expected, actual)

    @patch(
        "lpot.ux.components.configuration_wizard.params_feeder.OBJECTIVES",
        {"objective1": {}, "objective2": {}, "objective3": {}},
    )
    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_help_lpot_params")
    def test_get_objectives(
        self,
        mocked_load_help_lpot_params: MagicMock,
    ) -> None:
        """Test get_objectives function."""
        mocked_load_help_lpot_params.return_value = {
            "__help__objective1": "help1",
            "__help__objective_unknown": "this should be skipped",
            "__help__objective2": "help2",
        }
        expected = [
            {
                "name": "objective1",
                "help": "help1",
            },
            {
                "name": "objective2",
                "help": "help2",
            },
            {
                "name": "objective3",
                "help": "",
            },
        ]

        actual = Feeder.get_objectives()

        mocked_load_help_lpot_params.assert_called_once_with("objectives")
        self.assertEqual(expected, actual)

    @patch(
        "lpot.ux.components.configuration_wizard.params_feeder.STRATEGIES",
        {"strategy1": {}, "strategy2": {}, "strategy3": {}, "sigopt": {}},
    )
    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_help_lpot_params")
    def test_get_strategies(
        self,
        mocked_load_help_lpot_params: MagicMock,
    ) -> None:
        """Test get_strategies function."""
        mocked_load_help_lpot_params.return_value = {
            "__help__strategy1": "help1",
            "__help__strategy_unknown": "this should be skipped",
            "__help__strategy2": "help2",
        }
        expected = [
            {
                "name": "strategy1",
                "help": "help1",
            },
            {
                "name": "strategy2",
                "help": "help2",
            },
            {
                "name": "strategy3",
                "help": "",
            },
        ]

        actual = Feeder.get_strategies()

        mocked_load_help_lpot_params.assert_called_once_with("strategies")
        self.assertEqual(expected, actual)

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_precisions_config")
    def test_get_precisions(
        self,
        mocked_load_precisions_config: MagicMock,
    ) -> None:
        """Test get_precisions function."""
        mocked_load_precisions_config.return_value = {
            "framework_foo": [
                {"foo": {}},
                {"bar": {}},
                {"baz": {}},
            ],
            "framework_bar": [
                {"foo": {}},
            ],
        }

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                },
            },
        )
        actual = feeder.get_precisions()

        self.assertEqual(
            [
                {"foo": {}},
                {"bar": {}},
                {"baz": {}},
            ],
            actual,
        )

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_precisions_config")
    def test_get_precisions_for_unknown_framework(
        self,
        mocked_load_precisions_config: MagicMock,
    ) -> None:
        """Test get_precisions function."""
        mocked_load_precisions_config.return_value = {
            "framework_foo": [
                {"foo": {}},
                {"bar": {}},
                {"baz": {}},
            ],
            "framework_bar": [
                {"foo": {}},
            ],
        }

        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_baz",
                },
            },
        )
        actual = feeder.get_precisions()

        self.assertEqual([], actual)

    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_precisions_config")
    def test_get_precisions_for_missing_framework(
        self,
        mocked_load_precisions_config: MagicMock,
    ) -> None:
        """Test get_precisions function."""
        mocked_load_precisions_config.return_value = {
            "framework_foo": [
                {"foo": {}},
                {"bar": {}},
                {"baz": {}},
            ],
            "framework_bar": [
                {"foo": {}},
            ],
        }

        feeder = Feeder(data={})

        with self.assertRaisesRegex(ClientErrorException, "Framework not set."):
            feeder.get_precisions()

    def test_get_quantization_approaches_for_fake_framework(self) -> None:
        """Test get_quantization_approaches."""
        feeder = Feeder(
            data={
                "config": {
                    "framework": "framework_foo",
                },
            },
        )
        output = feeder.get_quantization_approaches()
        quantization_names = [approach.get("name") for approach in output]

        self.assertEqual(["post_training_static_quant"], quantization_names)

    def test_get_quantization_approaches_for_pytorch(self) -> None:
        """Test get_quantization_approaches."""
        feeder = Feeder(
            data={
                "config": {
                    "framework": "pytorch",
                },
            },
        )
        output = feeder.get_quantization_approaches()
        quantization_names = [approach.get("name") for approach in output]

        self.assertEqual(
            ["post_training_static_quant", "post_training_dynamic_quant"],
            quantization_names,
        )

    def test_get_quantization_approaches_for_onnxrt(self) -> None:
        """Test get_quantization_approaches."""
        feeder = Feeder(
            data={
                "config": {
                    "framework": "onnxrt",
                },
            },
        )
        output = feeder.get_quantization_approaches()
        quantization_names = [approach.get("name") for approach in output]

        self.assertEqual(
            ["post_training_static_quant", "post_training_dynamic_quant"],
            quantization_names,
        )

    def test_get_metrics_fails_without_framework(self) -> None:
        """Test that get_domains fails when no config given."""
        feeder = Feeder(data={})
        with self.assertRaisesRegex(ClientErrorException, "Framework not set."):
            feeder.get_metrics()

    class FakeMetrics:
        """Metrics class placeholder for tests."""

        def __init__(self) -> None:
            """Create object."""
            self.metrics: Dict[str, dict] = {
                "topk": {},
                "COCOmAP": {},
                "MSE": {},
                "RMSE": {},
                "MAE": {},
                "metric1": {},
            }

    @patch(
        "lpot.ux.components.configuration_wizard.params_feeder.framework_metrics",
        {"pytorch": FakeMetrics},
    )
    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_help_lpot_params")
    @patch("lpot.ux.components.configuration_wizard.params_feeder.check_module")
    def test_get_metrics_for_pytorch(
        self,
        mocked_check_module: MagicMock,
        mocked_load_help_lpot_params: MagicMock,
    ) -> None:
        """Test that get_domains fails when no config given."""
        mocked_load_help_lpot_params.return_value = {
            "__help__topk": "help for topk",
            "topk": {
                "__help__k": "help for k in topk",
                "__help__missing_param": "help for missing_param in topk",
            },
            "__help__metric1": "help for metric1",
            "__help__metric3": "help for metric3",
        }

        expected = [
            {
                "name": "topk",
                "help": "help for topk",
                "params": [
                    {
                        "name": "k",
                        "help": "help for k in topk",
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
                        "help": "",
                        "value": "",
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
                "help": "help for metric1",
                "value": None,
            },
            {
                "name": "custom",
                "help": "",
                "value": None,
            },
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "pytorch",
                },
            },
        )

        actual = feeder.get_metrics()

        mocked_check_module.assert_called_once_with("ignite")
        mocked_load_help_lpot_params.assert_called_once_with("metrics")
        self.assertEqual(expected, actual)

    @patch(
        "lpot.ux.components.configuration_wizard.params_feeder.framework_metrics",
        {"onnxrt_qlinearops": FakeMetrics},
    )
    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_help_lpot_params")
    @patch("lpot.ux.components.configuration_wizard.params_feeder.check_module")
    def test_get_metrics_for_onnxrt(
        self,
        mocked_check_module: MagicMock,
        mocked_load_help_lpot_params: MagicMock,
    ) -> None:
        """Test that get_domains fails when no config given."""
        mocked_load_help_lpot_params.return_value = {
            "__help__topk": "help for topk",
            "topk": {
                "__help__k": "help for k in topk",
                "__help__missing_param": "help for missing_param in topk",
            },
            "__help__metric1": "help for metric1",
            "__help__metric3": "help for metric3",
        }

        expected = [
            {
                "name": "topk",
                "help": "help for topk",
                "params": [
                    {
                        "name": "k",
                        "help": "help for k in topk",
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
                        "help": "",
                        "value": "",
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
                "help": "help for metric1",
                "value": None,
            },
            {
                "name": "custom",
                "help": "",
                "value": None,
            },
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "onnxrt",
                },
            },
        )

        actual = feeder.get_metrics()

        mocked_check_module.assert_called_once_with("onnxrt")
        mocked_load_help_lpot_params.assert_called_once_with("metrics")
        self.assertEqual(expected, actual)

    @patch("lpot.ux.components.configuration_wizard.params_feeder.framework_metrics", {})
    @patch("lpot.ux.components.configuration_wizard.params_feeder.load_help_lpot_params")
    @patch("lpot.ux.components.configuration_wizard.params_feeder.check_module")
    def test_get_metrics_for_unknown_framework(
        self,
        mocked_check_module: MagicMock,
        mocked_load_help_lpot_params: MagicMock,
    ) -> None:
        """Test that get_domains fails when no config given."""
        mocked_load_help_lpot_params.return_value = {
            "__help__topk": "help for topk",
            "topk": {
                "__help__k": "help for k in topk",
                "__help__missing_param": "help for missing_param in topk",
            },
            "__help__metric1": "help for metric1",
            "__help__metric3": "help for metric3",
        }

        expected = [
            {
                "name": "custom",
                "help": "",
                "value": None,
            },
        ]

        feeder = Feeder(
            data={
                "config": {
                    "framework": "unknown_framework",
                },
            },
        )

        actual = feeder.get_metrics()

        mocked_check_module.assert_called_once_with("unknown_framework")
        mocked_load_help_lpot_params.assert_called_once_with("metrics")
        self.assertEqual(expected, actual)

    @patch("lpot.ux.components.configuration_wizard.params_feeder.Feeder")
    def test_get_possible_values(
        self,
        mocked_feeder: MagicMock,
    ) -> None:
        """Test get_possible_values function."""
        from lpot.ux.components.configuration_wizard.params_feeder import get_possible_values

        data = {
            "foo": "bar",
        }
        expected = {
            "a": "b",
            "c": "d",
        }
        mocked_feeder.return_value.feed.return_value = expected

        actual = get_possible_values(data)

        mocked_feeder.assert_called_once_with(data)
        mocked_feeder.return_value.feed.assert_called_once()
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
