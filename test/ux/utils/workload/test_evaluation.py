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
"""Evaluation config test."""

import unittest
from unittest.mock import MagicMock, patch

from neural_compressor.ux.utils.workload.evaluation import (
    Accuracy,
    Configs,
    Evaluation,
    Metric,
    Performance,
    Postprocess,
    PostprocessSchema,
)


class TestMetricConfig(unittest.TestCase):
    """Metric config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Metric config test constructor."""
        super().__init__(*args, **kwargs)

    def test_metric_constructor(self) -> None:
        """Test Metric config constructor."""
        data = {
            "name": "topk",
            "param": 1,
        }
        metric = Metric(data)

        self.assertEqual(metric.name, "topk")
        self.assertEqual(metric.param, 1)

    def test_metric_constructor_from_yaml(self) -> None:
        """Test Metric config constructor."""
        data = {"topk": 1}
        metric = Metric(data)

        self.assertEqual(metric.name, "topk")
        self.assertEqual(metric.param, 1)

    def test_metric_constructor_defaults(self) -> None:
        """Test Metric config constructor defaults."""
        metric = Metric()

        self.assertIsNone(metric.name)
        self.assertIsNone(metric.param)

    def test_metric_setters(self) -> None:
        """Test metric config setters."""
        metric = Metric()
        metric.name = "topk"
        metric.param = 1

        self.assertEqual(metric.name, "topk")
        self.assertEqual(metric.param, 1)

    def test_metric_setters_with_empty_params(self) -> None:
        """Test metric config setters with empty parameters."""
        metric = Metric()
        metric.name = "TestMetric"
        metric.param = ""

        self.assertEqual(metric.name, "TestMetric")
        self.assertIsNone(metric.param)

    def test_metric_serializer(self) -> None:
        """Test Metric config serializer."""
        data = {
            "name": "topk",
            "param": 1,
        }
        metric = Metric(data)
        result = metric.serialize()

        self.assertDictEqual(
            result,
            {
                "topk": 1,
            },
        )

    def test_MSE_RMSE_MAE_metric_serializer(self) -> None:
        """Test MSE, RMSE and MAE metric config serializer."""
        for metric_name in ["MSE", "RMSE", "MAE"]:
            data = {
                "name": metric_name,
                "param": True,
            }
            metric = Metric(data)
            result = metric.serialize()

            self.assertDictEqual(
                result,
                {
                    f"{metric_name}": {"compare_label": True},
                },
            )

    def test_COCOmAP_metric_serializer_with_anno_path_param(self) -> None:
        """Test COCOmAP metric config serializer."""
        for metric_name in ["COCOmAP"]:
            data = {
                "name": metric_name,
                "param": "/foo/bar/baz",
            }
            metric = Metric(data)
            result = metric.serialize()

            self.assertDictEqual(
                result,
                {
                    f"{metric_name}": {"anno_path": "/foo/bar/baz"},
                },
            )

    def test_COCOmAP_metric_serializer_without_anno_path_param(self) -> None:
        """Test COCOmAP metric config serializer."""
        for metric_name in ["COCOmAP"]:
            data = {
                "name": metric_name,
                "param": {},
            }
            metric = Metric(data)
            result = metric.serialize()
            self.assertDictEqual(
                result,
                {
                    f"{metric_name}": {},
                },
            )

    def test_unnamed_metric_serializer(self) -> None:
        """Test unnamed metric config serializer."""
        data = {
            "param": 1,
            "other param": 2,
        }
        metric = Metric(data)
        result = metric.serialize()

        self.assertDictEqual(result, {})

    def test_COCOmAP_yaml_style_with_params(self) -> None:
        """Test metric with already defined params defaults in yaml style."""
        data = {"COCOmAP": {"anno_path": "/path/to/annotation"}}
        metric = Metric(data)
        self.assertDictEqual(
            metric.serialize(),
            data,
        )

    def test_COCOmAP_yaml_style_without_params(self) -> None:
        """Test metric without defined params defaults in yaml style."""
        data = {"COCOmAP": {}}
        metric = Metric(data)
        self.assertDictEqual(
            metric.serialize(),
            data,
        )

    def test_MSE_RMSE_MAE_metric_yaml_style(self) -> None:
        """Test MSE, RMSE and MAE metrics serialization in yaml style."""
        for metric_name in ["MSE", "RMSE", "MAE"]:
            data = {
                f"{metric_name}": {
                    "compare_label": True,
                },
            }
            metric = Metric(data)
            result = metric.serialize()

            self.assertDictEqual(
                result,
                {
                    f"{metric_name}": {"compare_label": True},
                },
            )


class TestConfigsConfig(unittest.TestCase):
    """Configs config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Configs config test constructor."""
        super().__init__(*args, **kwargs)

    @patch("neural_compressor.ux.utils.workload.evaluation.HWInfo")
    def test_configs_constructor(self, mock_hwinfo: MagicMock) -> None:
        """Test Configs config constructor."""
        mock_hwinfo.return_value.cores_per_socket = 5
        mock_hwinfo.return_value.sockets = 3

        data = {
            "cores_per_instance": 2,
            "num_of_instance": 4,
            "inter_num_of_threads": 8,
            "intra_num_of_threads": 16,
            "kmp_blocktime": 3,
        }
        configs = Configs(data)

        self.assertEqual(configs.cores_per_instance, 2)
        self.assertEqual(configs.num_of_instance, 4)
        self.assertEqual(configs.inter_num_of_threads, 8)
        self.assertEqual(configs.intra_num_of_threads, 16)
        self.assertEqual(configs.kmp_blocktime, 3)

    @patch("neural_compressor.ux.utils.workload.evaluation.HWInfo")
    def test_configs_constructor_defaults(self, mock_hwinfo: MagicMock) -> None:
        """Test Configs config constructor defaults."""
        mock_hwinfo.return_value.cores = 5
        mock_hwinfo.return_value.sockets = 3

        configs = Configs()

        self.assertEqual(configs.cores_per_instance, 4)
        self.assertEqual(configs.num_of_instance, 1)
        self.assertIsNone(configs.inter_num_of_threads)
        self.assertIsNone(configs.intra_num_of_threads)
        self.assertEqual(1, configs.kmp_blocktime)

    @patch("neural_compressor.ux.utils.workload.evaluation.HWInfo")
    def test_configs_serializer(self, mock_hwinfo: MagicMock) -> None:
        """Test Configs config serializer."""
        mock_hwinfo.return_value.cores_per_socket = 5
        mock_hwinfo.return_value.sockets = 3

        data = {
            "cores_per_instance": 2,
            "num_of_instance": 4,
            "inter_num_of_threads": 8,
            "intra_num_of_threads": 16,
            "kmp_blocktime": 3,
        }
        configs = Configs(data)
        result = configs.serialize()

        self.assertDictEqual(
            result,
            {
                "cores_per_instance": 2,
                "num_of_instance": 4,
                "inter_num_of_threads": 8,
                "intra_num_of_threads": 16,
                "kmp_blocktime": 3,
            },
        )


class TestPostprocessSchemaConfig(unittest.TestCase):
    """PostprocessSchema config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Postprocess Schema config test constructor."""
        super().__init__(*args, **kwargs)

    def test_postprocess_schema_constructor(self) -> None:
        """Test PostprocessSchema config constructor."""
        data = {
            "LabelShift": 1,
            "SquadV1": {
                "label_file": "/path/to/dev-v1.1.json",
                "vocab_file": "/path/to/vocab.txt",
            },
        }
        postprocess_schema = PostprocessSchema(data)

        self.assertEqual(postprocess_schema.LabelShift, 1)
        self.assertDictEqual(
            postprocess_schema.SquadV1,
            {
                "label_file": "/path/to/dev-v1.1.json",
                "vocab_file": "/path/to/vocab.txt",
            },
        )

    def test_postprocess_schema_constructor_defaults(self) -> None:
        """Test PostprocessSchema config constructor defaults."""
        postprocess_schema = PostprocessSchema()

        self.assertIsNone(postprocess_schema.LabelShift)
        self.assertIsNone(postprocess_schema.SquadV1)


class TestPostprocessConfig(unittest.TestCase):
    """Postprocess config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Postprocess config test constructor."""
        super().__init__(*args, **kwargs)

    def test_postprocess_constructor(self) -> None:
        """Test Postprocess config constructor."""
        data = {
            "transform": {
                "LabelShift": 1,
                "SquadV1": {
                    "label_file": "/path/to/dev-v1.1.json",
                    "vocab_file": "/path/to/vocab.txt",
                },
            },
        }
        postprocess = Postprocess(data)

        self.assertIsNotNone(postprocess.transform)
        self.assertEqual(postprocess.transform.LabelShift, 1)
        self.assertDictEqual(
            postprocess.transform.SquadV1,
            {
                "label_file": "/path/to/dev-v1.1.json",
                "vocab_file": "/path/to/vocab.txt",
            },
        )

    def test_postprocess_constructor_defaults(self) -> None:
        """Test Postprocess config constructor defaults."""
        postprocess = Postprocess()

        self.assertIsNone(postprocess.transform)


class TestAccuracyConfig(unittest.TestCase):
    """Accuracy config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Accuracy config test constructor."""
        super().__init__(*args, **kwargs)

    @patch("neural_compressor.ux.utils.workload.evaluation.HWInfo")
    def test_accuracy_constructor(self, mock_hwinfo: MagicMock) -> None:
        """Test Accuracy config constructor."""
        mock_hwinfo.return_value.cores_per_socket = 5
        mock_hwinfo.return_value.sockets = 3
        data = {
            "metric": {"topk": 1},
            "configs": {
                "cores_per_instance": 2,
                "num_of_instance": 4,
                "inter_num_of_threads": 8,
                "intra_num_of_threads": 16,
                "kmp_blocktime": 3,
            },
            "dataloader": {
                "last_batch": "rollover",
                "batch_size": 2,
                "dataset": {
                    "TestDataset": {
                        "dataset_param": "/some/path",
                        "bool_param": True,
                        "list_param": ["item1", "item2"],
                    },
                },
                "transform": {
                    "TestTransform": {"shape": [1000, 224, 224, 3], "some_op": True},
                    "AnotherTestTransform": {
                        "shape": [10, 299, 299, 3],
                        "some_op": False,
                    },
                },
                "filter": {
                    "LabelBalance": {"size": 1},
                },
            },
            "postprocess": {
                "transform": {
                    "LabelShift": 1,
                    "SquadV1": {
                        "label_file": "/path/to/dev-v1.1.json",
                        "vocab_file": "/path/to/vocab.txt",
                    },
                },
            },
        }
        accuracy = Accuracy(data)

        self.assertIsNotNone(accuracy.metric)
        self.assertEqual(accuracy.metric.name, "topk")
        self.assertEqual(accuracy.metric.param, 1)

        self.assertIsNotNone(accuracy.configs)
        self.assertEqual(
            accuracy.configs.cores_per_instance,
            2,
        )
        self.assertEqual(
            accuracy.configs.num_of_instance,
            4,
        )
        self.assertEqual(accuracy.configs.inter_num_of_threads, 8)
        self.assertEqual(accuracy.configs.intra_num_of_threads, 16)
        self.assertEqual(accuracy.configs.kmp_blocktime, 3)

        self.assertIsNotNone(accuracy.dataloader)
        self.assertEqual(accuracy.dataloader.last_batch, "rollover")
        self.assertEqual(
            accuracy.dataloader.batch_size,
            2,
        )
        self.assertIsNotNone(accuracy.dataloader.dataset)
        self.assertEqual(accuracy.dataloader.dataset.name, "TestDataset")
        self.assertDictEqual(
            accuracy.dataloader.dataset.params,
            {
                "dataset_param": "/some/path",
                "bool_param": True,
                "list_param": ["item1", "item2"],
            },
        )
        transform_name, transform = list(accuracy.dataloader.transform.items())[0]
        self.assertEqual(transform_name, "TestTransform")
        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(
            transform.parameters,
            {
                "shape": [1000, 224, 224, 3],
                "some_op": True,
            },
        )
        transform_name, transform = list(accuracy.dataloader.transform.items())[1]
        self.assertEqual(transform_name, "AnotherTestTransform")
        self.assertEqual(transform.name, "AnotherTestTransform")
        self.assertDictEqual(
            transform.parameters,
            {"shape": [10, 299, 299, 3], "some_op": False},
        )
        self.assertIsNotNone(accuracy.dataloader.filter)
        self.assertIsNotNone(accuracy.dataloader.filter.LabelBalance)
        self.assertEqual(accuracy.dataloader.filter.LabelBalance.size, 1)

        self.assertIsNotNone(accuracy.postprocess)
        self.assertIsNotNone(accuracy.postprocess.transform)
        self.assertEqual(accuracy.postprocess.transform.LabelShift, 1)
        self.assertDictEqual(
            accuracy.postprocess.transform.SquadV1,
            {
                "label_file": "/path/to/dev-v1.1.json",
                "vocab_file": "/path/to/vocab.txt",
            },
        )

    def test_accuracy_constructor_defaults(self) -> None:
        """Test Accuracy config constructor defaults."""
        accuracy = Accuracy()

        self.assertIsNone(accuracy.metric)
        self.assertIsNone(accuracy.configs)
        self.assertIsNone(accuracy.dataloader)
        self.assertIsNone(accuracy.postprocess)


class TestPerformanceConfig(unittest.TestCase):
    """Performance config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize Performance config test."""
        super().__init__(*args, **kwargs)

    @patch("neural_compressor.ux.utils.workload.evaluation.HWInfo")
    def test_performance_constructor(self, mock_hwinfo: MagicMock) -> None:
        """Test Performance config constructor."""
        mock_hwinfo.return_value.cores_per_socket = 5
        mock_hwinfo.return_value.sockets = 3
        data = {
            "warmup": 100,
            "iteration": 1000,
            "configs": {
                "cores_per_instance": 2,
                "num_of_instance": 4,
                "inter_num_of_threads": 8,
                "intra_num_of_threads": 16,
                "kmp_blocktime": 3,
            },
            "dataloader": {
                "last_batch": "rollover",
                "batch_size": 2,
                "dataset": {
                    "TestDataset": {
                        "dataset_param": "/some/path",
                        "bool_param": True,
                        "list_param": ["item1", "item2"],
                    },
                },
                "transform": {
                    "TestTransform": {"shape": [1000, 224, 224, 3], "some_op": True},
                    "AnotherTestTransform": {
                        "shape": [10, 299, 299, 3],
                        "some_op": False,
                    },
                },
                "filter": {
                    "LabelBalance": {"size": 1},
                },
            },
            "postprocess": {
                "transform": {
                    "LabelShift": 1,
                    "SquadV1": {
                        "label_file": "/path/to/dev-v1.1.json",
                        "vocab_file": "/path/to/vocab.txt",
                    },
                },
            },
        }
        performance = Performance(data)

        self.assertEqual(performance.warmup, 100)
        self.assertEqual(performance.iteration, 1000)

        self.assertIsNotNone(performance.configs)
        self.assertEqual(
            performance.configs.cores_per_instance,
            2,
        )
        self.assertEqual(
            performance.configs.num_of_instance,
            4,
        )
        self.assertEqual(performance.configs.inter_num_of_threads, 8)
        self.assertEqual(performance.configs.intra_num_of_threads, 16)
        self.assertEqual(performance.configs.kmp_blocktime, 3)

        self.assertIsNotNone(performance.dataloader)
        self.assertEqual(performance.dataloader.last_batch, "rollover")
        self.assertEqual(
            performance.dataloader.batch_size,
            2,
        )
        self.assertIsNotNone(performance.dataloader.dataset)
        self.assertEqual(performance.dataloader.dataset.name, "TestDataset")
        self.assertDictEqual(
            performance.dataloader.dataset.params,
            {
                "dataset_param": "/some/path",
                "bool_param": True,
                "list_param": ["item1", "item2"],
            },
        )
        transform_name, transform = list(performance.dataloader.transform.items())[0]
        self.assertEqual(transform_name, "TestTransform")
        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(
            transform.parameters,
            {
                "shape": [1000, 224, 224, 3],
                "some_op": True,
            },
        )
        transform_name, transform = list(performance.dataloader.transform.items())[1]
        self.assertEqual(transform_name, "AnotherTestTransform")
        self.assertEqual(transform.name, "AnotherTestTransform")
        self.assertDictEqual(
            transform.parameters,
            {"shape": [10, 299, 299, 3], "some_op": False},
        )
        self.assertIsNotNone(performance.dataloader.filter)
        self.assertIsNotNone(performance.dataloader.filter.LabelBalance)
        self.assertEqual(performance.dataloader.filter.LabelBalance.size, 1)

        self.assertIsNotNone(performance.postprocess)
        self.assertIsNotNone(performance.postprocess.transform)
        self.assertEqual(performance.postprocess.transform.LabelShift, 1)
        self.assertDictEqual(
            performance.postprocess.transform.SquadV1,
            {
                "label_file": "/path/to/dev-v1.1.json",
                "vocab_file": "/path/to/vocab.txt",
            },
        )

    @patch("neural_compressor.ux.utils.workload.evaluation.HWInfo")
    def test_performance_constructor_defaults(self, mock_hwinfo: MagicMock) -> None:
        """Test Performance config constructor defaults."""
        mock_hwinfo.return_value.cores = 5
        mock_hwinfo.return_value.sockets = 3

        performance = Performance()

        self.assertEqual(performance.warmup, 5)

        self.assertEqual(performance.iteration, -1)

        self.assertIsNotNone(performance.configs)
        self.assertEqual(performance.configs.cores_per_instance, 4)
        self.assertEqual(performance.configs.num_of_instance, 1)
        self.assertIsNone(performance.configs.inter_num_of_threads)
        self.assertIsNone(performance.configs.intra_num_of_threads)
        self.assertEqual(1, performance.configs.kmp_blocktime)

        self.assertIsNone(performance.dataloader)

        self.assertIsNone(performance.postprocess)


class TestEvaluationConfig(unittest.TestCase):
    """Evaluation config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Initialize Evaluation config test."""
        super().__init__(*args, **kwargs)

    @patch("neural_compressor.ux.utils.workload.evaluation.HWInfo")
    def test_evaluation_constructor(self, mock_hwinfo: MagicMock) -> None:
        """Test Evaluation config constructor."""
        mock_hwinfo.return_value.cores_per_socket = 5
        mock_hwinfo.return_value.sockets = 3
        data = {
            "accuracy": {
                "metric": {"topk": 1},
                "configs": {
                    "cores_per_instance": 2,
                    "num_of_instance": 4,
                    "inter_num_of_threads": 8,
                    "intra_num_of_threads": 16,
                    "kmp_blocktime": 3,
                },
                "dataloader": {
                    "last_batch": "rollover",
                    "batch_size": 2,
                    "dataset": {
                        "TestDataset": {
                            "dataset_param": "/some/path",
                            "bool_param": True,
                            "list_param": ["item1", "item2"],
                        },
                    },
                    "transform": {
                        "TestTransform": {
                            "shape": [1000, 224, 224, 3],
                            "some_op": True,
                        },
                        "AnotherTestTransform": {
                            "shape": [10, 299, 299, 3],
                            "some_op": False,
                        },
                    },
                    "filter": {
                        "LabelBalance": {"size": 1},
                    },
                },
                "postprocess": {
                    "transform": {
                        "LabelShift": 1,
                        "SquadV1": {
                            "label_file": "/path/to/dev-v1.1.json",
                            "vocab_file": "/path/to/vocab.txt",
                        },
                    },
                },
            },
            "performance": {
                "warmup": 100,
                "iteration": 1000,
                "configs": {
                    "cores_per_instance": 2,
                    "num_of_instance": 4,
                    "inter_num_of_threads": 8,
                    "intra_num_of_threads": 16,
                    "kmp_blocktime": 3,
                },
                "dataloader": {
                    "last_batch": "rollover",
                    "batch_size": 2,
                    "dataset": {
                        "TestDataset": {
                            "dataset_param": "/some/path",
                            "bool_param": True,
                            "list_param": ["item1", "item2"],
                        },
                    },
                    "transform": {
                        "TestTransform": {
                            "shape": [1000, 224, 224, 3],
                            "some_op": True,
                        },
                        "AnotherTestTransform": {
                            "shape": [10, 299, 299, 3],
                            "some_op": False,
                        },
                    },
                    "filter": {
                        "LabelBalance": {"size": 1},
                    },
                },
                "postprocess": {
                    "transform": {
                        "LabelShift": 1,
                        "SquadV1": {
                            "label_file": "/path/to/dev-v1.1.json",
                            "vocab_file": "/path/to/vocab.txt",
                        },
                    },
                },
            },
        }
        evaluation = Evaluation(data)

        self.assertIsNotNone(evaluation.accuracy)
        self.assertIsNotNone(evaluation.accuracy.metric)
        self.assertEqual(evaluation.accuracy.metric.name, "topk")
        self.assertEqual(evaluation.accuracy.metric.param, 1)

        self.assertIsNotNone(evaluation.accuracy.configs)
        self.assertEqual(
            evaluation.accuracy.configs.cores_per_instance,
            2,
        )
        self.assertEqual(
            evaluation.accuracy.configs.num_of_instance,
            4,
        )
        self.assertEqual(evaluation.accuracy.configs.inter_num_of_threads, 8)
        self.assertEqual(evaluation.accuracy.configs.intra_num_of_threads, 16)
        self.assertEqual(evaluation.accuracy.configs.kmp_blocktime, 3)

        self.assertIsNotNone(evaluation.accuracy.dataloader)
        self.assertEqual(evaluation.accuracy.dataloader.last_batch, "rollover")
        self.assertEqual(
            evaluation.accuracy.dataloader.batch_size,
            2,
        )
        self.assertIsNotNone(evaluation.accuracy.dataloader.dataset)
        self.assertEqual(evaluation.accuracy.dataloader.dataset.name, "TestDataset")
        self.assertDictEqual(
            evaluation.accuracy.dataloader.dataset.params,
            {
                "dataset_param": "/some/path",
                "bool_param": True,
                "list_param": ["item1", "item2"],
            },
        )
        transform_name, transform = list(
            evaluation.accuracy.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "TestTransform")
        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(
            transform.parameters,
            {
                "shape": [1000, 224, 224, 3],
                "some_op": True,
            },
        )
        transform_name, transform = list(
            evaluation.accuracy.dataloader.transform.items(),
        )[1]
        self.assertEqual(transform_name, "AnotherTestTransform")
        self.assertEqual(transform.name, "AnotherTestTransform")
        self.assertDictEqual(
            transform.parameters,
            {"shape": [10, 299, 299, 3], "some_op": False},
        )
        self.assertIsNotNone(evaluation.accuracy.dataloader.filter)
        self.assertIsNotNone(evaluation.accuracy.dataloader.filter.LabelBalance)
        self.assertEqual(evaluation.accuracy.dataloader.filter.LabelBalance.size, 1)

        self.assertIsNotNone(evaluation.accuracy.postprocess)
        self.assertIsNotNone(evaluation.accuracy.postprocess.transform)
        self.assertEqual(evaluation.accuracy.postprocess.transform.LabelShift, 1)
        self.assertDictEqual(
            evaluation.accuracy.postprocess.transform.SquadV1,
            {
                "label_file": "/path/to/dev-v1.1.json",
                "vocab_file": "/path/to/vocab.txt",
            },
        )

        self.assertIsNotNone(evaluation.performance)
        self.assertEqual(evaluation.performance.warmup, 100)
        self.assertEqual(evaluation.performance.iteration, 1000)

        self.assertIsNotNone(evaluation.performance.configs)
        self.assertEqual(
            evaluation.performance.configs.cores_per_instance,
            2,
        )
        self.assertEqual(
            evaluation.performance.configs.num_of_instance,
            4,
        )
        self.assertEqual(evaluation.performance.configs.inter_num_of_threads, 8)
        self.assertEqual(evaluation.performance.configs.intra_num_of_threads, 16)
        self.assertEqual(evaluation.performance.configs.kmp_blocktime, 3)

        self.assertIsNotNone(evaluation.performance.dataloader)
        self.assertEqual(evaluation.performance.dataloader.last_batch, "rollover")
        self.assertEqual(
            evaluation.performance.dataloader.batch_size,
            2,
        )
        self.assertIsNotNone(evaluation.performance.dataloader.dataset)
        self.assertEqual(
            evaluation.performance.dataloader.dataset.name,
            "TestDataset",
        )
        self.assertDictEqual(
            evaluation.performance.dataloader.dataset.params,
            {
                "dataset_param": "/some/path",
                "bool_param": True,
                "list_param": ["item1", "item2"],
            },
        )
        transform_name, transform = list(
            evaluation.performance.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "TestTransform")
        self.assertEqual(transform.name, "TestTransform")
        self.assertDictEqual(
            transform.parameters,
            {
                "shape": [1000, 224, 224, 3],
                "some_op": True,
            },
        )
        transform_name, transform = list(
            evaluation.performance.dataloader.transform.items(),
        )[1]
        self.assertEqual(transform_name, "AnotherTestTransform")
        self.assertEqual(transform.name, "AnotherTestTransform")
        self.assertDictEqual(
            transform.parameters,
            {"shape": [10, 299, 299, 3], "some_op": False},
        )
        self.assertIsNotNone(evaluation.performance.dataloader.filter)
        self.assertIsNotNone(evaluation.performance.dataloader.filter.LabelBalance)
        self.assertEqual(
            evaluation.performance.dataloader.filter.LabelBalance.size,
            1,
        )

        self.assertIsNotNone(evaluation.performance.postprocess)
        self.assertIsNotNone(evaluation.performance.postprocess.transform)
        self.assertEqual(evaluation.performance.postprocess.transform.LabelShift, 1)
        self.assertDictEqual(
            evaluation.performance.postprocess.transform.SquadV1,
            {
                "label_file": "/path/to/dev-v1.1.json",
                "vocab_file": "/path/to/vocab.txt",
            },
        )

    def test_evaluation_constructor_defaults(self) -> None:
        """Test Evaluation config constructor defaults."""
        evaluation = Evaluation()

        self.assertIsNone(evaluation.accuracy)
        self.assertIsNone(evaluation.performance)


if __name__ == "__main__":
    unittest.main()
