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
"""Config test."""
import unittest
from unittest.mock import MagicMock, mock_open, patch

import yaml

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.workload.config import Config


class TestConfig(unittest.TestCase):
    """Config tests."""

    def setUp(self) -> None:
        """Prepare test."""
        super().setUp()
        self.predefined_config = {
            "model_path": "/path/to/model",
            "domain": "image_recognition",
            "device": "cpu",
            "model": {
                "name": "resnet50_v1_5",
                "framework": "tensorflow",
                "outputs": "softmax_tensor",
            },
            "quantization": {
                "calibration": {
                    "sampling_size": 100,
                    "dataloader": {
                        "batch_size": 10,
                        "dataset": {"ImageRecord": {"root": "/path/to/calibration/dataset"}},
                        "transform": {
                            "ResizeCropImagenet": {
                                "height": 224,
                                "width": 224,
                                "mean_value": [123.68, 116.78, 103.94],
                            },
                        },
                    },
                },
                "model_wise": {"activation": {"algorithm": "minmax"}},
            },
            "evaluation": {
                "accuracy": {
                    "metric": {"topk": 1},
                    "dataloader": {
                        "batch_size": 32,
                        "dataset": {"ImageRecord": {"root": "/path/to/evaluation/dataset"}},
                        "transform": {
                            "ResizeCropImagenet": {
                                "height": 224,
                                "width": 224,
                                "mean_value": [123.68, 116.78, 103.94],
                            },
                        },
                    },
                    "postprocess": {
                        "transform": {
                            "LabelShift": {
                                "Param1": True,
                            },
                        },
                    },
                },
                "performance": {
                    "configs": {"cores_per_instance": 3, "num_of_instance": 2},
                    "dataloader": {
                        "batch_size": 1,
                        "dataset": {"ImageRecord": {"root": "/path/to/evaluation/dataset"}},
                        "transform": {
                            "ResizeCropImagenet": {
                                "height": 224,
                                "width": 224,
                                "mean_value": [123.68, 116.78, 103.94],
                            },
                        },
                    },
                },
            },
            "tuning": {
                "accuracy_criterion": {"relative": 0.01},
                "exit_policy": {"timeout": 0},
                "random_seed": 9527,
            },
            "graph_optimization": {
                "precisions": "bf16, fp32",
                "op_wise": {
                    "weight": {
                        "granularity": "per_channel",
                        "scheme": "asym",
                        "dtype": "bf16",
                        "algorithm": "minmax",
                    },
                    "activation": {
                        "granularity": "per_tensor",
                        "scheme": "sym",
                        "dtype": "int8",
                        "algorithm": "minmax",
                    },
                },
            },
            "pruning": {
                "magnitude": {
                    "weights": [1, 2, 3],
                    "method": "per_tensor",
                    "init_sparsity": 0.42,
                    "target_sparsity": 0.1337,
                    "start_epoch": 13,
                    "end_epoch": 888,
                },
                "start_epoch": 1,
                "end_epoch": 2,
                "frequency": 3,
                "init_sparsity": 0.4,
                "target_sparsity": 0.5,
            },
        }

    def test_config_constructor(self) -> None:
        """Test Config constructor."""
        config = Config(self.predefined_config)

        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(config.domain, "image_recognition")
        self.assertEqual(config.device, "cpu")

        self.assertEqual(config.model.name, "resnet50_v1_5")
        self.assertEqual(config.model.framework, "tensorflow")
        self.assertEqual(config.model.inputs, [])
        self.assertEqual(config.model.outputs, "softmax_tensor")

        self.assertIsNotNone(config.tuning)
        self.assertIsNotNone(config.tuning.strategy)
        self.assertEqual(config.tuning.strategy.name, "basic")
        self.assertIsNone(config.tuning.strategy.accuracy_weight)
        self.assertIsNone(config.tuning.strategy.latency_weight)
        self.assertIsNotNone(config.tuning.accuracy_criterion)
        self.assertEqual(config.tuning.accuracy_criterion.relative, 0.01)
        self.assertIsNone(config.tuning.accuracy_criterion.absolute)
        self.assertIsNone(config.tuning.objective)
        self.assertIsNotNone(config.tuning.exit_policy)
        self.assertEqual(config.tuning.exit_policy.timeout, 0)
        self.assertIsNone(config.tuning.exit_policy.max_trials)
        self.assertEqual(config.tuning.random_seed, 9527)
        self.assertIsNone(config.tuning.tensorboard)
        self.assertIsNotNone(config.tuning.workspace)
        self.assertIsNone(config.tuning.workspace.path)
        self.assertIsNone(config.tuning.workspace.resume)

        self.assertIsNotNone(config.quantization)
        self.assertIsNotNone(config.quantization.calibration)
        self.assertEqual(config.quantization.calibration.sampling_size, 100)
        self.assertIsNone(config.quantization.calibration.dataloader.last_batch)
        self.assertEqual(
            config.quantization.calibration.dataloader.batch_size,
            1,
        )  # Calibration batch size should be always set to 1
        self.assertIsNotNone(config.quantization.calibration.dataloader.dataset)
        self.assertEqual(
            config.quantization.calibration.dataloader.dataset.name,
            "ImageRecord",
        )
        self.assertDictEqual(
            config.quantization.calibration.dataloader.dataset.params,
            {"root": "/path/to/calibration/dataset"},
        )
        transform_name, transform = list(
            config.quantization.calibration.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "ResizeCropImagenet")
        self.assertDictEqual(
            transform.parameters,
            {"height": 224, "width": 224, "mean_value": [123.68, 116.78, 103.94]},
        )
        self.assertIsNone(config.quantization.calibration.dataloader.filter)
        self.assertIsNotNone(config.quantization.model_wise)
        self.assertIsNone(config.quantization.model_wise.weight)
        self.assertIsNotNone(config.quantization.model_wise.activation)
        self.assertIsNone(config.quantization.model_wise.activation.granularity)
        self.assertIsNone(config.quantization.model_wise.activation.scheme)
        self.assertIsNone(config.quantization.model_wise.activation.dtype)
        self.assertEqual(config.quantization.model_wise.activation.algorithm, "minmax")
        self.assertEqual(config.quantization.approach, "post_training_static_quant")
        self.assertIsNone(config.quantization.advance)

        self.assertIsNotNone(config.evaluation)
        self.assertIsNotNone(config.evaluation.accuracy)
        self.assertIsNotNone(config.evaluation.accuracy.metric)
        self.assertEqual(config.evaluation.accuracy.metric.name, "topk")
        self.assertEqual(config.evaluation.accuracy.metric.param, 1)
        self.assertIsNone(config.evaluation.accuracy.configs)

        self.assertIsNotNone(config.evaluation.accuracy.dataloader)
        self.assertIsNone(config.evaluation.accuracy.dataloader.last_batch)
        self.assertEqual(
            config.evaluation.accuracy.dataloader.batch_size,
            32,
        )
        self.assertIsNotNone(config.evaluation.accuracy.dataloader.dataset)
        self.assertEqual(
            config.evaluation.accuracy.dataloader.dataset.name,
            "ImageRecord",
        )
        self.assertDictEqual(
            config.evaluation.accuracy.dataloader.dataset.params,
            {"root": "/path/to/evaluation/dataset"},
        )
        transform_name, transform = list(
            config.evaluation.accuracy.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "ResizeCropImagenet")
        self.assertDictEqual(
            transform.parameters,
            {"height": 224, "width": 224, "mean_value": [123.68, 116.78, 103.94]},
        )
        self.assertIsNone(config.evaluation.accuracy.dataloader.filter)
        self.assertIsNotNone(config.evaluation.accuracy.postprocess)
        self.assertIsNotNone(config.evaluation.accuracy.postprocess.transform)
        self.assertEqual(
            {"Param1": True},
            config.evaluation.accuracy.postprocess.transform.LabelShift,
        )

        self.assertIsNotNone(config.evaluation.performance)
        self.assertEqual(config.evaluation.performance.warmup, 10)
        self.assertEqual(config.evaluation.performance.iteration, -1)
        self.assertIsNotNone(config.evaluation.performance.configs)
        self.assertEqual(
            config.evaluation.performance.configs.cores_per_instance,
            3,
        )
        self.assertEqual(
            config.evaluation.performance.configs.num_of_instance,
            2,
        )
        self.assertEqual(config.evaluation.performance.configs.inter_num_of_threads, None)
        self.assertEqual(config.evaluation.performance.configs.kmp_blocktime, None)
        self.assertIsNotNone(config.evaluation.performance.dataloader)
        self.assertIsNone(config.evaluation.performance.dataloader.last_batch)
        self.assertEqual(
            config.evaluation.performance.dataloader.batch_size,
            1,
        )
        self.assertIsNotNone(config.evaluation.performance.dataloader.dataset)
        self.assertEqual(
            config.evaluation.performance.dataloader.dataset.name,
            "ImageRecord",
        )
        self.assertDictEqual(
            config.evaluation.performance.dataloader.dataset.params,
            {
                "root": "/path/to/evaluation/dataset",
            },
        )
        transform_name, transform = list(
            config.quantization.calibration.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "ResizeCropImagenet")
        self.assertDictEqual(
            transform.parameters,
            {"height": 224, "width": 224, "mean_value": [123.68, 116.78, 103.94]},
        )
        self.assertIsNone(config.evaluation.performance.dataloader.filter)
        self.assertIsNone(config.evaluation.performance.postprocess)

        self.assertIsNotNone(config.pruning)
        self.assertIsNotNone(config.pruning.magnitude)
        self.assertEqual(config.pruning.start_epoch, 1)
        self.assertEqual(config.pruning.end_epoch, 2)
        self.assertEqual(config.pruning.frequency, 3)
        self.assertEqual(config.pruning.init_sparsity, 0.4)
        self.assertEqual(config.pruning.target_sparsity, 0.5)

        self.assertEqual(config.pruning.magnitude.weights, [1, 2, 3])
        self.assertEqual(config.pruning.magnitude.method, "per_tensor")
        self.assertEqual(config.pruning.magnitude.init_sparsity, 0.42)
        self.assertEqual(config.pruning.magnitude.target_sparsity, 0.1337)
        self.assertEqual(config.pruning.magnitude.start_epoch, 13)
        self.assertEqual(config.pruning.magnitude.end_epoch, 888)

    def test_config_constructor_with_empty_data(self) -> None:
        """Test Config constructor with empty data."""
        config = Config()

        self.assertEqual(config.model_path, "")
        self.assertIsNone(config.domain)
        self.assertIsNone(config.device)

        self.assertIsNone(config.model.name)
        self.assertEqual(config.model.framework, {})
        self.assertEqual(config.model.inputs, [])
        self.assertEqual(config.model.outputs, [])

        self.assertIsNotNone(config.tuning)
        self.assertIsNotNone(config.tuning.strategy)
        self.assertEqual(config.tuning.strategy.name, "basic")
        self.assertIsNone(config.tuning.strategy.accuracy_weight)
        self.assertIsNone(config.tuning.strategy.latency_weight)
        self.assertIsNotNone(config.tuning.accuracy_criterion)
        self.assertIsNone(config.tuning.accuracy_criterion.relative)
        self.assertIsNone(config.tuning.accuracy_criterion.absolute)
        self.assertIsNone(config.tuning.objective)
        self.assertIsNone(config.tuning.exit_policy)
        self.assertIsNone(config.tuning.random_seed)
        self.assertIsNone(config.tuning.tensorboard)
        self.assertIsNotNone(config.tuning.workspace)
        self.assertIsNone(config.tuning.workspace.path)
        self.assertIsNone(config.tuning.workspace.resume)

        self.assertIsNone(config.quantization)

        self.assertIsNone(config.evaluation)

        self.assertIsNone(config.pruning)

    def test_config_serializer(self) -> None:
        """Test Config serializer."""
        config = Config(self.predefined_config)
        result = config.serialize()

        self.assertDictEqual(
            result,
            {
                "domain": "image_recognition",
                "device": "cpu",
                "model": {
                    "name": "resnet50_v1_5",
                    "framework": "tensorflow",
                    "outputs": "softmax_tensor",
                },
                "quantization": {
                    "calibration": {
                        "sampling_size": 100,
                        "dataloader": {
                            "batch_size": 1,
                            "dataset": {
                                "ImageRecord": {
                                    "root": "/path/to/calibration/dataset",
                                },
                            },
                            "transform": {
                                "ResizeCropImagenet": {
                                    "height": 224,
                                    "width": 224,
                                    "mean_value": [123.68, 116.78, 103.94],
                                },
                            },
                        },
                    },
                    "model_wise": {"activation": {"algorithm": "minmax"}},
                    "approach": "post_training_static_quant",
                },
                "evaluation": {
                    "accuracy": {
                        "metric": {"topk": 1},
                        "dataloader": {
                            "batch_size": 32,
                            "dataset": {
                                "ImageRecord": {
                                    "root": "/path/to/evaluation/dataset",
                                },
                            },
                            "transform": {
                                "ResizeCropImagenet": {
                                    "height": 224,
                                    "width": 224,
                                    "mean_value": [123.68, 116.78, 103.94],
                                },
                            },
                        },
                        "postprocess": {
                            "transform": {
                                "LabelShift": {
                                    "Param1": True,
                                },
                            },
                        },
                    },
                    "performance": {
                        "warmup": 10,
                        "iteration": -1,
                        "configs": {
                            "cores_per_instance": 3,
                            "num_of_instance": 2,
                        },
                        "dataloader": {
                            "batch_size": 1,
                            "dataset": {
                                "ImageRecord": {
                                    "root": "/path/to/evaluation/dataset",
                                },
                            },
                            "transform": {
                                "ResizeCropImagenet": {
                                    "height": 224,
                                    "width": 224,
                                    "mean_value": [123.68, 116.78, 103.94],
                                },
                            },
                        },
                    },
                },
                "tuning": {
                    "strategy": {
                        "name": "basic",
                    },
                    "accuracy_criterion": {"relative": 0.01},
                    "exit_policy": {"timeout": 0},
                    "random_seed": 9527,
                },
                "graph_optimization": {
                    "precisions": "bf16,fp32",
                    "op_wise": {
                        "weight": {
                            "granularity": "per_channel",
                            "scheme": "asym",
                            "dtype": "bf16",
                            "algorithm": "minmax",
                        },
                        "activation": {
                            "granularity": "per_tensor",
                            "scheme": "sym",
                            "dtype": "int8",
                            "algorithm": "minmax",
                        },
                    },
                },
                "pruning": {
                    "magnitude": {
                        "weights": [1, 2, 3],
                        "method": "per_tensor",
                        "init_sparsity": 0.42,
                        "target_sparsity": 0.1337,
                        "start_epoch": 13,
                        "end_epoch": 888,
                    },
                    "start_epoch": 1,
                    "end_epoch": 2,
                    "frequency": 3,
                    "init_sparsity": 0.4,
                    "target_sparsity": 0.5,
                },
            },
        )

    def test_remove_dataloader(self) -> None:
        """Test remove_dataloader."""
        config = Config(self.predefined_config)

        self.assertEqual("ImageRecord", config.evaluation.accuracy.dataloader.dataset.name)
        self.assertEqual("ImageRecord", config.evaluation.performance.dataloader.dataset.name)
        self.assertEqual("ImageRecord", config.quantization.calibration.dataloader.dataset.name)

        config.remove_dataloader()

        self.assertIsNone(config.evaluation.accuracy.dataloader)
        self.assertIsNone(config.evaluation.performance.dataloader)
        self.assertIsNone(config.quantization.calibration.dataloader)

    def test_remove_dataloader_on_empty_config(self) -> None:
        """Test remove_dataloader on empty config."""
        config = Config()

        config.remove_dataloader()

        self.assertIsNone(config.evaluation)
        self.assertIsNone(config.quantization)

    def test_remove_accuracy_metric(self) -> None:
        """Test remove_accuracy_metric."""
        config = Config(self.predefined_config)

        self.assertEqual("topk", config.evaluation.accuracy.metric.name)

        config.remove_accuracy_metric()

        self.assertIsNone(config.evaluation.accuracy)

    def test_remove_accuracy_metric_on_empty_config(self) -> None:
        """Test remove_accuracy_metric on empty config."""
        config = Config()

        config.remove_accuracy_metric()

        self.assertIsNone(config.evaluation)

    def test_set_evaluation_dataloader(self) -> None:
        """Test set_evaluation_dataloader."""
        config = Config(self.predefined_config)

        config.set_evaluation_dataloader(
            {
                "name": "dataloader_name",
            },
        )

        self.assertEqual("dataloader_name", config.evaluation.accuracy.dataloader.dataset.name)
        self.assertEqual("dataloader_name", config.evaluation.performance.dataloader.dataset.name)

    def test_set_evaluation_dataloader_on_empty_config(self) -> None:
        """Test set_evaluation_dataloader on empty config."""
        config = Config()

        config.set_evaluation_dataloader(
            {
                "name": "dataloader_name",
            },
        )

        self.assertIsNone(config.evaluation)

    def test_set_evaluation_dataset_path(self) -> None:
        """Test set_evaluation_dataset_path."""
        config = Config(self.predefined_config)

        config.set_evaluation_dataset_path("new dataset path")

        self.assertEqual(
            "new dataset path",
            config.evaluation.accuracy.dataloader.dataset.params.get("root"),
        )
        self.assertEqual(
            "new dataset path",
            config.evaluation.performance.dataloader.dataset.params.get("root"),
        )
        self.assertEqual(
            "/path/to/calibration/dataset",
            config.quantization.calibration.dataloader.dataset.params.get("root"),
        )

    def test_set_evaluation_dataset_path_skips_no_dataset_location(self) -> None:
        """Test set_evaluation_dataset_path."""
        config = Config(self.predefined_config)

        config.set_evaluation_dataset_path("no_dataset_location")

        self.assertEqual(
            "/path/to/evaluation/dataset",
            config.evaluation.accuracy.dataloader.dataset.params.get("root"),
        )
        self.assertEqual(
            "/path/to/evaluation/dataset",
            config.evaluation.performance.dataloader.dataset.params.get("root"),
        )
        self.assertEqual(
            "/path/to/calibration/dataset",
            config.quantization.calibration.dataloader.dataset.params.get("root"),
        )

    def test_set_evaluation_dataset_path_on_empty_config(self) -> None:
        """Test set_evaluation_dataset_path on empty config."""
        config = Config()

        config.set_evaluation_dataset_path("new dataset path")

        self.assertIsNone(config.evaluation)

    def test_get_performance_configs(self) -> None:
        """Test get_performance_configs."""
        config = Config(self.predefined_config)

        self.assertIsNotNone(config.evaluation.performance.configs)
        self.assertEqual(config.evaluation.performance.configs, config.get_performance_configs())

    def test_get_performance_configs_on_empty_config(self) -> None:
        """Test get_performance_configs on empty config."""
        config = Config()

        self.assertIsNone(config.get_performance_configs())

    def test_set_performance_cores_per_instance(self) -> None:
        """Test set_performance_cores_per_instance."""
        config = Config(self.predefined_config)

        self.assertNotEqual(1234, config.get_performance_cores_per_instance())

        config.set_performance_cores_per_instance(1234)

        self.assertEqual(1234, config.get_performance_cores_per_instance())

    def test_set_performance_cores_per_instance_on_empty_config(self) -> None:
        """Test set_performance_cores_per_instance on empty config."""
        config = Config()

        config.set_performance_cores_per_instance(1234)

        self.assertIsNone(config.get_performance_cores_per_instance())

    def test_set_performance_num_of_instance(self) -> None:
        """Test set_performance_num_of_instance."""
        config = Config(self.predefined_config)

        self.assertNotEqual(1234, config.get_performance_num_of_instance())

        config.set_performance_num_of_instance(1234)

        self.assertEqual(1234, config.get_performance_num_of_instance())

    def test_set_performance_num_of_instance_on_empty_config(self) -> None:
        """Test set_performance_num_of_instance on empty config."""
        config = Config()

        config.set_performance_num_of_instance(1234)

        self.assertIsNone(config.get_performance_num_of_instance())

    def test_set_performance_batch_size(self) -> None:
        """Test set_performance_batch_size."""
        config = Config(self.predefined_config)

        config.set_performance_batch_size(1234)

        self.assertEqual(1234, config.evaluation.performance.dataloader.batch_size)

    def test_set_performance_batch_size_on_empty_config(self) -> None:
        """Test set_performance_batch_size."""
        config = Config()

        config.set_performance_batch_size(1234)

        self.assertIsNone(config.evaluation)

    def test_set_quantization_dataloader(self) -> None:
        """Test set_quantization_dataloader."""
        config = Config(self.predefined_config)

        config.set_quantization_dataloader(
            {
                "name": "dataloader_name",
            },
        )

        self.assertEqual(
            "dataloader_name",
            config.quantization.calibration.dataloader.dataset.name,
        )

    def test_set_quantization_dataloader_on_empty_config(self) -> None:
        """Test set_quantization_dataloader on empty config."""
        config = Config()

        config.set_quantization_dataloader(
            {
                "name": "dataloader_name",
            },
        )

        self.assertIsNone(config.quantization)

    def test_set_quantization_dataset_path(self) -> None:
        """Test set_quantization_dataset_path."""
        config = Config(self.predefined_config)

        config.set_quantization_dataset_path("new dataset path")

        self.assertEqual(
            "/path/to/evaluation/dataset",
            config.evaluation.accuracy.dataloader.dataset.params.get("root"),
        )
        self.assertEqual(
            "/path/to/evaluation/dataset",
            config.evaluation.performance.dataloader.dataset.params.get("root"),
        )
        self.assertEqual(
            "new dataset path",
            config.quantization.calibration.dataloader.dataset.params.get("root"),
        )

    def test_set_quantization_dataset_path_skips_no_dataset_location(self) -> None:
        """Test set_quantization_dataset_path."""
        config = Config(self.predefined_config)

        config.set_quantization_dataset_path("no_dataset_location")

        self.assertEqual(
            "/path/to/evaluation/dataset",
            config.evaluation.accuracy.dataloader.dataset.params.get("root"),
        )
        self.assertEqual(
            "/path/to/evaluation/dataset",
            config.evaluation.performance.dataloader.dataset.params.get("root"),
        )
        self.assertEqual(
            "/path/to/calibration/dataset",
            config.quantization.calibration.dataloader.dataset.params.get("root"),
        )

    def test_set_quantization_dataset_path_on_empty_config(self) -> None:
        """Test set_quantization_dataset_path on empty config."""
        config = Config()

        config.set_quantization_dataset_path("new dataset path")

        self.assertIsNone(config.quantization)

    def test_set_workspace(self) -> None:
        """Test set_workspace."""
        config = Config(self.predefined_config)

        config.set_workspace("new/workspace/path")

        self.assertEqual("new/workspace/path", config.tuning.workspace.path)

    def test_set_accuracy_goal(self) -> None:
        """Test set_accuracy_goal."""
        config = Config(self.predefined_config)

        config.set_accuracy_goal(1234)

        self.assertEqual(1234, config.tuning.accuracy_criterion.relative)

    def test_set_accuracy_goal_to_negative_value(self) -> None:
        """Test set_accuracy_goal."""
        config = Config(self.predefined_config)

        original_accuracy_goal = config.tuning.accuracy_criterion.relative

        with self.assertRaises(ClientErrorException):
            config.set_accuracy_goal(-1234)
        self.assertEqual(original_accuracy_goal, config.tuning.accuracy_criterion.relative)

    def test_set_accuracy_goal_on_empty_config(self) -> None:
        """Test set_accuracy_goal."""
        config = Config()

        config.set_accuracy_goal(1234)

        self.assertIsNone(config.tuning.accuracy_criterion.relative)

    def test_set_accuracy_metric(self) -> None:
        """Test set_accuracy_metric."""
        config = Config(self.predefined_config)

        config.set_accuracy_metric({"metric": "new metric", "metric_param": {"param1": True}})

        self.assertEqual("new metric", config.evaluation.accuracy.metric.name)
        self.assertEqual({"param1": True}, config.evaluation.accuracy.metric.param)

    def test_set_accuracy_metric_on_empty_config(self) -> None:
        """Test set_accuracy_metric."""
        config = Config()

        config.set_accuracy_metric({"metric": "new metric", "metric_param": {"param1": True}})

        self.assertIsNone(config.evaluation)

    def test_set_transform(self) -> None:
        """Test set_transform."""
        config = Config(self.predefined_config)

        config.set_transform(
            [
                {"name": "Some transform1", "params": {"param12": True}},
                {"name": "SquadV1", "params": {"param1": True}},
                {"name": "Some transform2", "params": {"param123": True}},
            ],
        )

        self.assertEqual(
            {
                "SquadV1": {"param1": True},
            },
            config.evaluation.accuracy.postprocess.transform,
        )

        self.assertEqual(
            ["Some transform1", "Some transform2"],
            list(config.quantization.calibration.dataloader.transform.keys()),
        )
        self.assertEqual(
            ["Some transform1", "Some transform2"],
            list(config.evaluation.accuracy.dataloader.transform.keys()),
        )
        self.assertEqual(
            ["Some transform1", "Some transform2"],
            list(config.evaluation.performance.dataloader.transform.keys()),
        )

    def test_set_transform_without_suqadV1(self) -> None:
        """Test set_transform."""
        config = Config(self.predefined_config)

        config.set_transform(
            [
                {"name": "Some transform1", "params": {"param12": True}},
                {"name": "Some transform2", "params": {"param123": True}},
            ],
        )

        self.assertEqual(
            {"Param1": True},
            config.evaluation.accuracy.postprocess.transform.LabelShift,
        )

        self.assertEqual(
            ["Some transform1", "Some transform2"],
            list(config.quantization.calibration.dataloader.transform.keys()),
        )
        self.assertEqual(
            ["Some transform1", "Some transform2"],
            list(config.evaluation.accuracy.dataloader.transform.keys()),
        )
        self.assertEqual(
            ["Some transform1", "Some transform2"],
            list(config.evaluation.performance.dataloader.transform.keys()),
        )

    def test_set_transform_on_empty_config(self) -> None:
        """Test set_transform."""
        config = Config()

        config.set_transform(
            [
                {"name": "Some transform1", "params": {"param12": True}},
                {"name": "SquadV1", "params": {"param1": True}},
                {"name": "Some transform2", "params": {"param123": True}},
            ],
        )

        self.assertIsNone(config.evaluation)
        self.assertIsNone(config.quantization)

    def test_set_quantization_approach(self) -> None:
        """Test set_quantization_approach."""
        config = Config(self.predefined_config)

        config.set_quantization_approach("Some quantization approach")

        self.assertEqual("Some quantization approach", config.quantization.approach)

    def test_set_quantization_approach_on_empty_config(self) -> None:
        """Test set_quantization_approach."""
        config = Config()

        config.set_quantization_approach("Some quantization approach")

        self.assertIsNone(config.quantization)

    def test_set_model_path(self) -> None:
        """Test set_model_path."""
        config = Config()

        config.set_model_path("new/model/path")

        self.assertEqual("new/model/path", config.model_path)

    def test_set_inputs(self) -> None:
        """Test set_model_path."""
        config = Config()

        config.set_inputs(["input1", "input2"])

        self.assertEqual(["input1", "input2"], config.model.inputs)

    def test_set_outputs(self) -> None:
        """Test set_model_path."""
        config = Config()

        config.set_outputs(["output1", "output2"])

        self.assertEqual(["output1", "output2"], config.model.outputs)

    def test_set_quantization_sampling_size(self) -> None:
        """Test set_model_path."""
        config = Config(self.predefined_config)

        config.set_quantization_sampling_size("new sampling size")

        self.assertEqual("new sampling size", config.quantization.calibration.sampling_size)

    def test_set_quantization_sampling_size_on_empty_config(self) -> None:
        """Test set_model_path."""
        config = Config()

        config.set_quantization_sampling_size("new sampling size")

        self.assertIsNone(config.quantization)

    def test_set_performance_warmup(self) -> None:
        """Test set_performance_warmup."""
        config = Config(self.predefined_config)

        config.set_performance_warmup(1234)

        self.assertEqual(1234, config.evaluation.performance.warmup)

    def test_set_performance_warmup_to_negative_value(self) -> None:
        """Test set_performance_warmup."""
        config = Config(self.predefined_config)

        original_performance_warmup = config.evaluation.performance.warmup

        with self.assertRaises(ClientErrorException):
            config.set_performance_warmup(-1234)
        self.assertEqual(original_performance_warmup, config.evaluation.performance.warmup)

    def test_set_performance_warmup_on_empty_config(self) -> None:
        """Test set_performance_warmup."""
        config = Config()

        config.set_performance_warmup(1234)

        self.assertIsNone(config.evaluation)

    def test_set_performance_iterations(self) -> None:
        """Test set_performance_iterations."""
        config = Config(self.predefined_config)

        config.set_performance_iterations(1234)

        self.assertEqual(1234, config.evaluation.performance.iteration)

    def test_set_performance_iterations_to_negative_value(self) -> None:
        """Test set_performance_iterations."""
        config = Config(self.predefined_config)

        original_performance_iterations = config.evaluation.performance.iteration

        with self.assertRaises(ClientErrorException):
            config.set_performance_iterations(-1234)
        self.assertEqual(original_performance_iterations, config.evaluation.performance.iteration)

    def test_set_performance_iterations_on_empty_config(self) -> None:
        """Test set_performance_iterations."""
        config = Config()

        config.set_performance_iterations(1234)

        self.assertIsNone(config.evaluation)

    @patch("lpot.ux.utils.workload.config.load_precisions_config")
    def test_set_optimization_precision_to_unknown_precision(
        self,
        mocked_load_precisions_config: MagicMock,
    ) -> None:
        """Test set_optimization_precision."""
        mocked_load_precisions_config.return_value = {
            "framework_foo": [
                {
                    "name": "precision1",
                },
                {
                    "name": "precision2",
                },
                {
                    "name": "precision3",
                },
            ],
            "framework_bar": [
                {
                    "name": "precision1",
                },
            ],
        }

        config = Config(self.predefined_config)

        with self.assertRaisesRegex(
            ClientErrorException,
            "Precision unknown_precision is not supported "
            "in graph optimization for framework framework_foo.",
        ):
            config.set_optimization_precision("framework_foo", "unknown_precision")
        mocked_load_precisions_config.assert_called_once()

    @patch("lpot.ux.utils.workload.config.load_precisions_config")
    def test_set_optimization_precision_to_unknown_framework(
        self,
        mocked_load_precisions_config: MagicMock,
    ) -> None:
        """Test set_optimization_precision."""
        mocked_load_precisions_config.return_value = {
            "framework_foo": [
                {
                    "name": "precision1",
                },
                {
                    "name": "precision2",
                },
                {
                    "name": "precision3",
                },
            ],
            "framework_bar": [
                {
                    "name": "precision1",
                },
            ],
        }
        config = Config(self.predefined_config)

        with self.assertRaisesRegex(
            ClientErrorException,
            "Precision precision1 is not supported "
            "in graph optimization for framework framework_baz.",
        ):
            config.set_optimization_precision("framework_baz", "precision1")
        mocked_load_precisions_config.assert_called_once()

    @patch("lpot.ux.utils.workload.config.load_precisions_config")
    def test_set_optimization_precision(
        self,
        mocked_load_precisions_config: MagicMock,
    ) -> None:
        """Test set_optimization_precision."""
        mocked_load_precisions_config.return_value = {
            "framework_foo": [
                {
                    "name": "precision1",
                },
                {
                    "name": "precision2",
                },
                {
                    "name": "precision3",
                },
            ],
            "framework_bar": [
                {
                    "name": "precision1",
                },
            ],
        }

        config = Config(self.predefined_config)

        config.set_optimization_precision("framework_foo", "precision2")

        self.assertEqual("precision2", config.graph_optimization.precisions)
        mocked_load_precisions_config.assert_called_once()

    @patch("lpot.ux.utils.workload.config.load_precisions_config")
    def test_set_optimization_precision_on_empty_config(
        self,
        mocked_load_precisions_config: MagicMock,
    ) -> None:
        """Test set_optimization_precision."""
        mocked_load_precisions_config.return_value = {
            "framework_foo": [
                {
                    "name": "precision1",
                },
                {
                    "name": "precision2",
                },
                {
                    "name": "precision3",
                },
            ],
            "framework_bar": [
                {
                    "name": "precision1",
                },
            ],
        }

        config = Config()

        config.set_optimization_precision("framework_foo", "precision2")

        self.assertEqual("precision2", config.graph_optimization.precisions)
        mocked_load_precisions_config.assert_called_once()

    def test_load(self) -> None:
        """Test load."""
        config = Config()

        read_yaml = yaml.dump(self.predefined_config, sort_keys=False)

        with patch(
            "lpot.ux.utils.workload.config.open",
            mock_open(read_data=read_yaml),
        ) as mocked_open:
            config.load("path to yaml file")

            mocked_open.assert_called_once_with("path to yaml file")

        expected = Config(self.predefined_config)

        self.assertEqual(expected.serialize(), config.serialize())

    def test_dump(self) -> None:
        """Test dump."""
        config = Config(self.predefined_config)

        expected_yaml = yaml.dump(
            config.serialize(),
            indent=4,
            default_flow_style=None,
            sort_keys=False,
        )

        with patch("lpot.ux.utils.workload.config.open", mock_open()) as mocked_open:
            config.dump("path to yaml file")

            mocked_open.assert_called_once_with("path to yaml file", "w")
            mocked_open().write.assert_called_once_with(expected_yaml)


if __name__ == "__main__":
    unittest.main()
