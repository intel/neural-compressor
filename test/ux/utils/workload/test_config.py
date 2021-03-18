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
from unittest.mock import MagicMock, patch

from lpot.ux.utils.workload.config import Config

predefined_config = {
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
                    "ParseDecodeImagenet": None,
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
                    "ParseDecodeImagenet": None,
                    "ResizeCropImagenet": {
                        "height": 224,
                        "width": 224,
                        "mean_value": [123.68, 116.78, 103.94],
                    },
                },
            },
        },
        "performance": {
            "configs": {"cores_per_instance": 4, "num_of_instance": 7},
            "dataloader": {
                "batch_size": 1,
                "dataset": {"ImageRecord": {"root": "/path/to/evaluation/dataset"}},
                "transform": {
                    "ParseDecodeImagenet": None,
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
}


class TestConfig(unittest.TestCase):
    """Config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Config test constructor."""
        super().__init__(*args, **kwargs)

    @patch("psutil.cpu_count")
    def test_config_constructor(self, mock_cpu_count: MagicMock) -> None:
        """Test Config constructor."""
        mock_cpu_count.return_value = 8

        config = Config(predefined_config)

        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(config.domain, "image_recognition")
        self.assertEqual(config.device, "cpu")

        self.assertIsNotNone(config.model)
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
        self.assertEqual(transform_name, "ParseDecodeImagenet")
        self.assertIsNone(transform.parameters)
        transform_name, transform = list(
            config.quantization.calibration.dataloader.transform.items(),
        )[1]
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
        self.assertEqual(transform_name, "ParseDecodeImagenet")
        self.assertIsNone(transform.parameters)
        transform_name, transform = list(
            config.evaluation.accuracy.dataloader.transform.items(),
        )[1]
        self.assertEqual(transform_name, "ResizeCropImagenet")
        self.assertDictEqual(
            transform.parameters,
            {"height": 224, "width": 224, "mean_value": [123.68, 116.78, 103.94]},
        )
        self.assertIsNone(config.evaluation.accuracy.dataloader.filter)
        self.assertIsNone(config.evaluation.accuracy.postprocess)

        self.assertIsNotNone(config.evaluation.performance)
        self.assertEqual(config.evaluation.performance.warmup, 10)
        self.assertEqual(config.evaluation.performance.iteration, -1)
        self.assertIsNotNone(config.evaluation.performance.configs)
        self.assertEqual(
            config.evaluation.performance.configs.cores_per_instance,
            4,
        )  # Cores per instance should be equal to 4
        self.assertEqual(
            config.evaluation.performance.configs.num_of_instance,
            2,
        )  # 8 cores / 4 instances = 2
        self.assertEqual(config.evaluation.performance.configs.inter_num_of_threads, 4)
        self.assertEqual(config.evaluation.performance.configs.kmp_blocktime, 1)
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
        self.assertEqual(transform_name, "ParseDecodeImagenet")
        self.assertIsNone(transform.parameters)
        transform_name, transform = list(
            config.quantization.calibration.dataloader.transform.items(),
        )[1]
        self.assertEqual(transform_name, "ResizeCropImagenet")
        self.assertDictEqual(
            transform.parameters,
            {"height": 224, "width": 224, "mean_value": [123.68, 116.78, 103.94]},
        )
        self.assertIsNone(config.evaluation.performance.dataloader.filter)
        self.assertIsNone(config.evaluation.performance.postprocess)

        self.assertIsNone(config.pruning)

    @patch("psutil.cpu_count")
    def test_config_serializer(self, mock_cpu_count: MagicMock) -> None:
        """Test Config serializer."""
        mock_cpu_count.return_value = 8

        config = Config(predefined_config)
        result = config.serialize()
        print(result)

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
                                "ParseDecodeImagenet": None,
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
                                "ParseDecodeImagenet": None,
                                "ResizeCropImagenet": {
                                    "height": 224,
                                    "width": 224,
                                    "mean_value": [123.68, 116.78, 103.94],
                                },
                            },
                        },
                    },
                    "performance": {
                        "warmup": 10,
                        "iteration": -1,
                        "configs": {
                            "cores_per_instance": 4,
                            "num_of_instance": 2,
                            "inter_num_of_threads": 4,
                            "kmp_blocktime": 1,
                        },
                        "dataloader": {
                            "batch_size": 1,
                            "dataset": {
                                "ImageRecord": {
                                    "root": "/path/to/evaluation/dataset",
                                },
                            },
                            "transform": {
                                "ParseDecodeImagenet": None,
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
            },
        )


if __name__ == "__main__":
    unittest.main()
