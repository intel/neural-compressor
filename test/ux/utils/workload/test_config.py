# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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
from copy import deepcopy
from unittest.mock import MagicMock, mock_open, patch

import schema
import yaml

from neural_compressor.conf.config import Pruner
from neural_compressor.conf.config import schema as inc_config_schema
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.yaml_utils import float_representer, pruner_representer

pruner_config = {
    "initial_sparsity": 0.0,
    "target_sparsity": 0.97,
    "start_epoch": 0,
    "end_epoch": 2,
    "prune_type": "basic_magnitude",
    "update_frequency": 0.1,
    "names": ["layer1.0.conv1.weight"],
}
pruner = Pruner(**pruner_config)

yaml.add_constructor("!Pruner", Pruner, yaml.SafeLoader)


class TestConfig(unittest.TestCase):
    """Config tests."""

    def setUp(self) -> None:
        """Prepare test."""
        super().setUp()
        self.predefined_config = {
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
                            "LabelShift": -1,
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
                "diagnosis": {
                    "diagnosis_after_tuning": True,
                    "op_list": ["op_1", "op_2"],
                    "iteration_list": [1, 2],
                    "inspect_type": "weight",
                    "save_to_disk": True,
                    "save_path": "/path/to/save/diagnosis/results",
                },
            },
            "graph_optimization": {
                "precisions": "bf16, fp32",
                "op_wise": {
                    "weight": {
                        "dtype": "bf16",
                    },
                    "activation": {
                        "dtype": "fp32",
                    },
                },
            },
            "pruning": {
                "train": {
                    "optimizer": {
                        "Adam": {
                            "learning_rate": 0.123,
                            "beta_1": 0.99,
                            "beta_2": 0.9999,
                            "epsilon": 1e-06,
                            "amsgrad": True,
                        },
                    },
                    "criterion": {
                        "CrossEntropyLoss": {
                            "reduction": "auto",
                            "from_logits": True,
                        },
                        "SparseCategoricalCrossentropy": {
                            "reduction": "sum",
                            "from_logits": True,
                        },
                        "KnowledgeDistillationLoss": {
                            "temperature": 1.0,
                            "loss_types": ["CE"],
                            "loss_weights": [0.3, 0.7],
                        },
                    },
                    "dataloader": {
                        "batch_size": 32,
                        "dataset": {"ImageRecord": {"root": "/path/to/pruning/dataset"}},
                        "transform": {
                            "ResizeCropImagenet": {
                                "height": 224,
                                "width": 224,
                                "mean_value": [123.68, 116.78, 103.94],
                            },
                        },
                    },
                    "epoch": 5,
                    "start_epoch": 1,
                    "end_epoch": 2,
                    "iteration": 10,
                    "frequency": 3,
                    "execution_mode": "graph",
                    "postprocess": {
                        "transform": {
                            "LabelShift": -1,
                            "Collect": {
                                "length": 1,
                            },
                            "SquadV1": {
                                "label_file": "/path/to/label_file",
                                "vocab_file": "/path/to/vocab_file",
                                "do_lower_case": False,
                            },
                        },
                    },
                    "hostfile": "/some/file",
                },
                "approach": {
                    "weight_compression": {
                        "initial_sparsity": 0.042,
                        "target_sparsity": 0.1337,
                        "start_epoch": 13,
                        "end_epoch": 888,
                        "pruners": [
                            pruner,
                        ],
                    },
                },
            },
        }

    def test_config_constructor(self) -> None:
        """Test Config constructor."""
        config = Config(self.predefined_config)

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
        self.assertIsNone(config.tuning.multi_objectives)
        self.assertIsNotNone(config.tuning.exit_policy)
        self.assertEqual(config.tuning.exit_policy.timeout, 0)
        self.assertIsNone(config.tuning.exit_policy.max_trials)
        self.assertEqual(config.tuning.random_seed, 9527)
        self.assertIsNone(config.tuning.tensorboard)
        self.assertIsNone(config.tuning.workspace)

        self.assertIsNotNone(config.tuning.diagnosis)
        self.assertTrue(config.tuning.diagnosis.diagnosis_after_tuning)
        self.assertListEqual(config.tuning.diagnosis.op_list, ["op_1", "op_2"])
        self.assertListEqual(config.tuning.diagnosis.iteration_list, [1, 2])
        self.assertEqual(config.tuning.diagnosis.inspect_type, "weight")
        self.assertTrue(config.tuning.diagnosis.save_to_disk)
        self.assertEqual(config.tuning.diagnosis.save_path, "/path/to/save/diagnosis/results")

        self.assertIsNotNone(config.quantization)
        self.assertIsNotNone(config.quantization.calibration)
        self.assertEqual(config.quantization.calibration.sampling_size, 100)
        self.assertIsNone(config.quantization.calibration.dataloader.last_batch)
        self.assertEqual(config.quantization.calibration.dataloader.batch_size, 10)
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
        self.assertEqual(-1, config.evaluation.accuracy.postprocess.transform.LabelShift)

        self.assertIsNotNone(config.evaluation.performance)
        self.assertEqual(config.evaluation.performance.warmup, 5)
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
        self.assertEqual(1, config.evaluation.performance.configs.kmp_blocktime)
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
        self.assertIsNotNone(config.pruning.train)
        self.assertIsNotNone(config.pruning.train.optimizer)
        self.assertIsNotNone(config.pruning.train.optimizer.Adam)
        self.assertEqual(config.pruning.train.optimizer.Adam.learning_rate, 0.123)
        self.assertEqual(config.pruning.train.optimizer.Adam.beta_1, 0.99)
        self.assertEqual(config.pruning.train.optimizer.Adam.beta_2, 0.9999)
        self.assertEqual(config.pruning.train.optimizer.Adam.epsilon, 1e-06)
        self.assertEqual(config.pruning.train.optimizer.Adam.amsgrad, True)
        self.assertIsNotNone(config.pruning.train.criterion)
        self.assertIsNotNone(config.pruning.train.criterion.CrossEntropyLoss)
        self.assertEqual(config.pruning.train.criterion.CrossEntropyLoss.reduction, "auto")
        self.assertEqual(config.pruning.train.criterion.CrossEntropyLoss.from_logits, True)
        self.assertIsNotNone(config.pruning.train.criterion.SparseCategoricalCrossentropy)
        self.assertEqual(
            config.pruning.train.criterion.SparseCategoricalCrossentropy.reduction,
            "sum",
        )
        self.assertEqual(
            config.pruning.train.criterion.SparseCategoricalCrossentropy.from_logits,
            True,
        )
        self.assertIsNotNone(config.pruning.train.criterion.KnowledgeDistillationLoss)
        self.assertEqual(config.pruning.train.criterion.KnowledgeDistillationLoss.temperature, 1.0)
        self.assertListEqual(
            config.pruning.train.criterion.KnowledgeDistillationLoss.loss_types,
            ["CE"],
        )
        self.assertListEqual(
            config.pruning.train.criterion.KnowledgeDistillationLoss.loss_weights,
            [0.3, 0.7],
        )
        self.assertIsNotNone(config.pruning.train.dataloader)
        self.assertEqual(
            config.pruning.train.dataloader.dataset.name,
            "ImageRecord",
        )
        self.assertDictEqual(
            config.pruning.train.dataloader.dataset.params,
            {"root": "/path/to/pruning/dataset"},
        )
        transform_name, transform = list(
            config.pruning.train.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "ResizeCropImagenet")
        self.assertDictEqual(
            transform.parameters,
            {"height": 224, "width": 224, "mean_value": [123.68, 116.78, 103.94]},
        )
        self.assertIsNone(config.pruning.train.dataloader.filter)
        self.assertEqual(config.pruning.train.epoch, 5)
        self.assertEqual(config.pruning.train.start_epoch, 1)
        self.assertEqual(config.pruning.train.end_epoch, 2)
        self.assertEqual(config.pruning.train.iteration, 10)
        self.assertEqual(config.pruning.train.frequency, 3)
        self.assertEqual(config.pruning.train.execution_mode, "graph")
        self.assertIsNotNone(config.pruning.train.postprocess)
        self.assertEqual(config.pruning.train.hostfile, "/some/file")

        self.assertIsNotNone(config.pruning.approach)
        self.assertIsNotNone(config.pruning.approach.weight_compression)
        self.assertEqual(config.pruning.approach.weight_compression.initial_sparsity, 0.042)
        self.assertEqual(config.pruning.approach.weight_compression.target_sparsity, 0.1337)
        self.assertEqual(config.pruning.approach.weight_compression.start_epoch, 13)
        self.assertEqual(config.pruning.approach.weight_compression.end_epoch, 888)
        self.assertEqual(len(config.pruning.approach.weight_compression.pruners), 1)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_config_constructor_with_empty_data(self) -> None:
        """Test Config constructor with empty data."""
        config = Config()

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
        self.assertEqual(config.tuning.accuracy_criterion.relative, 0.1)
        self.assertIsNone(config.tuning.accuracy_criterion.absolute)
        self.assertIsNone(config.tuning.multi_objectives)
        self.assertIsNone(config.tuning.exit_policy)
        self.assertIsNone(config.tuning.random_seed)
        self.assertIsNone(config.tuning.tensorboard)
        self.assertIsNone(config.tuning.workspace)

        self.assertIsNone(config.quantization)

        self.assertIsNone(config.evaluation)

        self.assertIsNone(config.pruning)

        serialized_config = config.serialize()
        with self.assertRaises(schema.SchemaError):
            inc_config_schema.validate(serialized_config)

    def test_config_serializer(self) -> None:
        """Test Config serializer."""
        config = Config(self.predefined_config)
        result = config.serialize()
        self.maxDiff = None
        self.assertDictEqual(
            result,
            {
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
                                "LabelShift": -1,
                            },
                        },
                    },
                    "performance": {
                        "warmup": 5,
                        "iteration": -1,
                        "configs": {
                            "cores_per_instance": 3,
                            "num_of_instance": 2,
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
                    "diagnosis": {
                        "diagnosis_after_tuning": True,
                        "op_list": ["op_1", "op_2"],
                        "iteration_list": [1, 2],
                        "inspect_type": "weight",
                        "save_to_disk": True,
                        "save_path": "/path/to/save/diagnosis/results",
                    },
                },
                "graph_optimization": {
                    "precisions": "bf16,fp32",
                    "op_wise": {
                        "weight": {
                            "dtype": "bf16",
                        },
                        "activation": {
                            "dtype": "fp32",
                        },
                    },
                },
                "pruning": {
                    "train": {
                        "optimizer": {
                            "Adam": {
                                "learning_rate": 0.123,
                                "beta_1": 0.99,
                                "beta_2": 0.9999,
                                "epsilon": 1e-06,
                                "amsgrad": True,
                            },
                        },
                        "criterion": {
                            "CrossEntropyLoss": {
                                "reduction": "auto",
                                "from_logits": True,
                            },
                            "SparseCategoricalCrossentropy": {
                                "reduction": "sum",
                                "from_logits": True,
                            },
                            "KnowledgeDistillationLoss": {
                                "temperature": 1.0,
                                "loss_types": ["CE"],
                                "loss_weights": [0.3, 0.7],
                            },
                        },
                        "dataloader": {
                            "batch_size": 32,
                            "dataset": {"ImageRecord": {"root": "/path/to/pruning/dataset"}},
                            "transform": {
                                "ResizeCropImagenet": {
                                    "height": 224,
                                    "width": 224,
                                    "mean_value": [123.68, 116.78, 103.94],
                                },
                            },
                        },
                        "epoch": 5,
                        "start_epoch": 1,
                        "end_epoch": 2,
                        "iteration": 10,
                        "frequency": 3,
                        "execution_mode": "graph",
                        "postprocess": {
                            "transform": {
                                "LabelShift": -1,
                                "Collect": {
                                    "length": 1,
                                },
                                "SquadV1": {
                                    "label_file": "/path/to/label_file",
                                    "vocab_file": "/path/to/vocab_file",
                                    "do_lower_case": False,
                                },
                            },
                        },
                        "hostfile": "/some/file",
                    },
                    "approach": {
                        "weight_compression": {
                            "initial_sparsity": 0.042,
                            "target_sparsity": 0.1337,
                            "start_epoch": 13,
                            "end_epoch": 888,
                            "pruners": [
                                pruner,
                            ],
                        },
                    },
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

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

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

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

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

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

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

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

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

        config.set_accuracy_and_performance_batch_sizes(1234)

        self.assertEqual(1234, config.evaluation.accuracy.dataloader.batch_size)
        self.assertEqual(1234, config.evaluation.performance.dataloader.batch_size)

    def test_set_performance_batch_size_on_empty_config(self) -> None:
        """Test set_performance_batch_size."""
        config = Config()

        config.set_accuracy_and_performance_batch_sizes(1234)

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

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

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

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_quantization_dataset_path_on_empty_config(self) -> None:
        """Test set_quantization_dataset_path on empty config."""
        config = Config()

        config.set_quantization_dataset_path("new dataset path")

        self.assertIsNone(config.quantization)

    def test_set_quantization_batch_size(self) -> None:
        """Test set_quantization_batch_size."""
        config = Config(self.predefined_config)

        config.set_quantization_batch_size("31337")

        self.assertEqual(31337, config.quantization.calibration.dataloader.batch_size)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_workspace(self) -> None:
        """Test set_workspace."""
        config = Config(self.predefined_config)

        config.set_workspace("new/workspace/path")

        self.assertEqual("new/workspace/path", config.tuning.workspace.path)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_accuracy_goal(self) -> None:
        """Test set_accuracy_goal."""
        config = Config(self.predefined_config)

        config.set_accuracy_goal(1234)

        self.assertEqual(1234, config.tuning.accuracy_criterion.relative)
        self.assertIsNone(config.tuning.accuracy_criterion.absolute)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_absolute_accuracy_goal(self) -> None:
        """Test set_accuracy_goal with absolute value."""
        predefined_config = deepcopy(self.predefined_config)
        if predefined_config.get("tuning", {}).get("accuracy_criterion", None) is not None:
            predefined_config["tuning"]["accuracy_criterion"] = {"absolute": 0.01}
        config = Config(predefined_config)

        config.set_accuracy_goal(0.1234)

        self.assertIsNone(config.tuning.accuracy_criterion.relative)
        self.assertEqual(0.1234, config.tuning.accuracy_criterion.absolute)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_accuracy_goal_to_negative_value(self) -> None:
        """Test set_accuracy_goal."""
        config = Config(self.predefined_config)

        original_accuracy_goal = config.tuning.accuracy_criterion.relative

        with self.assertRaises(ClientErrorException):
            config.set_accuracy_goal(-1234)
        self.assertEqual(original_accuracy_goal, config.tuning.accuracy_criterion.relative)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_accuracy_goal_on_empty_config(self) -> None:
        """Test set_accuracy_goal."""
        config = Config()

        config.set_accuracy_goal(1234)

        self.assertEqual(config.tuning.accuracy_criterion.relative, 1234)
        self.assertIsNone(config.tuning.accuracy_criterion.absolute)

    def test_set_accuracy_metric(self) -> None:
        """Test set_accuracy_metric."""
        config = Config(self.predefined_config)

        config.set_accuracy_metric({"metric": "topk", "metric_param": 1})

        self.assertEqual("topk", config.evaluation.accuracy.metric.name)
        self.assertEqual(1, config.evaluation.accuracy.metric.param)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

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

    def test_set_transform_without_squadV1(self) -> None:
        """Test set_transform."""
        config = Config(self.predefined_config)

        config.set_transform(
            [
                {"name": "Some transform1", "params": {"param12": True}},
                {"name": "Some transform2", "params": {"param123": True}},
            ],
        )

        self.assertEqual(-1, config.evaluation.accuracy.postprocess.transform.LabelShift)

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

        config.set_quantization_approach("quant_aware_training")

        self.assertEqual("quant_aware_training", config.quantization.approach)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_quantization_approach_on_empty_config(self) -> None:
        """Test set_quantization_approach."""
        config = Config()

        config.set_quantization_approach("Some quantization approach")

        self.assertIsNone(config.quantization)

    def test_set_inputs(self) -> None:
        """Test set_inputs."""
        config = Config()

        config.set_inputs(["input1", "input2"])

        self.assertEqual(["input1", "input2"], config.model.inputs)

    def test_set_outputs(self) -> None:
        """Test set_outputs."""
        config = Config()

        config.set_outputs(["output1", "output2"])

        self.assertEqual(["output1", "output2"], config.model.outputs)

    def test_set_quantization_sampling_size(self) -> None:
        """Test set_quantization_sampling_size."""
        config = Config(self.predefined_config)

        config.set_quantization_sampling_size("new sampling size")

        self.assertEqual("new sampling size", config.quantization.calibration.sampling_size)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_quantization_sampling_size_on_empty_config(self) -> None:
        """Test set_quantization_sampling_size."""
        config = Config()

        config.set_quantization_sampling_size("new sampling size")

        self.assertIsNone(config.quantization)

    def test_set_performance_warmup(self) -> None:
        """Test set_performance_warmup."""
        config = Config(self.predefined_config)

        config.set_performance_warmup(1234)

        self.assertEqual(1234, config.evaluation.performance.warmup)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_performance_warmup_to_negative_value(self) -> None:
        """Test set_performance_warmup."""
        config = Config(self.predefined_config)

        original_performance_warmup = config.evaluation.performance.warmup

        with self.assertRaises(ClientErrorException):
            config.set_performance_warmup(-1234)
        self.assertEqual(original_performance_warmup, config.evaluation.performance.warmup)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

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

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_performance_iterations_to_negative_value(self) -> None:
        """Test set_performance_iterations."""
        config = Config(self.predefined_config)

        original_performance_iterations = config.evaluation.performance.iteration

        with self.assertRaises(ClientErrorException):
            config.set_performance_iterations(-1234)
        self.assertEqual(original_performance_iterations, config.evaluation.performance.iteration)

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    def test_set_performance_iterations_on_empty_config(self) -> None:
        """Test set_performance_iterations."""
        config = Config()

        config.set_performance_iterations(1234)

        self.assertIsNone(config.evaluation)

    @patch("neural_compressor.ux.utils.workload.config.load_precisions_config")
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

    @patch("neural_compressor.ux.utils.workload.config.load_precisions_config")
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

    @patch("neural_compressor.ux.utils.workload.config.load_precisions_config")
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

    @patch("neural_compressor.ux.utils.workload.config.load_precisions_config")
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
        predefined_config = deepcopy(self.predefined_config)
        del predefined_config["pruning"]["approach"]["weight_compression"]["pruners"]

        read_yaml = yaml.dump(predefined_config, sort_keys=False)

        with patch(
            "neural_compressor.ux.utils.workload.config.open",
            mock_open(read_data=read_yaml),
        ) as mocked_open:
            config.load("path to yaml file")

            mocked_open.assert_called_once_with("path to yaml file")

        expected = Config(predefined_config)

        self.assertEqual(expected.serialize(), config.serialize())

        serialized_config = config.serialize()
        inc_config_schema.validate(serialized_config)

    @patch("os.makedirs")
    def test_dump(self, mocked_makedirs: MagicMock) -> None:
        """Test dump."""
        config = Config(self.predefined_config)

        yaml.add_representer(float, float_representer)  # type: ignore
        yaml.add_representer(Pruner, pruner_representer)  # type: ignore
        expected_yaml = yaml.dump(
            config.serialize(),
            indent=4,
            default_flow_style=None,
            sort_keys=False,
        )
        print("expected_yaml")
        print(expected_yaml)

        with patch("neural_compressor.ux.utils.workload.config.open", mock_open()) as mocked_open:
            config.dump("path to yaml file")

            mocked_open.assert_called_once_with("path to yaml file", "w")
            mocked_open().write.assert_called_once_with(expected_yaml)
            mocked_makedirs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
