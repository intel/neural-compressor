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
"""Pruning config test."""

import unittest

from neural_compressor.conf.config import Pruner
from neural_compressor.ux.utils.workload.pruning import Pruning, Train


class TestTrainConfig(unittest.TestCase):
    """Train config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Pruning config constructor."""
        super().__init__(*args, **kwargs)
        self.train_config = {
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
        }

    def test_train_constructor(self) -> None:
        """Test Train config constructor."""
        train = Train(self.train_config)

        self.assertIsNotNone(train)
        self.assertIsNotNone(train.optimizer)
        self.assertIsNotNone(train.optimizer.Adam)
        self.assertEqual(train.optimizer.Adam.learning_rate, 0.123)
        self.assertEqual(train.optimizer.Adam.beta_1, 0.99)
        self.assertEqual(train.optimizer.Adam.beta_2, 0.9999)
        self.assertEqual(train.optimizer.Adam.epsilon, 1e-06)
        self.assertEqual(train.optimizer.Adam.amsgrad, True)
        self.assertIsNotNone(train.criterion)
        self.assertIsNotNone(train.criterion.CrossEntropyLoss)
        self.assertEqual(train.criterion.CrossEntropyLoss.reduction, "auto")
        self.assertEqual(train.criterion.CrossEntropyLoss.from_logits, True)
        self.assertIsNotNone(train.criterion.SparseCategoricalCrossentropy)
        self.assertEqual(
            train.criterion.SparseCategoricalCrossentropy.reduction,
            "sum",
        )
        self.assertEqual(
            train.criterion.SparseCategoricalCrossentropy.from_logits,
            True,
        )
        self.assertIsNotNone(train.criterion.KnowledgeDistillationLoss)
        self.assertEqual(train.criterion.KnowledgeDistillationLoss.temperature, 1.0)
        self.assertListEqual(
            train.criterion.KnowledgeDistillationLoss.loss_types,
            ["CE"],
        )
        self.assertListEqual(
            train.criterion.KnowledgeDistillationLoss.loss_weights,
            [0.3, 0.7],
        )
        self.assertIsNotNone(train.dataloader)
        self.assertEqual(
            train.dataloader.dataset.name,
            "ImageRecord",
        )
        self.assertDictEqual(
            train.dataloader.dataset.params,
            {"root": "/path/to/pruning/dataset"},
        )
        transform_name, transform = list(
            train.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "ResizeCropImagenet")
        self.assertDictEqual(
            transform.parameters,
            {"height": 224, "width": 224, "mean_value": [123.68, 116.78, 103.94]},
        )
        self.assertIsNone(train.dataloader.filter)
        self.assertEqual(train.epoch, 5)
        self.assertEqual(train.start_epoch, 1)
        self.assertEqual(train.end_epoch, 2)
        self.assertEqual(train.iteration, 10)
        self.assertEqual(train.frequency, 3)
        self.assertEqual(train.execution_mode, "graph")
        self.assertIsNotNone(train.postprocess)
        self.assertEqual(train.hostfile, "/some/file")

    def test_train_constructor_defaults(self) -> None:
        """Test Train config constructor defaults."""
        train = Train()

        self.assertIsNotNone(train.optimizer)
        self.assertIsNone(train.optimizer.SGD)
        self.assertIsNone(train.optimizer.Adam)
        self.assertIsNone(train.optimizer.AdamW)

        self.assertIsNotNone(train.criterion)
        self.assertIsNone(train.criterion.CrossEntropyLoss)
        self.assertIsNone(train.criterion.SparseCategoricalCrossentropy)
        self.assertIsNone(train.criterion.KnowledgeDistillationLoss)

        self.assertIsNotNone(train.dataloader)

        self.assertIsNone(train.epoch)
        self.assertIsNone(train.start_epoch)
        self.assertIsNone(train.end_epoch)
        self.assertIsNone(train.iteration)
        self.assertIsNone(train.frequency)
        self.assertIsNone(train.execution_mode)
        self.assertIsNone(train.postprocess)
        self.assertIsNone(train.hostfile)

    def test_train_serializer_defaults(self) -> None:
        """Test Train config constructor defaults."""
        train = Train()
        result = train.serialize()

        self.assertEqual(type(result), dict)
        self.assertDictEqual(result, {})

    def test_train_serializer(self) -> None:
        """Test Train config constructor."""
        train = Train(self.train_config)
        result = train.serialize()

        self.assertDictEqual(
            result,
            self.train_config,
        )


class TestPruningConfig(unittest.TestCase):
    """Pruning config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Pruning config constructor."""
        super().__init__(*args, **kwargs)
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

        self.pruning_config = {
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
        }

    def test_pruning_constructor(self) -> None:
        """Test Pruning config constructor."""
        pruning = Pruning(self.pruning_config)

        self.assertIsNotNone(pruning.train)
        self.assertIsNotNone(pruning.train.optimizer)
        self.assertIsNotNone(pruning.train.optimizer.Adam)
        self.assertEqual(pruning.train.optimizer.Adam.learning_rate, 0.123)
        self.assertEqual(pruning.train.optimizer.Adam.beta_1, 0.99)
        self.assertEqual(pruning.train.optimizer.Adam.beta_2, 0.9999)
        self.assertEqual(pruning.train.optimizer.Adam.epsilon, 1e-06)
        self.assertEqual(pruning.train.optimizer.Adam.amsgrad, True)
        self.assertIsNotNone(pruning.train.criterion)
        self.assertIsNotNone(pruning.train.criterion.CrossEntropyLoss)
        self.assertEqual(pruning.train.criterion.CrossEntropyLoss.reduction, "auto")
        self.assertEqual(pruning.train.criterion.CrossEntropyLoss.from_logits, True)
        self.assertIsNotNone(pruning.train.criterion.SparseCategoricalCrossentropy)
        self.assertEqual(
            pruning.train.criterion.SparseCategoricalCrossentropy.reduction,
            "sum",
        )
        self.assertEqual(
            pruning.train.criterion.SparseCategoricalCrossentropy.from_logits,
            True,
        )
        self.assertIsNotNone(pruning.train.criterion.KnowledgeDistillationLoss)
        self.assertEqual(pruning.train.criterion.KnowledgeDistillationLoss.temperature, 1.0)
        self.assertListEqual(
            pruning.train.criterion.KnowledgeDistillationLoss.loss_types,
            ["CE"],
        )
        self.assertListEqual(
            pruning.train.criterion.KnowledgeDistillationLoss.loss_weights,
            [0.3, 0.7],
        )
        self.assertIsNotNone(pruning.train.dataloader)
        self.assertEqual(
            pruning.train.dataloader.dataset.name,
            "ImageRecord",
        )
        self.assertDictEqual(
            pruning.train.dataloader.dataset.params,
            {"root": "/path/to/pruning/dataset"},
        )
        transform_name, transform = list(
            pruning.train.dataloader.transform.items(),
        )[0]
        self.assertEqual(transform_name, "ResizeCropImagenet")
        self.assertDictEqual(
            transform.parameters,
            {"height": 224, "width": 224, "mean_value": [123.68, 116.78, 103.94]},
        )
        self.assertIsNone(pruning.train.dataloader.filter)
        self.assertEqual(pruning.train.epoch, 5)
        self.assertEqual(pruning.train.start_epoch, 1)
        self.assertEqual(pruning.train.end_epoch, 2)
        self.assertEqual(pruning.train.iteration, 10)
        self.assertEqual(pruning.train.frequency, 3)
        self.assertEqual(pruning.train.execution_mode, "graph")
        self.assertIsNotNone(pruning.train.postprocess)
        self.assertEqual(pruning.train.hostfile, "/some/file")

        self.assertIsNotNone(pruning.approach)
        self.assertIsNotNone(pruning.approach.weight_compression)
        self.assertEqual(pruning.approach.weight_compression.initial_sparsity, 0.042)
        self.assertEqual(pruning.approach.weight_compression.target_sparsity, 0.1337)
        self.assertEqual(pruning.approach.weight_compression.start_epoch, 13)
        self.assertEqual(pruning.approach.weight_compression.end_epoch, 888)
        self.assertEqual(len(pruning.approach.weight_compression.pruners), 1)

    def test_pruning_constructor_defaults(self) -> None:
        """Test Pruning config constructor defaults."""
        pruning = Pruning()

        self.assertIsNone(pruning.train)
        self.assertIsNone(pruning.approach)

    def test_pruning_serializer_defaults(self) -> None:
        """Test Pruning config constructor defaults."""
        pruning = Pruning()
        result = pruning.serialize()

        self.assertDictEqual(result, {})

    def test_pruning_serializer(self) -> None:
        """Test Pruning config constructor."""
        pruning = Pruning(self.pruning_config)
        result = pruning.serialize()

        expected = {
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
                        {
                            "initial_sparsity": 0.0,
                            "target_sparsity": 0.97,
                            "start_epoch": 0,
                            "end_epoch": 2,
                            "prune_type": "basic_magnitude",
                            "update_frequency": 0.1,
                            "names": ["layer1.0.conv1.weight"],
                            "pattern": "tile_pattern_1x1",
                            "method": "per_tensor",
                        },
                    ],
                },
            },
        }

        self.assertDictEqual(
            expected,
            result,
        )


if __name__ == "__main__":
    unittest.main()
