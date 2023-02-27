# -*- coding: utf-8 -*-
# Copyright (c) 2022 Intel Corporation
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
"""Pruning configuration parser test."""
import unittest
from unittest.mock import patch

from neural_compressor.ux.components.configuration_wizard.pruning_config_parser import (
    PruningConfigParser,
)


@patch("sys.argv", ["inc_bench.py", "-p5000"])
class TestPruningConfigParser(unittest.TestCase):
    """Main test class for parser."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Parser tests constructor."""
        super().__init__(*args, **kwargs)

    def test_generate_pruning_config_tree(self) -> None:
        """Test generating pruning config tree."""
        data = {
            "train": {
                "optimizer": {
                    "SGD": {
                        "learning_rate": 0.001,
                        "momentum": 1,
                        "nesterov": True,
                        "weight_decay": "asd",
                    },
                    "AdamW": {
                        "weight_decay": None,
                        "learning_rate": "0.001",
                        "beta_1": 0.9,
                        "beta_2": 0.999,
                        "epsilon": 1.00e-07,
                        "amsgrad": False,
                    },
                    "Adam": {
                        "learning_rate": "0.001",
                        "beta_1": "0.9",
                        "beta_2": "0.999",
                        "epsilon": "1.00E-07",
                        "amsgrad": False,
                    },
                },
                "criterion": {
                    "CrossEntropyLoss": {
                        "reduction": "mean",
                        "from_logits": False,
                    },
                    "SparseCategoricalCrossentropy": {
                        "reduction": "mean",
                        "from_logits": False,
                    },
                    "KnowledgeDistillationLoss": {
                        "temperature": None,
                        "loss_types": "CE",
                        "loss_weights": None,
                    },
                    "IntermediateLayersKnowledgeDistillationLoss": {
                        "layer_mappings": None,
                        "loss_types": "MSE",
                        "loss_weights": None,
                        "add_origin_loss": None,
                    },
                    "SelfKnowledgeDistillationLoss": {
                        "layer_mappings": None,
                        "loss_types": "L2",
                        "loss_weights": None,
                        "add_origin_loss": True,
                        "temperature": None,
                    },
                },
                "epoch": 1,
                "start_epoch": 0,
                "end_epoch": 4,
                "iteration": 1,
                "frequency": None,
                "execution_mode": "eager",
                "postprocess": {
                    "transform": {
                        "LabelShift": None,
                        "Collect": {
                            "length": None,
                        },
                        "SquadV1": {
                            "label_file": None,
                            "vocab_file": None,
                            "do_lower_case": True,
                            "max_seq_length": "384",
                        },
                        "SquadV1ModelZoo": {
                            "label_file": None,
                            "vocab_file": None,
                            "do_lower_case": True,
                            "max_seq_length": "384",
                        },
                    },
                },
                "hostfile": None,
            },
            "approach": {
                "weight_compression": {
                    "initial_sparsity": "0",
                    "target_sparsity": "0.97",
                    "max_sparsity_ratio_per_layer": "0.98",
                    "prune_type": "basic_magnitude",
                    "start_epoch": "0",
                    "end_epoch": "4",
                    "start_step": "0",
                    "end_step": "0",
                    "update_frequency": "1",
                    "update_frequency_on_step": "1",
                    "excluded_names": "[]",
                    "prune_domain": "global",
                    "names": "[]",
                    "extra_excluded_names": "None",
                    "prune_layer_type": "None",
                    "sparsity_decay_type": "exp",
                    "pattern": "tile_pattern_1x1",
                    "pruners": [
                        {
                            "!Pruner": {
                                "start_epoch": "None",
                                "end_epoch": "None",
                                "initial_sparsity": "None",
                                "target_sparsity": "None",
                                "update_frequency": "1",
                                "method": "per_tensor",
                                "prune_type": "basic_magnitude",
                                "start_step": "None",
                                "end_step": "None",
                                "update_frequency_on_step": "None",
                                "prune_domain": "global",
                                "sparsity_decay_type": "None",
                                "pattern": "tile_pattern_1x1",
                                "names": "None",
                                "extra_excluded_names": "None",
                                "parameters": "None",
                            },
                        },
                    ],
                },
                "weight_compression_pytorch": {
                    "initial_sparsity": "0",
                    "target_sparsity": "0.97",
                    "max_sparsity_ratio_per_layer": "0.98",
                    "prune_type": "basic_magnitude",
                    "start_epoch": "0",
                    "end_epoch": "4",
                    "start_step": "0",
                    "end_step": "0",
                    "update_frequency": "1",
                    "update_frequency_on_step": "1",
                    "excluded_names": "[]",
                    "prune_domain": "global",
                    "names": "[]",
                    "extra_excluded_names": "None",
                    "prune_layer_type": "None",
                    "sparsity_decay_type": "exp",
                    "pattern": "tile_pattern_1x1",
                    "pruners": [
                        {
                            "!Pruner": {
                                "start_epoch": "None",
                                "end_epoch": "None",
                                "initial_sparsity": "None",
                                "target_sparsity": "None",
                                "update_frequency": "1",
                                "method": "per_tensor",
                                "prune_type": "basic_magnitude",
                                "start_step": "None",
                                "end_step": "None",
                                "update_frequency_on_step": "None",
                                "prune_domain": "global",
                                "sparsity_decay_type": "None",
                                "pattern": "tile_pattern_1x1",
                                "names": "None",
                                "extra_excluded_names": "None",
                                "parameters": "None",
                            },
                        },
                    ],
                },
            },
        }
        expected_pruning_details = [
            {
                "name": "train",
                "children": [
                    {
                        "name": "optimizer",
                        "children": [
                            {
                                "name": "SGD",
                                "children": [
                                    {"name": "learning_rate", "value": 0.001},
                                    {"name": "momentum", "value": 1},
                                    {"name": "nesterov", "value": True},
                                    {"name": "weight_decay", "value": "asd"},
                                ],
                            },
                            {
                                "name": "AdamW",
                                "children": [
                                    {"name": "weight_decay", "value": None},
                                    {"name": "learning_rate", "value": "0.001"},
                                    {"name": "beta_1", "value": 0.9},
                                    {"name": "beta_2", "value": 0.999},
                                    {"name": "epsilon", "value": 1e-7},
                                    {"name": "amsgrad", "value": False},
                                ],
                            },
                            {
                                "name": "Adam",
                                "children": [
                                    {"name": "learning_rate", "value": "0.001"},
                                    {"name": "beta_1", "value": "0.9"},
                                    {"name": "beta_2", "value": "0.999"},
                                    {"name": "epsilon", "value": "1.00E-07"},
                                    {"name": "amsgrad", "value": False},
                                ],
                            },
                        ],
                    },
                    {
                        "name": "criterion",
                        "children": [
                            {
                                "name": "CrossEntropyLoss",
                                "children": [
                                    {"name": "reduction", "value": "mean"},
                                    {"name": "from_logits", "value": False},
                                ],
                            },
                            {
                                "name": "SparseCategoricalCrossentropy",
                                "children": [
                                    {"name": "reduction", "value": "mean"},
                                    {"name": "from_logits", "value": False},
                                ],
                            },
                            {
                                "name": "KnowledgeDistillationLoss",
                                "children": [
                                    {"name": "temperature", "value": None},
                                    {"name": "loss_types", "value": "CE"},
                                    {"name": "loss_weights", "value": None},
                                ],
                            },
                            {
                                "name": "IntermediateLayersKnowledgeDistillationLoss",
                                "children": [
                                    {"name": "layer_mappings", "value": None},
                                    {"name": "loss_types", "value": "MSE"},
                                    {"name": "loss_weights", "value": None},
                                    {"name": "add_origin_loss", "value": None},
                                ],
                            },
                            {
                                "name": "SelfKnowledgeDistillationLoss",
                                "children": [
                                    {"name": "layer_mappings", "value": None},
                                    {"name": "loss_types", "value": "L2"},
                                    {"name": "loss_weights", "value": None},
                                    {"name": "add_origin_loss", "value": True},
                                    {"name": "temperature", "value": None},
                                ],
                            },
                        ],
                    },
                    {"name": "epoch", "value": 1},
                    {"name": "start_epoch", "value": 0},
                    {"name": "end_epoch", "value": 4},
                    {"name": "iteration", "value": 1},
                    {"name": "frequency", "value": None},
                    {"name": "execution_mode", "value": "eager"},
                    {
                        "name": "postprocess",
                        "children": [
                            {
                                "name": "transform",
                                "children": [
                                    {"name": "LabelShift", "value": None},
                                    {
                                        "name": "Collect",
                                        "children": [{"name": "length", "value": None}],
                                    },
                                    {
                                        "name": "SquadV1",
                                        "children": [
                                            {"name": "label_file", "value": None},
                                            {"name": "vocab_file", "value": None},
                                            {"name": "do_lower_case", "value": True},
                                            {"name": "max_seq_length", "value": "384"},
                                        ],
                                    },
                                    {
                                        "name": "SquadV1ModelZoo",
                                        "children": [
                                            {"name": "label_file", "value": None},
                                            {"name": "vocab_file", "value": None},
                                            {"name": "do_lower_case", "value": True},
                                            {"name": "max_seq_length", "value": "384"},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                    {"name": "hostfile", "value": None},
                ],
            },
            {
                "name": "approach",
                "children": [
                    {
                        "name": "weight_compression",
                        "children": [
                            {"name": "initial_sparsity", "value": "0"},
                            {"name": "target_sparsity", "value": "0.97"},
                            {"name": "max_sparsity_ratio_per_layer", "value": "0.98"},
                            {"name": "prune_type", "value": "basic_magnitude"},
                            {"name": "start_epoch", "value": "0"},
                            {"name": "end_epoch", "value": "4"},
                            {"name": "start_step", "value": "0"},
                            {"name": "end_step", "value": "0"},
                            {"name": "update_frequency", "value": "1"},
                            {"name": "update_frequency_on_step", "value": "1"},
                            {"name": "excluded_names", "value": "[]"},
                            {"name": "prune_domain", "value": "global"},
                            {"name": "names", "value": "[]"},
                            {"name": "extra_excluded_names", "value": "None"},
                            {"name": "prune_layer_type", "value": "None"},
                            {"name": "sparsity_decay_type", "value": "exp"},
                            {"name": "pattern", "value": "tile_pattern_1x1"},
                            {
                                "name": "pruners",
                                "children": [
                                    {
                                        "name": "!Pruner",
                                        "children": [
                                            {"name": "start_epoch", "value": "None"},
                                            {"name": "end_epoch", "value": "None"},
                                            {"name": "initial_sparsity", "value": "None"},
                                            {"name": "target_sparsity", "value": "None"},
                                            {"name": "update_frequency", "value": "1"},
                                            {"name": "method", "value": "per_tensor"},
                                            {"name": "prune_type", "value": "basic_magnitude"},
                                            {"name": "start_step", "value": "None"},
                                            {"name": "end_step", "value": "None"},
                                            {"name": "update_frequency_on_step", "value": "None"},
                                            {"name": "prune_domain", "value": "global"},
                                            {"name": "sparsity_decay_type", "value": "None"},
                                            {"name": "pattern", "value": "tile_pattern_1x1"},
                                            {"name": "names", "value": "None"},
                                            {"name": "extra_excluded_names", "value": "None"},
                                            {"name": "parameters", "value": "None"},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        "name": "weight_compression_pytorch",
                        "children": [
                            {"name": "initial_sparsity", "value": "0"},
                            {"name": "target_sparsity", "value": "0.97"},
                            {"name": "max_sparsity_ratio_per_layer", "value": "0.98"},
                            {"name": "prune_type", "value": "basic_magnitude"},
                            {"name": "start_epoch", "value": "0"},
                            {"name": "end_epoch", "value": "4"},
                            {"name": "start_step", "value": "0"},
                            {"name": "end_step", "value": "0"},
                            {"name": "update_frequency", "value": "1"},
                            {"name": "update_frequency_on_step", "value": "1"},
                            {"name": "excluded_names", "value": "[]"},
                            {"name": "prune_domain", "value": "global"},
                            {"name": "names", "value": "[]"},
                            {"name": "extra_excluded_names", "value": "None"},
                            {"name": "prune_layer_type", "value": "None"},
                            {"name": "sparsity_decay_type", "value": "exp"},
                            {"name": "pattern", "value": "tile_pattern_1x1"},
                            {
                                "name": "pruners",
                                "children": [
                                    {
                                        "name": "!Pruner",
                                        "children": [
                                            {"name": "start_epoch", "value": "None"},
                                            {"name": "end_epoch", "value": "None"},
                                            {"name": "initial_sparsity", "value": "None"},
                                            {"name": "target_sparsity", "value": "None"},
                                            {"name": "update_frequency", "value": "1"},
                                            {"name": "method", "value": "per_tensor"},
                                            {"name": "prune_type", "value": "basic_magnitude"},
                                            {"name": "start_step", "value": "None"},
                                            {"name": "end_step", "value": "None"},
                                            {"name": "update_frequency_on_step", "value": "None"},
                                            {"name": "prune_domain", "value": "global"},
                                            {"name": "sparsity_decay_type", "value": "None"},
                                            {"name": "pattern", "value": "tile_pattern_1x1"},
                                            {"name": "names", "value": "None"},
                                            {"name": "extra_excluded_names", "value": "None"},
                                            {"name": "parameters", "value": "None"},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

        parser = PruningConfigParser()
        actual_pruning_details = parser.generate_tree(data)

        self.assertListEqual(expected_pruning_details, actual_pruning_details)

    def test_generate_pruning_config_tree_without_train(self) -> None:
        """Test generating pruning config tree."""
        data = {
            "train": None,
            "approach": {
                "weight_compression": {
                    "initial_sparsity": "0",
                    "target_sparsity": "0.97",
                    "max_sparsity_ratio_per_layer": "0.98",
                    "prune_type": "basic_magnitude",
                    "start_epoch": "0",
                    "end_epoch": "4",
                    "start_step": "0",
                    "end_step": "0",
                    "update_frequency": "1",
                    "update_frequency_on_step": "1",
                    "excluded_names": "[]",
                    "prune_domain": "global",
                    "names": "[]",
                    "extra_excluded_names": "None",
                    "prune_layer_type": "None",
                    "sparsity_decay_type": "exp",
                    "pattern": "tile_pattern_1x1",
                    "pruners": [
                        {
                            "!Pruner": {
                                "start_epoch": "None",
                                "end_epoch": "None",
                                "initial_sparsity": "None",
                                "target_sparsity": "None",
                                "update_frequency": "1",
                                "method": "per_tensor",
                                "prune_type": "basic_magnitude",
                                "start_step": "None",
                                "end_step": "None",
                                "update_frequency_on_step": "None",
                                "prune_domain": "global",
                                "sparsity_decay_type": "None",
                                "pattern": "tile_pattern_1x1",
                                "names": "None",
                                "extra_excluded_names": "None",
                                "parameters": "None",
                            },
                        },
                    ],
                },
                "weight_compression_pytorch": {
                    "initial_sparsity": "0",
                    "target_sparsity": "0.97",
                    "max_sparsity_ratio_per_layer": "0.98",
                    "prune_type": "basic_magnitude",
                    "start_epoch": "0",
                    "end_epoch": "4",
                    "start_step": "0",
                    "end_step": "0",
                    "update_frequency": "1",
                    "update_frequency_on_step": "1",
                    "excluded_names": "[]",
                    "prune_domain": "global",
                    "names": "[]",
                    "extra_excluded_names": "None",
                    "prune_layer_type": "None",
                    "sparsity_decay_type": "exp",
                    "pattern": "tile_pattern_1x1",
                    "pruners": [
                        {
                            "!Pruner": {
                                "start_epoch": "None",
                                "end_epoch": "None",
                                "initial_sparsity": "None",
                                "target_sparsity": "None",
                                "update_frequency": "1",
                                "method": "per_tensor",
                                "prune_type": "basic_magnitude",
                                "start_step": "None",
                                "end_step": "None",
                                "update_frequency_on_step": "None",
                                "prune_domain": "global",
                                "sparsity_decay_type": "None",
                                "pattern": "tile_pattern_1x1",
                                "names": "None",
                                "extra_excluded_names": "None",
                                "parameters": "None",
                            },
                        },
                    ],
                },
            },
        }
        expected_pruning_details = [
            {
                "name": "approach",
                "children": [
                    {
                        "name": "weight_compression",
                        "children": [
                            {"name": "initial_sparsity", "value": "0"},
                            {"name": "target_sparsity", "value": "0.97"},
                            {"name": "max_sparsity_ratio_per_layer", "value": "0.98"},
                            {"name": "prune_type", "value": "basic_magnitude"},
                            {"name": "start_epoch", "value": "0"},
                            {"name": "end_epoch", "value": "4"},
                            {"name": "start_step", "value": "0"},
                            {"name": "end_step", "value": "0"},
                            {"name": "update_frequency", "value": "1"},
                            {"name": "update_frequency_on_step", "value": "1"},
                            {"name": "excluded_names", "value": "[]"},
                            {"name": "prune_domain", "value": "global"},
                            {"name": "names", "value": "[]"},
                            {"name": "extra_excluded_names", "value": "None"},
                            {"name": "prune_layer_type", "value": "None"},
                            {"name": "sparsity_decay_type", "value": "exp"},
                            {"name": "pattern", "value": "tile_pattern_1x1"},
                            {
                                "name": "pruners",
                                "children": [
                                    {
                                        "name": "!Pruner",
                                        "children": [
                                            {"name": "start_epoch", "value": "None"},
                                            {"name": "end_epoch", "value": "None"},
                                            {"name": "initial_sparsity", "value": "None"},
                                            {"name": "target_sparsity", "value": "None"},
                                            {"name": "update_frequency", "value": "1"},
                                            {"name": "method", "value": "per_tensor"},
                                            {"name": "prune_type", "value": "basic_magnitude"},
                                            {"name": "start_step", "value": "None"},
                                            {"name": "end_step", "value": "None"},
                                            {"name": "update_frequency_on_step", "value": "None"},
                                            {"name": "prune_domain", "value": "global"},
                                            {"name": "sparsity_decay_type", "value": "None"},
                                            {"name": "pattern", "value": "tile_pattern_1x1"},
                                            {"name": "names", "value": "None"},
                                            {"name": "extra_excluded_names", "value": "None"},
                                            {"name": "parameters", "value": "None"},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        "name": "weight_compression_pytorch",
                        "children": [
                            {"name": "initial_sparsity", "value": "0"},
                            {"name": "target_sparsity", "value": "0.97"},
                            {"name": "max_sparsity_ratio_per_layer", "value": "0.98"},
                            {"name": "prune_type", "value": "basic_magnitude"},
                            {"name": "start_epoch", "value": "0"},
                            {"name": "end_epoch", "value": "4"},
                            {"name": "start_step", "value": "0"},
                            {"name": "end_step", "value": "0"},
                            {"name": "update_frequency", "value": "1"},
                            {"name": "update_frequency_on_step", "value": "1"},
                            {"name": "excluded_names", "value": "[]"},
                            {"name": "prune_domain", "value": "global"},
                            {"name": "names", "value": "[]"},
                            {"name": "extra_excluded_names", "value": "None"},
                            {"name": "prune_layer_type", "value": "None"},
                            {"name": "sparsity_decay_type", "value": "exp"},
                            {"name": "pattern", "value": "tile_pattern_1x1"},
                            {
                                "name": "pruners",
                                "children": [
                                    {
                                        "name": "!Pruner",
                                        "children": [
                                            {"name": "start_epoch", "value": "None"},
                                            {"name": "end_epoch", "value": "None"},
                                            {"name": "initial_sparsity", "value": "None"},
                                            {"name": "target_sparsity", "value": "None"},
                                            {"name": "update_frequency", "value": "1"},
                                            {"name": "method", "value": "per_tensor"},
                                            {"name": "prune_type", "value": "basic_magnitude"},
                                            {"name": "start_step", "value": "None"},
                                            {"name": "end_step", "value": "None"},
                                            {"name": "update_frequency_on_step", "value": "None"},
                                            {"name": "prune_domain", "value": "global"},
                                            {"name": "sparsity_decay_type", "value": "None"},
                                            {"name": "pattern", "value": "tile_pattern_1x1"},
                                            {"name": "names", "value": "None"},
                                            {"name": "extra_excluded_names", "value": "None"},
                                            {"name": "parameters", "value": "None"},
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ]

        parser = PruningConfigParser()
        actual_pruning_details = parser.generate_tree(data)

        self.assertListEqual(expected_pruning_details, actual_pruning_details)
