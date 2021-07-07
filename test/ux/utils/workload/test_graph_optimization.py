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
"""Graph Optimization config test."""

import unittest

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.workload.graph_optimization import GraphOptimization


class GraphOptimizationConfig(unittest.TestCase):
    """Graph Optimization config tests."""

    def __init__(self, *args: str, **kwargs: str) -> None:
        """Graph Optimization config test constructor."""
        super().__init__(*args, **kwargs)

    def test_graph_optimization_constructor(self) -> None:
        """Test Graph Optimization config constructor."""
        data = {
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
        }
        graph_optimization = GraphOptimization(data)

        self.assertEqual(graph_optimization.precisions, "bf16,fp32")
        self.assertIsNotNone(graph_optimization.op_wise)
        self.assertDictEqual(
            graph_optimization.op_wise,
            {
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
        )

    def test_graph_optimization_constructor_defaults(self) -> None:
        """Test Graph Optimization config constructor defaults."""
        graph_optimization = GraphOptimization()

        self.assertIsNone(graph_optimization.precisions)
        self.assertIsNone(graph_optimization.op_wise)

    def test_set_precisions_string(self) -> None:
        """Test setting precisions in Graph Optimization config."""
        graph_optimization = GraphOptimization()
        graph_optimization.set_precisions(" bf16, fp32 ")
        self.assertEqual(graph_optimization.precisions, "bf16,fp32")

    def test_set_precisions_list(self) -> None:
        """Test setting precisions in Graph Optimization config."""
        graph_optimization = GraphOptimization()
        graph_optimization.set_precisions(["bf16", "fp32 ", " int8"])
        self.assertEqual(graph_optimization.precisions, "bf16,fp32,int8")

    def test_set_precisions_error(self) -> None:
        """Test overwriting precisions in Graph Optimization config."""
        graph_optimization = GraphOptimization()
        with self.assertRaises(ClientErrorException):
            graph_optimization.set_precisions(1)

    def test_graph_optimization_serializer(self) -> None:
        """Test Graph Optimization config serializer."""
        data = {
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
        }
        graph_optimization = GraphOptimization(data)
        result = graph_optimization.serialize()

        self.assertDictEqual(
            result,
            {
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
        )

    def test_graph_optimization_serializer_defaults(self) -> None:
        """Test Graph Optimization config serializer."""
        graph_optimization = GraphOptimization()
        result = graph_optimization.serialize()

        self.assertDictEqual(result, {})


if __name__ == "__main__":
    unittest.main()
