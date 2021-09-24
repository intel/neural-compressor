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
"""Test frozen pb Model."""

import unittest
from typing import List
from unittest.mock import MagicMock, patch

from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.graph.node import Node
from neural_compressor.ux.components.model.domain import Domain
from neural_compressor.ux.components.model.tensorflow.frozen_pb import FrozenPbModel


class TestFrozenPbModel(unittest.TestCase):
    """Test FrozenPbModel class."""

    def setUp(self) -> None:
        """Prepare environment."""
        super().setUp()

        get_model_type_patcher = patch(
            "neural_compressor.ux.components.model.model_type_getter.nc_get_model_type",
        )
        self.addCleanup(get_model_type_patcher.stop)
        get_model_type_mock = get_model_type_patcher.start()
        get_model_type_mock.side_effect = self._get_model_type

        nc_tensorflow_model_patcher = patch(
            "neural_compressor.ux.components.model.tensorflow.model.NCModel",
        )
        self.addCleanup(nc_tensorflow_model_patcher.stop)
        nc_model_instance_mock = nc_tensorflow_model_patcher.start()
        nc_model_instance_mock.return_value.input_node_names = [
            "first input node",
            "second input node",
        ]
        nc_model_instance_mock.return_value.output_node_names = [
            "first output node",
            "second output node",
        ]

    def _get_model_type(self, path: str) -> str:
        """Return model type for well known paths."""
        if "/path/to/frozen_pb.pb" == path:
            return "frozen_pb"
        raise ValueError()

    def test_get_framework_name(self) -> None:
        """Test getting correct framework name."""
        self.assertEqual("tensorflow", FrozenPbModel.get_framework_name())

    def test_supports_correct_path(self) -> None:
        """Test getting correct framework name."""
        self.assertTrue(FrozenPbModel.supports_path("/path/to/frozen_pb.pb"))

    def test_supports_incorrect_path(self) -> None:
        """Test getting correct framework name."""
        self.assertFalse(FrozenPbModel.supports_path("/path/to/model.txt"))

    @patch("neural_compressor.ux.components.model.tensorflow.model.check_module")
    def test_guard_requirements_installed(self, mocked_check_module: MagicMock) -> None:
        """Test guard_requirements_installed."""
        model = FrozenPbModel("/path/to/frozen_pb.pb")

        model.guard_requirements_installed()

        mocked_check_module.assert_called_once_with("tensorflow")

    def test_get_input_nodes(self) -> None:
        """Test getting input nodes."""
        model = FrozenPbModel("/path/to/frozen_pb.pb")
        self.assertEqual(["first input node", "second input node"], model.get_input_nodes())

    def test_get_output_nodes(self) -> None:
        """Test getting output nodes."""
        model = FrozenPbModel("/path/to/frozen_pb.pb")
        self.assertEqual(
            ["first output node", "second output node", "custom"],
            model.get_output_nodes(),
        )

    def test_get_input_and_output_nodes(self) -> None:
        """Test getting input nodes."""
        model = FrozenPbModel("/path/to/frozen_pb.pb")
        self.assertEqual(["first input node", "second input node"], model.get_input_nodes())
        self.assertEqual(
            ["first output node", "second output node", "custom"],
            model.get_output_nodes(),
        )

    @patch("neural_compressor.ux.components.model.tensorflow.model.TensorflowReader", autospec=True)
    def test_get_model_graph(self, mocked_tensorflow_graph_reader: MagicMock) -> None:
        """Test getting Graph of a model."""
        expected = Graph()

        mocked_tensorflow_graph_reader.return_value.read.return_value = expected

        model = FrozenPbModel("/path/to/frozen_pb.pb")

        self.assertEqual(expected, model.get_model_graph())

        mocked_tensorflow_graph_reader.assert_called_once_with(model)

    def test_domain_object_detection_domain(self) -> None:
        """Test getting domain of a model."""
        self.assert_model_domain_matches_expected(
            node_names=["boxes", "scores", "classes"],
            expected_domain="object_detection",
            expected_domain_flavour="",
        )

    def test_domain_object_detection_domain_ssd(self) -> None:
        """Test getting domain of a model."""
        self.assert_model_domain_matches_expected(
            node_names=["bboxes", "scores", "classes", "ssd"],
            expected_domain="object_detection",
            expected_domain_flavour="ssd",
        )

    def test_domain_object_detection_domain_yolo(self) -> None:
        """Test getting domain of a model."""
        self.assert_model_domain_matches_expected(
            node_names=["boxes", "yolo"],
            expected_domain="object_detection",
            expected_domain_flavour="yolo",
        )

    def test_domain_image_recognition_resnet(self) -> None:
        """Test getting domain of a model."""
        self.assert_model_domain_matches_expected(
            node_names=["resnet_model/Pad"],
            expected_domain="image_recognition",
            expected_domain_flavour="",
        )

    def test_domain_unknown(self) -> None:
        """Test getting domain of a model."""
        self.assert_model_domain_matches_expected(
            node_names=["foo", "bar", "baz", "ssd"],
            expected_domain="",
            expected_domain_flavour="",
        )

    @patch("neural_compressor.ux.components.model.tensorflow.model.TensorflowReader", autospec=True)
    def test_domain_graph_reader_exception(
        self,
        mocked_tensorflow_graph_reader: MagicMock,
    ) -> None:
        """Test getting domain of a model."""
        mocked_tensorflow_graph_reader.return_value.read.side_effect = Exception()

        model = FrozenPbModel("/path/to/frozen_pb.pb")

        expected = Domain(domain="", domain_flavour="")

        self.assertEqual(expected, model.domain)
        mocked_tensorflow_graph_reader.assert_called_once_with(model)

    @patch("neural_compressor.ux.components.model.tensorflow.model.TensorflowReader", autospec=True)
    def assert_model_domain_matches_expected(
        self,
        mocked_tensorflow_graph_reader: MagicMock,
        node_names: List[str],
        expected_domain: str,
        expected_domain_flavour: str,
    ) -> None:
        """Test getting domain of a model."""

        def graph_with_nodes() -> Graph:
            """Create a graph with named nodes."""
            graph = Graph()
            for name in node_names:
                graph.add_node(Node(id=name, label=name))
            return graph

        mocked_tensorflow_graph_reader.return_value.read.return_value = graph_with_nodes()

        model = FrozenPbModel("/path/to/frozen_pb.pb")

        expected = Domain(domain=expected_domain, domain_flavour=expected_domain_flavour)

        self.assertEqual(expected, model.domain)
        mocked_tensorflow_graph_reader.assert_called_once_with(model)


if __name__ == "__main__":
    unittest.main()
