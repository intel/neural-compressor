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
"""Onnxrt model class."""
import os
import re
import sys
from typing import Any, List, Optional

from neural_compressor.experimental.common.model import Model as NCModel
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.utils.logger import Logger
from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.graph.reader.onnxrt_reader import OnnxrtReader
from neural_compressor.ux.components.model.domain import Domain
from neural_compressor.ux.components.model.model import Model
from neural_compressor.ux.components.model.shape import Shape
from neural_compressor.ux.utils.consts import DomainFlavours, Domains, Frameworks
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import check_module, get_file_extension


class OnnxrtModel(Model):
    """Onnxrt Model class."""

    def __init__(self, path: str) -> None:
        """Initialize object."""
        super().__init__(path)
        self._nc_model_instance: Optional[ONNXModel] = None

    @property
    def domain(self) -> Domain:
        """Get model domain."""
        try:
            input_node_names = {
                node.name for node in self.nc_model_instance.graph().input  # pylint: disable=E1101
            }
            node_names = {
                node.name for node in self.nc_model_instance.nodes()  # pylint: disable=E1101
            }
            boundary_nodes = [
                node.name for node in self.nc_model_instance.graph().input  # pylint: disable=E1101
            ]
            boundary_nodes.extend(
                [
                    node.name
                    for node in self.nc_model_instance.graph().output  # pylint: disable=E1101
                ],
            )
            op_names = {
                node.op_type for node in self.nc_model_instance.nodes()  # pylint: disable=E1101
            }

        except Exception:
            return Domain()

        nlp_domain = self._check_nlp_domain(input_node_names)
        if nlp_domain is not None:
            return nlp_domain

        object_detection_domain = self._check_object_detection_domain(
            node_names,
            boundary_nodes,
            op_names,
        )
        if object_detection_domain is not None:
            return object_detection_domain

        image_recognition_domain = self._check_image_recognition_domain(op_names)
        if image_recognition_domain is not None:
            return image_recognition_domain

        return Domain()

    @staticmethod
    def _has_any_name_parts(nodes: set, name_parts: List[str]) -> bool:
        """Check if there is at least one node for name_parts."""
        matching_names = []
        for node in nodes:
            for partial_name in name_parts:
                search = re.match(partial_name, node)
                if search:
                    matching_names.append(node)
        return bool(matching_names)

    def _check_nlp_domain(self, input_node_names: set) -> Optional[Domain]:
        """Check if model fits into NLP domain."""
        if self._has_any_name_parts(
            input_node_names,
            ["input_ids", "input_mask", "segment_ids"],
        ):
            return Domain(
                domain=Domains.NLP.value,
                domain_flavour=DomainFlavours.NONE.value,
            )
        return None

    def _check_object_detection_domain(
        self,
        node_names: set,
        boundary_nodes: list,
        op_names: set,
    ) -> Optional[Domain]:
        """Check if model fits into Object Detection domain."""
        if (
            self._has_any_name_parts(
                node_names,
                ["NonMaxSuppression", "box", "score", "class", "detection"],
            )
            or self._has_any_name_parts(op_names, ["NonMaxSuppression"])
            or self._has_any_name_parts(
                set(boundary_nodes),
                ["NonMaxSuppression", "box", "score", "class", "detection"],
            )
        ):
            return Domain(
                domain=Domains.OBJECT_DETECTION.value,
                domain_flavour=DomainFlavours.NONE.value,
            )
        return None

    def _check_image_recognition_domain(self, op_names: set) -> Optional[Domain]:
        """Check if model fits into Image Recognition domain."""
        if self._has_any_name_parts(op_names, ["Conv", "Relu/Clip"]):
            return Domain(
                domain=Domains.IMAGE_RECOGNITION.value,
                domain_flavour=DomainFlavours.NONE.value,
            )
        return None

    @property
    def input_shape(self) -> Shape:
        """Try to detect data shape."""
        from google.protobuf.json_format import MessageToDict

        shape = None
        trusted = False

        if self.domain.domain == Domains.IMAGE_RECOGNITION.value:
            shape_list = []

            for input_node in self.filtered_input_nodes:
                node_dict = MessageToDict(input_node)
                dimensions = (
                    node_dict.get("type", {}).get("tensorType", {}).get("shape", {}).get("dim", [])
                )
                input_shape = []
                for dim in dimensions:
                    if dim.get("dimValue", None) is not None:
                        input_shape.append(dim["dimValue"])
                shape_list.append(",".join(input_shape))

            if len(shape_list) == 1:
                shape = shape_list[0]
                shape = remove_number_of_samples_from_shape(shape)
                trusted = True
            if len(shape_list) > 1:
                adjusted_shapes_list = []
                for sub_shape in shape_list:
                    adjusted_shapes_list.append(
                        remove_number_of_samples_from_shape(sub_shape),
                    )
                shape = ",".join(
                    ["[" + sub_shape + "]" for sub_shape in adjusted_shapes_list],
                )
                trusted = True

        return Shape(shape=shape, trusted=trusted)

    @property
    def shape_elements_order(self) -> List[str]:
        """Get order of input shape channels."""
        return ["channels", "height", "width"]

    @staticmethod
    def get_framework_name() -> str:
        """Get the name of framework."""
        return Frameworks.ONNX.value

    def get_model_graph(self) -> Graph:
        """Get model Graph."""
        graph_reader = OnnxrtReader(self)
        return graph_reader.read()

    @property
    def nc_model_instance(self) -> ONNXModel:
        """Get Neural Compressor Model instance."""
        self._ensure_nc_model_instance()
        return self._nc_model_instance

    def _ensure_nc_model_instance(self) -> None:
        """Create Neural Compressor Model instance if needed."""
        if self._nc_model_instance is not None:
            return
        model_name = os.path.splitext(os.path.basename(self.path))[0]
        Logger().get_logger().setLevel(log.level)
        self._nc_model_instance = NCModel(self.path)
        self._nc_model_instance.name = model_name

    @staticmethod
    def supports_path(path: str) -> bool:
        """Check if given path is of supported model."""
        return "onnx" == get_file_extension(path)

    def guard_requirements_installed(self) -> None:
        """Ensure all requirements are installed."""
        check_module("onnx")
        check_module("onnxruntime")
        if sys.version_info < (3, 10):  # pragma: no cover
            check_module("onnxruntime_extensions")

    @property
    def filtered_input_nodes(self) -> List[Any]:
        """Get filtered input nodes."""
        input_nodes = self.nc_model_instance.graph().input  # pylint: disable=E1101
        name_to_input = {}
        for input in input_nodes:
            name_to_input[input.name] = input
        for initializer in self.nc_model_instance.graph().initializer:  # pylint: disable=E1101
            if initializer.name in name_to_input:
                input_nodes.remove(name_to_input[initializer.name])
        return input_nodes


def remove_number_of_samples_from_shape(shape: str) -> str:
    """Remove number of samples from shape if exists."""
    shape_as_list = shape.split(",")
    if len(shape_as_list) > 3:
        shape_as_list.pop(0)
    return ",".join(shape_as_list)
