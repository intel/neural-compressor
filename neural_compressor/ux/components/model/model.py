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
"""The abstract model class."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from neural_compressor.ux.components.graph.graph import Graph
from neural_compressor.ux.components.model.domain import Domain


class Model(ABC):
    """Abstract model class."""

    def __init__(self, path: str) -> None:
        """Initialize object."""
        self.ensure_supported_path(path)
        self.path = path

    def get_input_nodes(self) -> Optional[List[Any]]:
        """Get model input nodes."""
        self.guard_requirements_installed()
        return None

    def get_output_nodes(self) -> Optional[List[Any]]:
        """Get model output nodes."""
        self.guard_requirements_installed()
        return None

    def ensure_supported_path(self, path: str) -> None:
        """Make sure that provided path is to supported model."""
        if not self.supports_path(path):
            klass = self.__class__
            model_name = ".".join([klass.__module__, klass.__qualname__])
            raise AttributeError(f"Model path: {path} is not supported by {model_name} class.")

    def get_model_graph(self) -> Graph:
        """Get model graph."""
        raise NotImplementedError(f"Reading graph for model {self.path} is not supported.")

    @property
    def domain(self) -> Domain:
        """Get model domain."""
        try:
            node_names = {node.id for node in self.get_model_graph().nodes}
        except Exception:
            return Domain()

        def has_name_part(name_part: str) -> bool:
            """Check if a node exists with given name_part."""
            return bool([name for name in node_names if name_part in name])

        def has_all_name_parts(name_parts: List[str]) -> bool:
            """Check if there is a node for every name_part."""
            missing = [name_part for name_part in name_parts if not has_name_part(name_part)]
            return not missing

        if has_all_name_parts(["bboxes", "scores", "classes", "ssd"]):
            return Domain(domain="object_detection", domain_flavour="ssd")
        if has_all_name_parts(["boxes", "yolo"]):
            return Domain(domain="object_detection", domain_flavour="yolo")
        if has_all_name_parts(["boxes", "scores", "classes"]):
            return Domain(domain="object_detection")
        if has_all_name_parts(["resnet"]):
            return Domain(domain="image_recognition")
        return Domain()

    @staticmethod
    @abstractmethod
    def get_framework_name() -> str:
        """Get the name of framework."""

    @staticmethod
    @abstractmethod
    def supports_path(path: str) -> bool:
        """Check if given path is of supported model."""

    @abstractmethod
    def guard_requirements_installed(self) -> None:
        """Ensure all requirements are installed."""
