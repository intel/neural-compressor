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
"""Abstract Tensorflow model class."""
import os.path
from typing import Any, List, Optional

from lpot.experimental.common.model import Model as LpotModel
from lpot.model.base_model import BaseModel
from lpot.utils.logger import Logger
from lpot.ux.utils.logger import log
from lpot.ux.utils.utils import check_module

from ..model import Model


class TensorflowModel(Model):
    """Abstract TensorFlow Model class."""

    def __init__(self, path: str) -> None:
        """Initialize object."""
        super().__init__(path)
        self._lpot_model_instance: Optional[BaseModel] = None

    def get_input_nodes(self) -> Optional[List[Any]]:
        """Get model input nodes."""
        self.guard_requirements_installed()

        return getattr(self.lpot_model_instance, "input_node_names", [])

    def get_output_nodes(self) -> Optional[List[Any]]:
        """Get model output nodes."""
        self.guard_requirements_installed()

        return getattr(self.lpot_model_instance, "output_node_names", []) + ["custom"]

    @property
    def lpot_model_instance(self) -> BaseModel:
        """Get LPOT Model instance."""
        self._ensure_lpot_model_instance()
        return self._lpot_model_instance

    def _ensure_lpot_model_instance(self) -> None:
        """Create LPOT Model instance if needed."""
        if self._lpot_model_instance is not None:
            return
        model_name = os.path.splitext(os.path.basename(self.path))[0]
        Logger().get_logger().setLevel(log.level)
        self._lpot_model_instance = LpotModel(self.path)
        self._lpot_model_instance.name = model_name

    @staticmethod
    def get_framework_name() -> str:
        """Get the name of framework."""
        return "tensorflow"

    def guard_requirements_installed(self) -> None:
        """Ensure all requirements are installed."""
        check_module("tensorflow")
