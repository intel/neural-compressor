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
"""Tensorflow slim model."""

import re

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.utils import check_module, get_module_version

from ..model_type_getter import get_model_type
from .model import TensorflowModel as TFModel


class SlimModel(TFModel):
    """Slim model."""

    @staticmethod
    def supports_path(path: str) -> bool:
        """Check if given path is of supported model."""
        try:
            return "slim" == get_model_type(path)
        except ValueError:
            return False

    def guard_requirements_installed(self) -> None:
        """Ensure all requirements are installed."""
        super().guard_requirements_installed()

        tensorflow_version = get_module_version("tensorflow")
        if not tensorflow_version.startswith("1."):
            raise ClientErrorException(
                "TensorFlow slim models work only with TensorFlow 1.x. "
                f"Currently installed version is {tensorflow_version}. "
                "Please install TensorFlow 1.x to tune selected model.",
            )

        check_module("tf_slim")

    def _ensure_lpot_model_instance(self) -> None:
        """Create LPOT Model instance if needed."""
        try:
            super()._ensure_lpot_model_instance()
        except AssertionError as err:
            if "only support topology" in str(err):
                search = re.match(r"only support topology \[(.*)\]", str(err))
                if search:
                    supported_topologies = search.group(1)
                    raise ClientErrorException(
                        "Slim model require correct filename. "
                        "Check if your model is among the following and rename "
                        "it according to this notation. "
                        f"Supported models: {supported_topologies}",
                    )
            raise err
