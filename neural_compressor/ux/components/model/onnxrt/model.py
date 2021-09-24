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
"""Onnxrt model class."""

from neural_compressor.ux.components.model.model import Model
from neural_compressor.ux.utils.utils import check_module, get_file_extension


class OnnxrtModel(Model):
    """Onnxrt Model class."""

    @staticmethod
    def get_framework_name() -> str:
        """Get the name of framework."""
        return "onnxrt"

    @staticmethod
    def supports_path(path: str) -> bool:
        """Check if given path is of supported model."""
        return "onnx" == get_file_extension(path)

    def guard_requirements_installed(self) -> None:
        """Ensure all requirements are installed."""
        check_module("onnx")
        check_module("onnxruntime")
