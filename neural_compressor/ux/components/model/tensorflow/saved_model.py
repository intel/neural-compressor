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
"""Tensorflow saved_model model."""

from neural_compressor.ux.components.model.model_type_getter import get_model_type
from neural_compressor.ux.components.model.tensorflow.model import TensorflowModel as TFModel


class SavedModelModel(TFModel):
    """Saved_model model."""

    @staticmethod
    def supports_path(path: str) -> bool:
        """Check if given path is of supported model."""
        return "saved_model" == get_model_type(path)
