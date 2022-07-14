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
"""The OnnxRT diagnosis class."""
from typing import Optional

from neural_compressor.ux.components.diagnosis.diagnosis import Diagnosis
from neural_compressor.ux.components.model.onnxrt.model import OnnxrtModel
from neural_compressor.ux.components.model.repository import ModelRepository
from neural_compressor.ux.components.optimization.optimization import Optimization


class OnnxRtDiagnosis(Diagnosis):
    """OnnxRuntime diagnosis class."""

    def __init__(self, optimization: Optimization):
        """Initialize OnnxRtDiagnosis."""
        super().__init__(optimization)
        self._model: Optional[OnnxrtModel] = None

    @property
    def model(self) -> OnnxrtModel:
        """Get Neural Compressor Model instance."""
        self._ensure_model()
        return self._model  # type: ignore

    def _ensure_model(self) -> None:
        """Create INC Bench Model instance if needed."""
        if self._model is not None:
            return
        model_repository = ModelRepository()
        self._model = model_repository.get_model(self.model_path)  # type: ignore
