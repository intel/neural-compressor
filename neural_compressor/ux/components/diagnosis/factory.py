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
"""Diagnosis class factory."""
from neural_compressor.ux.components.diagnosis.diagnosis import Diagnosis
from neural_compressor.ux.components.diagnosis.onnx_diagnosis.onnxrt_diagnosis import (
    OnnxRtDiagnosis,
)
from neural_compressor.ux.components.diagnosis.tensorflow_diagnosis.tensorflow_diagnosis import (
    TensorflowDiagnosis,
)
from neural_compressor.ux.components.names_mapper.names_mapper import MappingDirection, NamesMapper
from neural_compressor.ux.components.optimization.optimization import Optimization
from neural_compressor.ux.utils.consts import Frameworks
from neural_compressor.ux.utils.exceptions import InternalException
from neural_compressor.ux.utils.logger import log


class DiagnosisFactory:
    """Optimization factory."""

    @staticmethod
    def get_diagnosis(
        optimization: Optimization,
    ) -> Diagnosis:
        """Get diagnosis for specified framework."""
        try:
            names_mapper = NamesMapper(MappingDirection.ToBench)
            framework_name: str = names_mapper.map_name(
                parameter_type="framework",
                value=optimization.framework,
            )
        except KeyError:
            raise InternalException("Missing framework name.")
        diagnosis_map = {
            Frameworks.ONNX.value: OnnxRtDiagnosis,
            Frameworks.TF.value: TensorflowDiagnosis,
        }
        diagnosis = diagnosis_map.get(framework_name, None)
        if diagnosis is None:
            raise InternalException(
                f"Could not find diagnosis class for {framework_name} framework.",
            )
        log.debug(f"Initializing {diagnosis.__name__} class.")
        return diagnosis(optimization)
