# -*- coding: utf-8 -*-
# Copyright (c) 2023 Intel Corporation
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
from neural_insights.components.diagnosis.diagnosis import Diagnosis
from neural_insights.components.diagnosis.onnx_diagnosis.onnxrt_diagnosis import OnnxRtDiagnosis
from neural_insights.components.diagnosis.tensorflow_diagnosis.tensorflow_diagnosis import TensorflowDiagnosis
from neural_insights.components.workload_manager.workload import Workload
from neural_insights.utils.consts import Frameworks
from neural_insights.utils.exceptions import InternalException
from neural_insights.utils.logger import log


class DiagnosisFactory:
    """Optimization factory."""

    @staticmethod
    def get_diagnosis(
        workload: Workload,
    ) -> Diagnosis:
        """Get diagnosis for specified framework."""
        diagnosis_map = {
            Frameworks.ONNX: OnnxRtDiagnosis,
            Frameworks.TF: TensorflowDiagnosis,
        }
        diagnosis = diagnosis_map.get(workload.framework, None)
        if diagnosis is None:
            raise InternalException(
                f"Could not find diagnosis class for {workload.framework.value} framework.",
            )
        log.debug(f"Initializing {diagnosis.__name__} class.")
        return diagnosis(workload)
