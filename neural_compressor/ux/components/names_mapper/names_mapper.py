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
"""Names mapper."""
from enum import Enum

from neural_compressor.ux.utils.consts import DomainFlavours, Domains, Frameworks, Precisions


class MappingDirection(Enum):
    """Mapping directions enumeration."""

    ToBench = "bench"
    ToCore = "core"


class NamesMapper:
    """Names Mapper class."""

    def __init__(self, direction: MappingDirection) -> None:
        """Initialize mapper."""
        self.direction = direction

    def map_name(self, parameter_type: str, value: str) -> str:
        """Map specified value to correct direction."""
        mappings = {
            "framework": self.framework_mappings,
            "domain": self.domain_mappings,
            "domain_flavour": self.domain_flavour_mappings,
            "precision": self.precision_mappings,
        }
        mapping = mappings.get(parameter_type, None)
        if mapping is None:
            return value
        return self._map(mapping, value)

    def _map(self, mapping: dict, value: str) -> str:
        """Map framework name."""
        mapped_value = mapping.get(self.direction, {}).get(value, None)
        if mapped_value is None:
            return value
        return mapped_value

    @property
    def framework_mappings(self) -> dict:
        """Get framework mappings."""
        return {
            MappingDirection.ToCore: {
                Frameworks.TF.value: "tensorflow",
                Frameworks.ONNX.value: "onnxrt_qlinearops",
                Frameworks.PT.value: "pytorch",
            },
            MappingDirection.ToBench: {
                "tensorflow": Frameworks.TF.value,
                "onnxrt_qlinearops": Frameworks.ONNX.value,
                "pytorch": Frameworks.PT.value,
            },
        }

    @property
    def domain_mappings(self) -> dict:
        """Get domain mappings."""
        return {
            MappingDirection.ToCore: {
                Domains.NONE.value: "",
                Domains.IMAGE_RECOGNITION.value: "image_recognition",
                Domains.OBJECT_DETECTION.value: "object_detection",
                Domains.RECOMMENDATION.value: "recommendation",
                Domains.NLP.value: "nlp",
            },
            MappingDirection.ToBench: {
                "": Domains.NONE.value,
                "image_recognition": Domains.IMAGE_RECOGNITION.value,
                "object_detection": Domains.OBJECT_DETECTION.value,
                "recommendation": Domains.RECOMMENDATION.value,
                "nlp": Domains.NLP.value,
            },
        }

    @property
    def domain_flavour_mappings(self) -> dict:
        """Get domain flavour mappings."""
        return {
            MappingDirection.ToCore: {
                DomainFlavours.NONE.value: "",
                DomainFlavours.SSD.value: "ssd",
                DomainFlavours.YOLO.value: "yolo",
            },
            MappingDirection.ToBench: {
                "": DomainFlavours.NONE.value,
                "ssd": DomainFlavours.SSD.value,
                "yolo": DomainFlavours.YOLO.value,
            },
        }

    @property
    def precision_mappings(self) -> dict:
        """Get precision mappings."""
        return {
            MappingDirection.ToCore: {
                Precisions.BF16.value: "bf16,fp32",
            },
            MappingDirection.ToBench: {
                "bf16,fp32": Precisions.BF16.value,
            },
        }
