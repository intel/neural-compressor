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
"""Configuration pruning module."""
import re
from typing import Any, Dict, List, Optional, Union

from neural_compressor.conf.config import Pruner
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.workload.dataloader import Dataloader
from neural_compressor.ux.utils.workload.evaluation import Postprocess


class SGDOptimizer(JsonSerializer):
    """Configuration SGDOptimizer class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration SGDOptimizer class."""
        super().__init__()
        self.learning_rate: float = float(data.get("learning_rage", None))
        self.momentum = data.get("momentum", None)
        self.nesterov = data.get("nesterov", None)
        self.weight_decay = data.get("weight_decay", None)


class AdamWOptimizer(JsonSerializer):
    """Configuration AdamWOptimizer class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration AdamWOptimizer class."""
        super().__init__()
        self.weight_decay: float = float(data.get("weight_decay", None))
        self.learning_rate = data.get("learning_rate", None)
        self.beta_1 = data.get("beta_1", None)
        self.beta_2 = data.get("beta_2", None)
        self.epsilon = data.get("epsilon", None)
        self.amsgrad = data.get("amsgrad", None)


class AdamOptimizer(JsonSerializer):
    """Configuration AdamOptimizer class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration AdamOptimizer class."""
        super().__init__()
        self.learning_rate = data.get("learning_rate", None)
        self.beta_1 = data.get("beta_1", None)
        self.beta_2 = data.get("beta_2", None)
        self.epsilon = data.get("epsilon", None)
        self.amsgrad = data.get("amsgrad", None)


class Optimizer(JsonSerializer):
    """Configuration Optimizer class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Optimizer class."""
        super().__init__()
        self.SGD: Optional[SGDOptimizer] = None
        if isinstance(data.get("SDG", None), dict):
            self.SGD = SGDOptimizer(data["SDG"])
        self.AdamW: Optional[AdamWOptimizer] = None
        if isinstance(data.get("AdamW", None), dict):
            self.AdamW = AdamWOptimizer(data["AdamW"])
        self.Adam: Optional[AdamOptimizer] = None
        if isinstance(data.get("Adam", None), dict):
            self.Adam = AdamOptimizer(data["Adam"])


class CrossEntropyLossCriterion(JsonSerializer):
    """Configuration CrossEntropyLossCriterion class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration CrossEntropyLossCriterion class."""
        super().__init__()
        self.reduction = data.get("reduction", None)
        self.from_logits = data.get("from_logits", None)


class SparseCategoricalCrossentropyCriterion(JsonSerializer):
    """Configuration SparseCategoricalCrossentropyCriterion class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration SparseCategoricalCrossentropyCriterion class."""
        super().__init__()
        self.reduction = data.get("reduction", None)
        self.from_logits = data.get("from_logits", None)


class KnowledgeDistillationLossCriterion(JsonSerializer):
    """Configuration KnowledgeDistillationLossCriterion class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration KnowledgeDistillationLossCriterion class."""
        super().__init__()
        self.temperature = data.get("temperature", None)
        self.loss_types = data.get("loss_types", None)
        self.loss_weights = data.get("loss_weights", None)


class Criterion(JsonSerializer):
    """Configuration Criterion class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Criterion class."""
        super().__init__()
        self.CrossEntropyLoss: Optional[CrossEntropyLossCriterion] = None
        if isinstance(data.get("CrossEntropyLoss", None), dict):
            self.CrossEntropyLoss = CrossEntropyLossCriterion(data["CrossEntropyLoss"])

        self.SparseCategoricalCrossentropy: Optional[SparseCategoricalCrossentropyCriterion] = None
        if isinstance(data.get("SparseCategoricalCrossentropy", None), dict):
            self.SparseCategoricalCrossentropy = SparseCategoricalCrossentropyCriterion(
                data["SparseCategoricalCrossentropy"],
            )

        self.KnowledgeDistillationLoss: Optional[KnowledgeDistillationLossCriterion] = None
        if isinstance(data.get("KnowledgeDistillationLoss", None), dict):
            self.KnowledgeDistillationLoss = KnowledgeDistillationLossCriterion(
                data["KnowledgeDistillationLoss"],
            )


class Train(JsonSerializer):
    """Configuration Train class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Train class."""
        super().__init__()
        self.optimizer = Optimizer(data.get("optimizer", {}))
        self.criterion = Criterion(data.get("criterion", {}))
        self.dataloader = Dataloader(data.get("dataloader", {}))
        self.epoch = data.get("epoch", None)
        self.start_epoch = data.get("start_epoch", None)
        self.end_epoch = data.get("end_epoch", None)
        self.iteration = data.get("iteration", None)
        self.frequency = data.get("frequency", None)
        self.execution_mode = data.get("execution_mode", None)
        self.postprocess: Optional[Postprocess] = None
        if isinstance(data.get("postprocess", None), dict):
            self.postprocess = Postprocess(data.get("postprocess", {}))
        self.hostfile = data.get("hostfile", None)

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Serialize Pruning class."""
        result = {}
        for key, value in self.__dict__.items():
            if key in self._skip:
                continue
            if value is None:
                continue
            variable_name = re.sub(r"^_", "", key)
            getter_value = value
            try:
                getter_value = getattr(self, variable_name)
            except AttributeError:
                log.warning(f"Found f{key} attribute without {variable_name} getter.")

            serialized_value = self._serialize_value(
                getter_value,
                serialization_type,
            )

            if serialized_value:
                result[variable_name] = serialized_value
        return result


class WeightCompressionApproach(JsonSerializer):
    """Configuration WeightCompressionApproach class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration WeightCompressionApproach class."""
        super().__init__()
        self.initial_sparsity: Optional[float] = parse_dict_value_to_float(
            data,
            "initial_sparsity",
        )

        self.target_sparsity: Optional[float] = parse_dict_value_to_float(
            data,
            "target_sparsity",
        )
        self.start_epoch: Optional[int] = parse_dict_value_to_int(data, "start_epoch")
        self.end_epoch: Optional[int] = parse_dict_value_to_int(data, "end_epoch")
        self.pruners: List[Pruner] = data.get("pruners", [])


class Approach(JsonSerializer):
    """Configuration Approach class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Approach class."""
        super().__init__()
        self.weight_compression: Optional[WeightCompressionApproach] = None
        if isinstance(data.get("weight_compression", {}), dict):
            self.weight_compression = WeightCompressionApproach(data["weight_compression"])


class Pruning(JsonSerializer):
    """Configuration Pruning class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Pruning class."""
        super().__init__()
        self.train: Optional[Train] = None
        if data.get("train", {}):
            self.train = Train(data.get("train", {}))
        self.approach: Optional[Approach] = None
        if data.get("approach", {}):
            self.approach = Approach(data.get("approach", {}))


def parse_dict_value_to_float(data: dict, key: str) -> Optional[float]:
    """Parse value to float or None if value is None."""
    try:
        parsed_float = float(data.get(key))  # type: ignore
        return parsed_float
    except ValueError:
        raise Exception("Could not parse value to float.")
    except TypeError:
        return None


def parse_dict_value_to_int(data: dict, key: str) -> Optional[int]:
    """Parse value to float or None if value is None."""
    try:
        parsed_int = int(data.get(key))  # type: ignore
        return parsed_int
    except ValueError:
        raise Exception("Could not parse value to float.")
    except TypeError:
        return None
