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
from neural_compressor.ux.utils.consts import postprocess_transforms
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.json_serializer import JsonSerializer
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.workload.dataloader import Dataloader
from neural_compressor.ux.utils.workload.evaluation import Postprocess, PostprocessSchema


class SGDOptimizer(JsonSerializer):
    """Configuration SGDOptimizer class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration SGDOptimizer class."""
        super().__init__()
        self.learning_rate: float = float(data.get("learning_rate", None))
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


class IntermediateLayersKnowledgeDistillationLoss(JsonSerializer):
    """Configuration IntermediateLayersKnowledgeDistillationLoss class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration IntermediateLayersKnowledgeDistillationLoss class."""
        super().__init__()
        self.layer_mappings = data.get("layer_mappings", None)
        self.loss_types = data.get("loss_types", None)
        self.loss_weights = data.get("loss_weights", None)
        self.add_origin_loss = data.get("add_origin_loss", None)


class SelfKnowledgeDistillationLoss(JsonSerializer):
    """Configuration SelfKnowledgeDistillationLoss class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration SelfKnowledgeDistillationLoss class."""
        super().__init__()
        self.layer_mappings = data.get("layer_mappings", None)
        self.loss_types = data.get("loss_types", None)
        self.loss_weights = data.get("loss_weights", None)
        self.add_origin_loss = data.get("add_origin_loss", None)
        self.temperature = data.get("temperature", None)


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

        self.IntermediateLayersKnowledgeDistillationLoss: Optional[
            IntermediateLayersKnowledgeDistillationLoss
        ] = None
        if isinstance(data.get("IntermediateLayersKnowledgeDistillationLoss", None), dict):
            self.IntermediateLayersKnowledgeDistillationLoss = (
                IntermediateLayersKnowledgeDistillationLoss(
                    data["IntermediateLayersKnowledgeDistillationLoss"],
                )
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

    def set_postprocess_transforms(self, transforms: List[Dict[str, Any]]) -> None:
        """Set postprocess transformation."""
        if transforms is None or len(transforms) <= 0:
            return
        transform_names = {transform["name"] for transform in transforms}
        has_postprocess_transforms = len(transform_names.intersection(postprocess_transforms)) > 0
        if not has_postprocess_transforms:
            return

        if self.postprocess is None:
            self.postprocess = Postprocess()

        postprocess_transforms_data = {}
        for single_transform in transforms:
            if single_transform["name"] in postprocess_transforms:
                postprocess_transforms_data.update(
                    {single_transform["name"]: single_transform["params"]},
                )
        self.postprocess.transform = PostprocessSchema(postprocess_transforms_data)

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
        self.max_sparsity_ratio_per_layer: Optional[float] = parse_dict_value_to_float(
            data,
            "max_sparsity_ratio_per_layer",
        )
        self.prune_type: Optional[str] = data.get("prune_type", None)

        self.start_epoch: Optional[int] = parse_dict_value_to_int(data, "start_epoch")
        self.end_epoch: Optional[int] = parse_dict_value_to_int(data, "end_epoch")

        self.start_step: Optional[int] = parse_dict_value_to_int(data, "start_step")
        self.end_step: Optional[int] = parse_dict_value_to_int(data, "end_step")

        self.update_frequency: Optional[float] = parse_dict_value_to_float(
            data,
            "update_frequency",
        )
        self.update_frequency_on_step: Optional[int] = parse_dict_value_to_int(
            data,
            "update_frequency_on_step",
        )
        self.excluded_names: List[str] = data.get("excluded_names", [])
        self.prune_domain: Optional[str] = data.get("prune_domain", None)
        self.names: List[str] = data.get("names", [])
        self.extra_excluded_names: List[str] = data.get("extra_excluded_names", [])
        self.prune_layer_type: Optional[List[Any]] = data.get("prune_layer_type", None)
        self.sparsity_decay_type: Optional[str] = data.get("sparsity_decay_type", None)
        self.pattern: Optional[str] = data.get("pattern", None)
        self.pruners: List[Pruner] = self.initialize_pruners(data.get("pruners", []))

    @staticmethod
    def initialize_pruners(pruner_dict_list: List[Union[dict, Pruner]]) -> List[Pruner]:
        """Initialize list of pruners from dict format."""
        pruner_list = []
        for pruner_entry in pruner_dict_list:
            if isinstance(pruner_entry, Pruner):
                pruner_list.append(pruner_entry)
                continue
            if isinstance(pruner_entry, dict):
                pruner_list.append(Pruner(**pruner_entry))
                continue
            raise ClientErrorException("Could not initialize pruners.")

        return pruner_list

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Serialize WeightCompressionApproach class."""
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

            if variable_name == "pruners":
                serialized_value = [self.serialize_pruner(pruner) for pruner in getter_value]
            else:
                serialized_value = self._serialize_value(
                    getter_value,
                    serialization_type,
                )

            if serialized_value:
                result[variable_name] = serialized_value
        return result

    @staticmethod
    def serialize_pruner(pruner: Pruner) -> dict:
        """Serialize INC Pruner instance."""
        pruner_fields = [
            "start_epoch",
            "end_epoch",
            "update_frequency",
            "target_sparsity",
            "initial_sparsity",
            "start_step",
            "end_step",
            "update_frequency_on_step",
            "prune_domain",
            "sparsity_decay_type",
            "extra_excluded_names",
            "pattern",
            "prune_type",
            "method",
            "names",
            "parameters",
        ]
        serialized_pruner = {}
        for field in pruner_fields:
            field_value = getattr(pruner, field)
            if field_value is None:
                continue
            serialized_pruner.update({field: field_value})
        return serialized_pruner


class Approach(JsonSerializer):
    """Configuration Approach class."""

    def __init__(self, data: Dict[str, Any] = {}):
        """Initialize Configuration Approach class."""
        super().__init__()

        self.weight_compression: Optional[WeightCompressionApproach] = None
        if isinstance(data.get("weight_compression", None), dict):
            self.weight_compression = WeightCompressionApproach(data["weight_compression"])

        self.weight_compression_pytorch: Optional[WeightCompressionApproach] = None
        if isinstance(data.get("weight_compression_pytorch", None), dict):
            self.weight_compression_pytorch = WeightCompressionApproach(
                data["weight_compression_pytorch"],
            )


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
