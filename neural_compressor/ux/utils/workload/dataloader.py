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
"""Configuration dataloader module."""

from collections import OrderedDict
from typing import Any, Dict, Optional

from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.json_serializer import JsonSerializer


class Dataset(JsonSerializer):
    """Configuration Dataset class."""

    def __init__(self, name: str, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Dataset class."""
        super().__init__()
        self.name: str = name
        self.params: dict = data

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Dataset class."""
        if self.params == {}:
            return {
                self.name: None,
            }
        return {
            self.name: self.params,
        }


class LabelBalance(JsonSerializer):
    """Configuration LabelBalance class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration LabelBalance class."""
        super().__init__()
        self.size = data.get("size", None)  # >0


class Filter(JsonSerializer):
    """Configuration Filter class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Filter class."""
        super().__init__()
        self.LabelBalance = None
        if data.get("LabelBalance"):
            self.LabelBalance = LabelBalance(data.get("LabelBalance", {}))


class Transform(JsonSerializer):
    """Configuration Transform class."""

    def __init__(self, name: str, parameters: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Transform class."""
        super().__init__()
        self.name: str = name
        self.parameters: Dict[str, Any] = parameters

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Transform class."""
        return self.parameters


class Dataloader(JsonSerializer):
    """Configuration Dataloader class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize Configuration Dataloader class."""
        super().__init__()
        self.last_batch = data.get("last_batch", None)  # One of "rollover", "discard"
        self.batch_size: Optional[int] = data.get("batch_size")
        self.dataset: Optional[Dataset] = None
        if data.get("dataset"):
            self.set_dataset(data.get("dataset", {}))

        self.transform: OrderedDict = OrderedDict()
        if data.get("transform"):
            for key, value in data.get("transform", {}).items():
                self.transform.update({key: Transform(key, value)})
                self.transform.move_to_end(key)

        self.filter = None
        if data.get("filter"):
            self.filter = Filter(data.get("filter", {}))

    def set_dataset(self, dataset_data: Dict[str, Any]) -> None:
        """Set dataset for dataloader."""
        if len(dataset_data.keys()) > 1:
            raise ClientErrorException(
                "There can be specified only one dataset per dataloader. "
                f"Found keys: {', '.join(dataset_data.keys())}.",
            )
        for key, val in dataset_data.items():
            self.dataset = Dataset(key, val)

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Dataloader class."""
        result = {}
        if self.last_batch:
            result["last_batch"] = self.last_batch

        if self.batch_size:
            result["batch_size"] = self.batch_size

        if self.dataset:
            result["dataset"] = self.dataset.serialize()

        if self.transform:
            transforms = {}
            for key, val in self.transform.items():
                transforms.update({key: val.serialize()})
            result["transform"] = transforms

        if self.filter:
            result["filter"] = self.filter.serialize()

        return result
