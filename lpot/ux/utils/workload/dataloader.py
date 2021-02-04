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
"""LPOT configuration dataloader module."""

from typing import Any, Dict

from lpot.ux.utils.json_serializer import JsonSerializer


class Dataset(JsonSerializer):
    """LPOT configuration Dataset class."""

    def __init__(self, name: str, data: Dict[str, Any] = {}) -> None:
        """Initialize LPOT configuration Dataset class."""
        super().__init__()
        self.name: str = name
        self.root: str = data.get("root", None)

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Dataset class."""
        return {
            self.name: {
                "root": self.root,
            },
        }


class LabelBalance(JsonSerializer):
    """LPOT configuration LabelBalance class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize LPOT configuration LabelBalance class."""
        super().__init__()
        self.size = data.get("size", None)  # >0


class Filter(JsonSerializer):
    """LPOT configuration Filter class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize LPOT configuration Filter class."""
        super().__init__()
        self.LabelBalance = None
        if data.get("LabelBalance"):
            self.LabelBalance = LabelBalance(data.get("LabelBalance", {}))


class Transform(JsonSerializer):
    """LPOT configuration Transform class."""

    def __init__(self, name: str, parameters: Dict[str, Any] = {}) -> None:
        """Initialize LPOT configuration Transform class."""
        super().__init__()
        self.name: str = name
        self.parameters: Dict[str, Any] = parameters

    def serialize(self, serialization_type: str = "default") -> Dict[str, Any]:
        """Serialize Transform class."""
        return self.parameters


class Dataloader(JsonSerializer):
    """LPOT configuration Dataloader class."""

    def __init__(self, data: Dict[str, Any] = {}) -> None:
        """Initialize LPOT configuration Dataloader class."""
        super().__init__()
        self.last_batch = data.get("last_batch", None)  # One of "rollover", "discard"
        self.batch_size: int = data.get("batch_size", 1)  # > 0
        self.dataset = None
        if data.get("dataset"):
            for key, value in data.get("dataset", {}).items():
                self.dataset = Dataset(key, value)

        self.transform = {}
        if data.get("transform"):
            for key, value in data.get("transform", {}).items():
                self.transform.update({key: Transform(key, value)})

        self.filter = None
        if data.get("filter"):
            self.filter = Filter(data.get("filter", {}))

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
