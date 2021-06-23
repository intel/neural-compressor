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
"""Dict with items expiring after given time."""
import time
from collections import UserDict
from typing import Any, Optional


class ExpiringDictItem:
    """Item that knows it it's already expired."""

    def __init__(self, value: Any, expires_at: float):
        """Create object."""
        self.value = value
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        """Check if item is already expired."""
        return time.time() > self.expires_at


class ExpiringDict(UserDict):
    """Dict with items expiring after given time."""

    def __init__(self, initial_value: Optional[dict] = None, ttl: int = 120) -> None:
        """Create object."""
        super().__init__()
        self.ttl = ttl
        if initial_value is None:
            initial_value = {}

        for (key, value) in initial_value.items():
            self[key] = value

    def __setitem__(self, key: str, item: Any) -> None:
        """Add item to dict."""
        super().__setitem__(key, self._create_item(value=item))

    def __getitem__(self, key: str) -> Any:
        """Get item from dict."""
        item: ExpiringDictItem = super().__getitem__(key)
        if item.is_expired():
            raise KeyError(key)
        return item.value

    def _create_item(self, value: Any) -> ExpiringDictItem:
        """Create item for collection."""
        return ExpiringDictItem(value=value, expires_at=time.time() + self.ttl)
