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
"""JsonSerializer module."""

import re
from typing import Any, Dict, List, Optional, Union

from lpot.ux.utils.logger import log


class JsonSerializer:
    """Dict serializable class."""

    def __init__(self) -> None:
        """Initialize json serializable class."""
        # List of variable names that will
        # be skipped during serialization
        self._skip = ["_skip"]

    def serialize(
        self,
        serialization_type: str = "default",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Serialize class to dict.

        :param serialization_type: serialization type, defaults to "default"
        :type serialization_type: str, optional
        :return: serialized class
        :rtype: Union[dict, List[dict]]
        """
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

            if serialized_value is not None:
                result[variable_name] = serialized_value

        return result

    def _serialize_value(self, value: Any, serialization_type: str) -> Any:
        """Serialize single value."""
        if isinstance(value, list):
            return self._serialize_list(value, serialization_type)
        if isinstance(value, dict):
            return self._serialize_dict(value, serialization_type)
        else:
            return self.serialize_item(value, serialization_type)

    def _serialize_list(
        self,
        value: list,
        serialization_type: str,
    ) -> Optional[List[Any]]:
        """Serialize list."""
        serialized_list = []

        for item in value:
            serialized_item = self.serialize_item(item, serialization_type)
            if serialized_item is not None:
                serialized_list.append(serialized_item)

        if len(serialized_list) == 0:
            return None

        return serialized_list

    def _serialize_dict(
        self,
        value: dict,
        serialization_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Serialize dict."""
        serialized_dict = {}

        for key in value.keys():
            serialized_item = self.serialize_item(value[key], serialization_type)
            if serialized_item is not None:
                serialized_dict[key] = serialized_item

        if len(serialized_dict.keys()) == 0:
            return None

        return serialized_dict

    @staticmethod
    def serialize_item(value: Any, serialization_type: str = "default") -> Any:
        """
        Serialize objects that don't support json dump.

        i.e datetime object can't be serialized to JSON format and throw an TypeError exception
        TypeError: datetime.datetime(2016, 4, 8, 11, 22, 3, 84913) is not JSON serializable
        To handle that override method serialize_item to convert object
            >>> serialize_item(datetime)
            "2016-04-08T11:22:03.084913"

        For all other cases it should return serializable object i.e. str, int float

        :param value: Any type
        :param serialization_type: serialization type
        :return: Value that can be handled by json.dump
        """
        if issubclass(type(value), JsonSerializer):
            # pylint: disable=maybe-no-member
            serialized_value = value.serialize(serialization_type)
            if isinstance(serialized_value, dict) and not serialized_value:  # Ignore empty objects
                return None
            return serialized_value
        return value
