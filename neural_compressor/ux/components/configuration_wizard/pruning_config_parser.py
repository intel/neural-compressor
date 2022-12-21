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
"""Configuration type parser."""
from typing import Any


class PruningConfigParser:
    """Pruning configuration parser class."""

    def parse(self, input_data: list) -> dict:
        """Parse configuration."""
        raise NotImplementedError

    def generate_tree(self, input_data: dict) -> list:
        """Generate tree from pruning configuration."""
        parsed_tree = self.parse_entry(input_data)
        return parsed_tree

    def parse_entry(self, input_data: dict) -> Any:
        """Parse configuration entry to tree element."""
        config_tree = []
        for key, value in input_data.items():
            if key in ["train", "approach"] and value is None:
                continue
            parsed_entry = {"name": key}
            if isinstance(value, dict):
                children = self.parse_entry(value)
                parsed_entry.update({"children": children})
            elif isinstance(value, list):
                for list_entry in value:
                    parsed_list_entries = self.parse_entry(list_entry)
                    parsed_entry.update({"children": parsed_list_entries})
            else:
                parsed_entry.update({"value": value})
            config_tree.append(parsed_entry)
        return config_tree
