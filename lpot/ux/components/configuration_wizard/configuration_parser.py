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
"""Configuration type parser."""
import json
from collections.abc import Iterable
from typing import Any, Dict, List, Type, Union

from lpot.ux.utils.exceptions import ClientErrorException
from lpot.ux.utils.hw_info import HWInfo
from lpot.ux.utils.utils import parse_bool_value


class ConfigurationParser:
    """Configuration type parser class."""

    def __init__(self) -> None:
        """Initialize configuration type parser."""
        self.transform_types: Dict[str, List[str]] = {
            "str": ["interpolation", "dtype", "label_file", "vocab_file"],
            "int": [
                "x",
                "y",
                "height",
                "width",
                "size",
                "offset_height",
                "offset_width",
                "target_height",
                "target_width",
                "dim",
                "resize_side",
                "label_shift",
                "size",
            ],
            "float": ["scale", "central_fraction"],
            "list<float>": ["mean", "std", "perm", "mean_value", "ratio"],
            "bool": ["random_crop", "random_flip_left_right"],
        }
        self.dataloader_types: Dict[str, List[str]] = {
            "str": [
                "root",
                "filenames",
                "compression_type",
                "data_path",
                "image_list",
                "img_dir",
                "anno_dir",
                "content_folder",
                "style_folder",
                "image_format",
                "dtype",
                "label_file",
            ],
            "int": ["buffer_size", "num_parallel_reads", "num_cores"],
            "list<int>": ["resize_shape"],
            "float": ["crop_ratio"],
            "list<float>": ["low", "high"],
            "list<list<int>>": ["shape"],
            "bool": ["train", "label"],
        }
        self.types_definitions: Dict[str, Union[Type, List[Any]]] = {
            "str": str,
            "int": int,
            "list<int>": [int],
            "list<list<int>>": [[int]],
            "float": float,
            "list<float>": [float],
            "bool": bool,
        }

    def parse(self, data: dict) -> dict:
        """Parse configuration."""
        data = set_defaults(data)

        transforms_data = data.get("transform", None)
        if transforms_data is not None:
            data.update({"transform": self.parse_transforms(transforms_data)})

        quantization_dataloader = data.get("quantization", {}).get("dataloader", None)
        if quantization_dataloader and isinstance(quantization_dataloader, dict):
            data["quantization"].update(
                {"dataloader": self.parse_dataloader(quantization_dataloader)},
            )

        evaluation_dataloader = data.get("evaluation", {}).get("dataloader", None)
        if evaluation_dataloader and isinstance(evaluation_dataloader, dict):
            data["evaluation"].update(
                {"dataloader": self.parse_dataloader(evaluation_dataloader)},
            )

        num_cores = HWInfo().cores
        cores_per_instance = int(data.get("evaluation", {}).get("cores_per_instance", 4))

        if cores_per_instance < 1:
            raise ClientErrorException(
                "At least one core per instance must be used.",
            )
        if cores_per_instance > num_cores:
            raise ClientErrorException(
                f"Requested {cores_per_instance} cores per instance, "
                f"while only {num_cores} available.",
            )

        max_number_of_instances = num_cores // cores_per_instance
        instances = int(data.get("evaluation", {}).get("instances", max_number_of_instances))

        if instances < 1:
            raise ClientErrorException("At least one instance must be used.")

        if instances > max_number_of_instances:
            raise ClientErrorException(
                f"Attempted to use {instances} instances, "
                f"while only {max_number_of_instances} allowed.",
            )

        if "evaluation" in data:
            data["evaluation"].update(
                {
                    "cores_per_instance": cores_per_instance,
                    "num_of_instance": instances,
                    "batch_size": int(data.get("evaluation", {}).get("batch_size", 1)),
                },
            )

        data["tuning"] = parse_bool_value(data["tuning"])

        return data

    def parse_transforms(self, transforms_data: List[dict]) -> List[dict]:
        """Parse transforms list."""
        for transform in transforms_data:
            params_to_parse = transform.get("params", None)
            if isinstance(params_to_parse, dict):
                for param_name, value in params_to_parse.items():
                    param_type: Union[Type, List[Type]] = self.get_param_type(
                        "transform",
                        param_name,
                    )
                    if transform.get("name") == "RandomResizedCrop" and param_name == "scale":
                        param_type = [float]
                    transform["params"].update(
                        {param_name: self.parse_value(value, param_type)},
                    )
        return transforms_data

    def parse_dataloader(self, dataloader_data: dict) -> dict:
        """Parse dataloader dict."""
        dataloader_params = dataloader_data.get("params", None)
        if isinstance(dataloader_params, dict):
            for param_name, value in dataloader_params.items():
                param_type: Union[Type, List[Type]] = self.get_param_type(
                    "dataloader",
                    param_name,
                )
                dataloader_data["params"].update(
                    {param_name: self.parse_value(value, param_type)},
                )
        return dataloader_data

    def get_param_type(
        self,
        param_group: str,
        param_name: str,
    ) -> Union[Type, List[Type]]:
        """Get parameter type."""
        params_definitions = {}
        if param_group == "transform":
            params_definitions = self.transform_types
        elif param_group == "dataloader":
            params_definitions = self.dataloader_types
        for param_type, param_names in params_definitions.items():
            if param_name in param_names:
                found_type = self.types_definitions.get(param_type, None)
                if found_type is not None:
                    return found_type
        raise Exception(
            f"Could not found type for {param_group} {param_name} parameter.",
        )

    @staticmethod
    def parse_value(value: Any, required_type: Union[Type, List[Type]]) -> Any:
        """Parse value to required type."""
        try:
            if required_type == bool:
                return parse_bool_value(value)
            if callable(required_type):
                return required_type(value)
            elif isinstance(required_type, list):
                return parse_list_value(value, required_type[0])
        except ValueError as err:
            raise ClientErrorException(f"Cannot cast {value}. {str(err)}")
        return value


def parse_list_value(value: Any, required_type: Union[Type, List[Type]]) -> List[Any]:
    """Parse value to list."""
    if isinstance(required_type, list):
        return parse_multidim_list(value, required_type[0])
    if isinstance(value, str):
        return [required_type(element.strip("")) for element in value.strip("[]").split(",")]
    elif isinstance(value, Iterable):
        return [required_type(item) for item in value]
    elif callable(required_type):
        return [required_type(value)]
    else:
        return [value]


def parse_multidim_list(value: Any, required_type: Type) -> List[Union[Any, List[Any]]]:
    """Parse multi dimensional list."""
    if isinstance(value, str):
        value = normalize_string_list(value)
        parsed_list = json.loads(value)
    else:
        parsed_list = value

    if callable(required_type):
        for top_idx, top_element in enumerate(parsed_list):
            if isinstance(top_element, list):
                for idx, element in enumerate(top_element):
                    parsed_list[top_idx][idx] = required_type(element)
            else:
                parsed_list[top_idx] = required_type(top_element)
    return parsed_list


def normalize_string_list(string_list: str) -> str:
    """Add wrap string list into brackets if missing."""
    if not isinstance(string_list, str):
        return string_list
    if not string_list.startswith("["):
        string_list = "[" + string_list
    if not string_list.endswith("]"):
        string_list += "]"
    return string_list


def set_defaults(data: dict) -> dict:
    """Set default values for data if missing."""
    # Set tuning as default
    if "tuning" not in data:
        data.update({"tuning": True})

    # Set int8 as default requested precision
    if "precision" not in data:
        data.update({"precision": "int8"})

    return data
