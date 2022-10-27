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
from copy import deepcopy
from typing import Any, Dict, List, Type, Union

from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.hw_info import HWInfo
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import parse_bool_value


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
                "offset_height",
                "offset_width",
                "target_height",
                "target_width",
                "dim",
                "resize_side",
                "label_shift",
            ],
            "float": ["scale", "central_fraction"],
            "list<float>": ["mean", "std", "mean_value", "std_value", "ratio"],
            "list<int>": ["perm", "size"],
            "bool": ["random_crop", "random_flip_left_right"],
        }
        self.dataloader_types: Dict[str, List[str]] = {
            "str": [
                "root",
                "filenames",
                "compression_type",
                "data_path",
                "data_dir",
                "image_list",
                "img_dir",
                "anno_dir",
                "content_folder",
                "content_path",
                "style_folder",
                "style_path",
                "image_format",
                "dtype",
                "label_file",
                "model_name_or_path",
                "task",
                "model_type",
            ],
            "int": ["buffer_size", "num_parallel_reads", "num_cores", "max_seq_length"],
            "list<int>": ["resize_shape"],
            "float": ["crop_ratio"],
            "list<float>": ["low", "high"],
            "list<list<int>>": ["shape", "input_shape", "label_shape"],
            "bool": ["train", "label", "do_lower_case", "dynamic_length"],
        }

        self.metric_types: Dict[str, List[str]] = {
            "str": ["anno_path", "task"],
            "int": ["num_detections", "boxes", "scores", "classes", "k"],
            "bool": ["compare_label"],
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

    def parse(self, input_data: dict) -> dict:
        """Parse configuration."""
        data = deepcopy(input_data)
        transforms_data = data.get("transform", None)
        if transforms_data is not None:
            data.update({"transform": self.parse_transforms(transforms_data)})

        quantization_dataloader = data.get("quantization", {}).get("dataloader", None)
        if quantization_dataloader and isinstance(quantization_dataloader, dict):
            data["quantization"].update(
                {"dataloader": self.parse_dataloader(quantization_dataloader)},
            )

        evaluation_data = data.get("evaluation", None)
        if evaluation_data and isinstance(evaluation_data, dict):
            self.parse_evaluation_data(evaluation_data)

        metric_params = data.get("metric_param", None)
        if metric_params and isinstance(metric_params, dict):
            parsed_metric_params = self.parse_metric(metric_params)

            data.update({"metric_param": parsed_metric_params})

        if "tuning" in data.keys():
            data["tuning"] = parse_bool_value(data["tuning"])

        return data

    def parse_evaluation_data(self, evaluation_data: dict) -> None:
        """Parse input evaluation data."""
        evaluation_dataloader = evaluation_data.get("dataloader", None)
        if evaluation_dataloader and isinstance(evaluation_dataloader, dict):
            evaluation_data.update(
                {"dataloader": self.parse_dataloader(evaluation_dataloader)},
            )
        metric_data = evaluation_data.get("metric_param", None)

        if metric_data and isinstance(metric_data, dict):
            parsed_metric_data = self.parse_metric(metric_data)
            evaluation_data.update(
                {"metric_param": parsed_metric_data},
            )

        num_cores = HWInfo().cores
        cores_per_instance = int(
            evaluation_data.get(
                "cores_per_instance",
                4,
            ),
        )

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
        instances = int(
            evaluation_data.get(
                "instances",
                max_number_of_instances,
            ),
        )

        if instances < 1:
            raise ClientErrorException("At least one instance must be used.")

        if instances > max_number_of_instances:
            raise ClientErrorException(
                f"Attempted to use {instances} instances, "
                f"while only {max_number_of_instances} allowed.",
            )

        evaluation_data.update(
            {
                "cores_per_instance": cores_per_instance,
                "num_of_instance": instances,
                "batch_size": int(evaluation_data.get("batch_size", 1)),
            },
        )

    def parse_transforms(self, transforms_data: List[dict]) -> List[dict]:
        """Parse transforms list."""
        parsed_transform_data: List[dict] = []
        for transform in transforms_data:
            parsed_transform_params: dict = {}
            params_to_parse = transform.get("params", None)
            if isinstance(params_to_parse, dict):
                for param_name, value in params_to_parse.items():
                    if value == "":
                        continue

                    param_type: Union[Type, List[Type]] = self.get_param_type(
                        "transform",
                        param_name,
                    )
                    if transform.get("name") == "RandomResizedCrop" and param_name == "scale":
                        param_type = [float]

                    parsed_transform_params.update(
                        {param_name: self.parse_value(value, param_type)},
                    )
            parsed_transform_data.append(
                {
                    "name": transform.get("name"),
                    "params": parsed_transform_params,
                },
            )
        return parsed_transform_data

    def parse_dataloader(self, dataloader_data: dict) -> dict:
        """Parse dataloader dict."""
        parsed_dataloader_data: dict = {"params": {}}
        dataloader_params = dataloader_data.get("params", None)
        if isinstance(dataloader_params, dict):
            for param_name, value in dataloader_params.items():
                if value == "":
                    continue
                param_type: Union[Type, List[Type]] = self.get_param_type(
                    "dataloader",
                    param_name,
                )
                parsed_dataloader_data["params"].update(
                    {param_name: self.parse_value(value, param_type)},
                )
        return parsed_dataloader_data

    def parse_metric(self, metric_data: dict) -> dict:
        """Parse metric data."""
        parsed_data = {}
        for param_name, param_value in metric_data.items():
            if isinstance(param_value, dict):
                parsed_data.update({param_name: self.parse_metric(param_value)})
            elif isinstance(param_value, str) or isinstance(param_value, int):
                if param_value == "":
                    continue
                param_type = self.get_param_type("metric", param_name)
                if param_type is None:
                    log.debug("Could not find param type.")
                    continue
                parsed_value = self.parse_value(param_value, param_type)
                parsed_data.update({param_name: parsed_value})
        return parsed_data

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
        elif param_group == "metric":
            params_definitions = self.metric_types
        for param_type, param_names in params_definitions.items():
            if param_name in param_names:
                found_type = self.types_definitions.get(param_type, None)
                if found_type is not None:
                    return found_type
        raise Exception(
            f"Could not found type for {param_group} {param_name} parameter.",
        )

    @staticmethod
    def parse_value(value: Any, required_type: Union[Type, List[Type], List[List[Type]]]) -> Any:
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


def parse_list_value(
    value: Any,
    required_type: Union[Type, List[Type], List[List[Type]]],
) -> List[Any]:
    """Parse value to list."""
    if isinstance(required_type, list):
        return parse_multidim_list(value, required_type)  # type: ignore
    if isinstance(value, str):
        return [required_type(element.strip("")) for element in value.strip("[]").split(",")]
    elif isinstance(value, Iterable):
        return [required_type(item) for item in value]
    elif callable(required_type):
        return [required_type(value)]
    else:
        return [value]


def parse_multidim_list(value: Any, required_type: List[Type]) -> List[Union[Any, List[Any]]]:
    """Parse multi dimensional list."""
    if isinstance(value, str):
        value = normalize_string_list(value, required_type)
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


def normalize_string_list(string_list: str, required_type: Union[Type, List[Type]]) -> str:
    """Add wrap string list into brackets if missing."""
    if not isinstance(string_list, str):
        return string_list
    if isinstance(required_type, list):
        string_list = string_list.replace("(", "[")
        string_list = string_list.replace(")", "]")
        while not string_list.startswith("[["):
            string_list = "[" + string_list
        while not string_list.endswith("]]"):
            string_list += "]"
        return string_list
    if not string_list.startswith("["):
        string_list = "[" + string_list
    if not string_list.endswith("]"):
        string_list += "]"
    return string_list
