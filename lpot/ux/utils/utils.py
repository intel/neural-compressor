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
"""UX server utils module."""

import json
import os
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Optional, Union

from lpot.ux.utils.exceptions import ClientErrorException

dataset_locations = {
    "tensorflow": {
        "image_recognition": {
            "name": "Imagenet",
            "path": "examples/test/dataset/imagenet",
        },
    },
}

model_domains = {
    "image_recognition": ["resnet50_v1_5"],
    "object_detection": ["ssd_mobilenet_v1"],
}

framework_extensions = {
    "tensorflow": ["pb"],
}


def is_hidden(path: str) -> bool:
    """Check if path is for hidden filesystem entry."""
    return "." == os.path.basename(path)[0]


def get_model_domain(model: str) -> str:
    """
    Get model domain.

    :param model: Model name.
    """
    for domain, models in model_domains.items():
        if model in models:
            return domain
    raise Exception(f"Could not found domain of {model} model.")


def get_dataset_path(framework: str, domain: str) -> str:
    """Get dataset path for specified framework and domain."""
    dataset = dataset_locations.get(framework, {}).get(domain, {})
    if dataset is None:
        raise Exception("Could not found dataset location.")
    if dataset.get("path") is None:
        raise Exception("Could not found dataset location.")
    return dataset["path"]


def get_framework_from_path(model_path: str) -> Optional[str]:
    """
    Get framework name from model extension.

    :param model_path: Path to model.
    """
    extension = get_file_extension(model_path)
    for framework, extensions in framework_extensions.items():
        if extension in extensions:
            return framework
    return None


def get_file_extension(path: str) -> str:
    """Get file extension without leading dot."""
    return os.path.splitext(path)[1][1:]


def is_model_file(path: str) -> bool:
    """Check if given path is a model of supported framework."""
    return get_framework_from_path(path) is not None


def get_predefined_config_path(framework: str, domain: str) -> str:
    """Get predefined config for specified model domain."""
    path = [os.environ["LPOT_REPOSITORY_PATH"], "examples", framework, domain]
    config_map = {
        "image_recognition": ["resnet50_v1_5.yaml"],
        "object_detection": ["ssd_mobilenet_v1.yaml"],
        "recommendation": ["wide_deep_large_ds", "wide_deep_large_ds.yaml"],
        "nlp": ["bert", "bert.yaml"],
    }

    config_path = config_map.get(domain, None)
    if config_path is None:
        raise Exception(f"Could not found config for {domain} domain.")

    path.extend(config_path)
    return os.path.join(*path)


def get_model_zoo_config_path(
    framework: str,
    domain: str,
    model_dict: Dict[str, Any],
) -> str:
    """Get predefined config for model from Model Zoo."""
    try:
        lpot_repository_path = os.environ["LPOT_REPOSITORY_PATH"]
    except KeyError:
        return ""

    yaml_relative_location = model_dict.get("yaml")
    if lpot_repository_path and yaml_relative_location:
        path = [
            lpot_repository_path,
            "examples",
            framework,
            domain,
            yaml_relative_location,
        ]
        return os.path.join(*path)
    return ""


def load_json(path: str) -> dict:
    """Load json file and convert it into dict.

    :param path: path to json file
    :type path: str
    :return: dict
    :rtype: dict
    """
    with open(path) as json_data:
        data = json.load(json_data)

    return data


def find_boundary_nodes(model_path: str) -> Dict[str, Any]:
    """Update model's input and output nodes in config file."""
    framework = get_framework_from_path(model_path)
    if framework is None:
        raise Exception("Could not found framework for specified model.")
    check_module(framework)
    # Inputs are only required for TF models
    if framework == "tensorflow":
        from lpot.adaptor.tf_utils.util import (
            get_graph_def,
            get_input_node_names,
            get_output_node_names,
        )

        graph_def = get_graph_def(model_path)
        return {
            "inputs": get_input_node_names(graph_def)[
                -1
            ],  # TODO: Remove passing last value
            "outputs": get_output_node_names(graph_def)[
                -1
            ],  # TODO: Remove passing last value
        }
    return {}


def check_module(module_name: str) -> None:
    """Check if module exists. Raise exception when not found."""
    module = find_spec(module_name)
    if module is None:
        raise Exception(f"Could not found {module_name} module.")


def get_module_version(module_name: str) -> str:
    """Check module version. Raise exception when not found."""
    check_module(module_name)
    module = import_module(module_name)
    version = getattr(module, "__version__")
    if version is None:
        raise ClientErrorException(f"Could not found version of {module_name} module.")
    return version


def get_size(path: str, unit: str = "MB", add_unit: bool = False) -> Union[str, int]:
    """Check file or directory size."""
    supported_units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
    }
    unit_modifier = supported_units.get(unit, None)
    if unit_modifier is None:
        raise Exception(
            "Unit not supported. Select one of following: "
            + str(supported_units.keys()),
        )
    root_dir = Path(path)
    if root_dir.is_file():
        size = root_dir.stat().st_size
    else:
        size = sum(f.stat().st_size for f in root_dir.glob("**/*") if f.is_file())
    size = int(round(size / unit_modifier))
    if add_unit:
        return f"{size}{unit}"
    return size


def load_model_config() -> Dict[str, Any]:
    """Load model configs from json."""
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "configs",
            "models.json",
        ),
        "r",
    ) as f:
        models_config = json.load(f)
    if isinstance(models_config, dict):
        return models_config
    return {}


def load_dataloader_config() -> Dict[str, Any]:
    """Load dataloader configs from json."""
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "configs",
            "dataloaders.json",
        ),
        "r",
    ) as f:
        dataloaders_config = json.load(f)
    if isinstance(dataloaders_config, dict):
        return dataloaders_config
    return {}


def load_transforms_config() -> Dict[str, Any]:
    """Load dataloader configs from json."""
    with open(
        os.path.join(
            os.path.dirname(__file__),
            "configs",
            "transforms.json",
        ),
        "r",
    ) as f:
        dataloaders_config = json.load(f)
    if isinstance(dataloaders_config, dict):
        return dataloaders_config
    return {}
