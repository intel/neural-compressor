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
"""Configuration to yaml."""

import logging
import os
from shutil import copy
from typing import Any, Dict, List, Optional, Tuple, Union

from neural_compressor.ux.components.configuration_wizard.configuration_parser import ConfigurationParser
from neural_compressor.ux.utils.exceptions import NotFoundException
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.utils.utils import replace_with_values
from neural_compressor.ux.utils.workload.config import Config
from neural_compressor.ux.utils.workload.dataloader import Dataset, Transform
from neural_compressor.ux.utils.workload.workload import Workload

logging.basicConfig(level=logging.INFO)


def change_performance_dataloader_to_dummy_if_possible(
    model_domain: str,
    config: Config,
) -> None:
    """Change config.evaluation.performance.dataloader.dataset to Dummy_v2."""
    if model_domain not in ["image_recognition", "object_detection"]:
        return

    if not (
        config.evaluation
        and config.evaluation.performance
        and config.evaluation.performance.dataloader
        and config.evaluation.performance.dataloader.dataset
    ):
        return

    if not (
        config.quantization
        and config.quantization.calibration
        and config.quantization.calibration.dataloader
        and config.quantization.calibration.dataloader.transform
    ):
        return

    try:
        transforms = [
            value for _, value in config.quantization.calibration.dataloader.transform.items()
        ]
        shape = get_shape_from_transforms(transforms)
        config.evaluation.performance.dataloader.transform.clear()
        config.evaluation.performance.dataloader.dataset = Dataset(
            "dummy_v2",
            {
                "input_shape": shape,
                "label_shape": [1],
            },
        )
    except (NotFoundException, ValueError):
        pass


def get_shape_from_transforms(transforms: List[Transform]) -> list:
    """Detect dataset sizes based on configured transforms."""
    shapes = {
        "channels": 3,
        "height": None,
        "width": None,
    }

    shape_elements_order = ["height", "width", "channels"]

    for transform in transforms:
        name = transform.name
        parameters = transform.parameters
        if name in [
            "Resize",
            "CenterCrop",
            "RandomResizedCrop",
            "RandomCrop",
            "CropResize",
        ]:
            shapes["height"], shapes["width"] = get_height_width_from_size(parameters.get("size"))
        elif "Transpose" == name:
            axes = parameters.get("perm")
            if not axes:
                raise ValueError("Unknown value of 'perm' argument in Transpose")
            shape_elements_order = [shape_elements_order[idx] for idx in axes]
        elif "CropToBoundingBox" == name:
            shapes["height"] = parameters.get("target_height")
            shapes["width"] = parameters.get("target_width")
        elif name in [
            "ResizeCropImagenet",
            "BilinearImagenet",
        ]:
            shapes["height"] = parameters.get("height")
            shapes["width"] = parameters.get("width")

    if not shapes["height"] or not shapes["width"]:
        raise NotFoundException("Unable to detect shape for Dummy dataset")

    return [shapes.get(dimension) for dimension in shape_elements_order]


def get_height_width_from_size(size: Any) -> Tuple[Optional[int], Optional[int]]:
    """Detect dataset sizes based on size param common in some Transforms."""
    if isinstance(size, int):
        return size, size
    elif isinstance(size, list):
        if len(size) == 1:
            return size[0], size[0]
        elif len(size) == 2:
            return size[0], size[1]
    return None, None


def save_workload(
    data: Dict[str, Any],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get configuration."""
    model_domain = data.get("domain", "")

    parser = ConfigurationParser()
    parsed_data = parser.parse(data)

    workload = Workload(parsed_data)
    workload.dump()

    workdir = Workdir(
        request_id=workload.id,
        project_name=workload.project_name,
        model_path=workload.model_path,
        input_precision=workload.input_precision,
        output_precision=workload.output_precision,
        mode=workload.mode,
        created_at=workload.created_at,
    )

    update_config(workload, parsed_data, workdir)
    change_performance_dataloader_to_dummy_if_possible(model_domain, workload.config)
    workload.config.dump(os.path.join(workdir.workload_path, workload.config_name))
    return workload.serialize()


def check_custom_dataloader(data: dict) -> bool:
    """Check if custom dataloader is used in calibration or evaluation."""
    return any(
        [
            data.get("quantization", {}).get("dataloader", {}).get("name") == "custom",
            data.get("evaluation", {}).get("dataloader", {}).get("name") == "custom",
        ],
    )


def update_config(workload: Workload, data: dict, workdir: Workdir) -> None:
    """Update config data from request."""
    config = workload.config
    is_custom_dataloader = check_custom_dataloader(data)
    is_custom_metric = data.get("evaluation", {}).get("metric") == "custom"

    map_key_to_action: Dict[str, Any] = {
        "transform": config.set_transform,
        "model_path": config.set_model_path,
        "inputs": config.set_inputs,
        "outputs": config.set_outputs,
    }

    for key, action in map_key_to_action.items():
        if data.get(key, None) is not None:
            action(data[key])
    if data.get("evaluation", None):
        update_evaluation_data(config, data["evaluation"])
    if data.get("quantization", None):
        update_quantization_data(config, data["quantization"])
    if is_custom_metric and is_custom_dataloader:
        generate_template(workload, workdir, "dataloader_and_metric")
        config.remove_dataloader()
        config.remove_accuracy_metric()
    elif is_custom_dataloader:
        generate_template(workload, workdir, "dataloader")
        config.remove_dataloader()
    elif is_custom_metric:
        generate_template(workload, workdir, "metric")
        config.remove_accuracy_metric()


def generate_template(workload: Workload, workdir: Workdir, type: str) -> None:
    """Generate code templates."""
    correct_paths = {
        "config_path": workload.config_path,
        "model_path": workload.model_path,
        "model_output_path": workload.model_output_path,
    }

    generated_template_path = os.path.join(workload.workload_path, "code_template.py")
    path_to_templates = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "utils",
        "templates",
        f"{type}_template.txt",
    )
    copy(path_to_templates, generated_template_path)
    replace_with_values(correct_paths, generated_template_path)
    workdir.set_code_template_path(generated_template_path)


def update_evaluation_data(config: Config, evaluation_data: Dict[str, Any]) -> None:
    """Update config with evaluation data."""
    config.set_accuracy_metric(evaluation_data)

    map_key_to_action: Dict[str, Any] = {
        "warmup": config.set_performance_warmup,
        "iterations": config.set_performance_iterations,
        "dataloader": config.set_evaluation_dataloader,
        "dataset_path": config.set_evaluation_dataset_path,
        "cores_per_instance": config.set_performance_cores_per_instance,
        "num_of_instance": config.set_performance_num_of_instance,
        "batch_size": config.set_accuracy_and_performance_batch_sizes,
    }

    for key, action in map_key_to_action.items():
        if evaluation_data.get(key, None):
            action(evaluation_data[key])


def update_quantization_data(config: Config, quantization_data: Dict[str, Any]) -> None:
    """Update config with quantization data."""
    map_key_to_action: Dict[str, Any] = {
        "accuracy_goal": config.set_accuracy_goal,
        "timeout": config.tuning.set_timeout,
        "max_trials": config.tuning.set_max_trials,
        "random_seed": config.tuning.set_random_seed,
        "approach": config.set_quantization_approach,
        "sampling_size": config.set_quantization_sampling_size,
        "dataloader": config.set_quantization_dataloader,
        "dataset_path": config.set_quantization_dataset_path,
        "batch_size": config.set_quantization_batch_size,
    }

    for key, action in map_key_to_action.items():
        if quantization_data.get(key, None):
            action(quantization_data[key])

    if quantization_data.get("strategy"):
        config.tuning.strategy.name = str(quantization_data["strategy"])
    if quantization_data.get("objective"):
        config.tuning.objective = str(quantization_data["objective"])
