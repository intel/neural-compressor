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
from typing import Any, Dict, List, Union

from lpot.ux.components.configuration_wizard.configuration_parser import ConfigurationParser
from lpot.ux.utils.templates.workdir import Workdir
from lpot.ux.utils.utils import replace_with_values
from lpot.ux.utils.workload.config import Config
from lpot.ux.utils.workload.workload import Workload

logging.basicConfig(level=logging.INFO)


def save_workload(
    data: Dict[str, Any],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get configuration."""
    parser = ConfigurationParser()
    parsed_data = parser.parse(data)

    workload = Workload(parsed_data)
    workload.dump()

    workdir = Workdir(
        workspace_path=data["workspace_path"],
        request_id=data["id"],
        model_path=data["model_path"],
        input_precision=workload.input_precision,
        output_precision=workload.output_precision,
        mode=workload.mode,
    )

    update_config(workload, parsed_data, workdir)
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
        "batch_size": config.set_performance_batch_size,
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
    }

    for key, action in map_key_to_action.items():
        if quantization_data.get(key, None):
            action(quantization_data[key])

    if quantization_data.get("strategy"):
        config.tuning.strategy.name = str(quantization_data["strategy"])
    if quantization_data.get("objective"):
        config.tuning.objective = str(quantization_data["objective"])
