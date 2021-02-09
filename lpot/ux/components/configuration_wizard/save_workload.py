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

from lpot.ux.utils.templates.workdir import Workdir
from lpot.ux.utils.workload.config import Config

logging.basicConfig(level=logging.INFO)


def save_workload(
    data: Dict[str, Any],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get configuration."""
    from lpot.ux.utils.workload.workload import Workload

    workdir = Workdir(
        workspace_path=data["workspace_path"],
        request_id=data["id"],
        model_path=data["model_path"],
    )

    workload = Workload(data)
    workload.dump()
    update_config(workload.config, data, workdir)
    workload.config.dump(os.path.join(workdir.workload_path, workload.config_name))
    return workload.serialize()


def update_config(config: Config, data: dict, workdir: Workdir) -> None:
    """Update config data from request."""
    is_custom_dataloader = data.get("dataloader", {}).get("name") == "custom"
    is_custom_metric = data.get("metric", {}).get("name") == "custom"

    map_key_to_action: Dict[str, Any] = {
        "workspace_path": config.set_workspace,
        "dataset_path": config.set_dataset_path,
        "transform": config.set_transform,
        "model_path": config.set_model_path,
        "inputs": config.set_inputs,
        "outputs": config.set_outputs,
    }

    for key, action in map_key_to_action.items():
        if data.get(key, None):
            action(data[key])
    if data.get("evaluation", None):
        update_evaluation_data(config, data["evaluation"])
    if data.get("quantization", None):
        update_quantization_data(config, data["quantization"])
    if is_custom_metric and is_custom_dataloader:
        generate_template(workdir, "dataloader_and_metric")
    elif is_custom_dataloader:
        generate_template(workdir, "dataloader")
    elif is_custom_metric:
        generate_template(workdir, "metric")


def generate_template(workdir: Workdir, type: str) -> None:
    """Generate code templates."""
    generated_template_path = os.path.join(workdir.workload_path, "code_template.py")
    path_to_templates = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "utils",
        "templates",
        f"{type}_template.txt",
    )
    copy(path_to_templates, generated_template_path)
    workdir.set_code_template_path(generated_template_path)


def update_evaluation_data(config: Config, evaluation_data: Dict[str, Any]) -> None:
    """Update config with evaluation data."""
    config.set_accuracy_metric(evaluation_data)
    if evaluation_data.get("warmup"):
        config.set_performance_warmup(evaluation_data["warmup"])
    if evaluation_data.get("iterations"):
        config.set_performance_iterations(evaluation_data["iterations"])


def update_quantization_data(config: Config, quantization_data: Dict[str, Any]) -> None:
    """Update config with quantization data."""
    config.set_accuracy_goal(quantization_data["accuracy_goal"])
    if quantization_data.get("strategy"):
        config.tuning.strategy.name = quantization_data["strategy"]
    if quantization_data.get("timeout"):
        config.tuning.set_timeout(quantization_data["timeout"])
    if quantization_data.get("max_trials"):
        config.tuning.set_max_trials(quantization_data["max_trials"])
    if quantization_data.get("objective"):
        config.tuning.objective = quantization_data["objective"]
    if quantization_data.get("random_seed"):
        config.tuning.random_seed = quantization_data["random_seed"]
    if quantization_data.get("approach"):
        config.set_quantization_approach(quantization_data["approach"])
    if quantization_data.get("sampling_size"):
        config.set_quantization_sampling_size(quantization_data["sampling_size"])
