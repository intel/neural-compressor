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

from pathlib import Path
from typing import Any, Dict, List, Union

from neural_compressor.ux.components.model.repository import ModelRepository
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.hw_info import HWInfo
from neural_compressor.ux.utils.workload.config import Config


def get_predefined_configuration(
    data: Dict[str, Any],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get configuration."""
    from neural_compressor.ux.utils.utils import get_framework_from_path, get_predefined_config_path

    model_path = data.get("model_path", "")
    if not ModelRepository.is_model_path(model_path):
        raise ClientErrorException(
            f"Could not find model in specified path: {model_path}.",
        )

    model_name = Path(model_path).stem

    domain = data.get("domain", None)
    domain_flavour = data.get("domain_flavour", "")

    if not domain:
        raise ClientErrorException("Domain is not defined!")

    framework = get_framework_from_path(model_path)
    if framework is None:
        raise ClientErrorException(
            f"Could not find framework for specified model {model_name} in path {model_path}.",
        )

    config = Config()
    predefined_config_path = get_predefined_config_path(framework, domain, domain_flavour)
    config.load(predefined_config_path)

    update_config_to_machine_specification(config)

    return {
        "config": config.serialize(),
        "framework": framework,
        "name": model_name,
        "domain": domain,
    }


def update_config_to_machine_specification(config: Config) -> None:
    """Change Config values based on local machine."""
    hwinfo = HWInfo()

    cores_per_socket = hwinfo.cores_per_socket

    if not config.evaluation:
        return

    if config.evaluation.accuracy and config.evaluation.accuracy.dataloader:
        config.evaluation.accuracy.dataloader.batch_size = cores_per_socket

    if config.evaluation.performance:
        config.evaluation.performance.configs.cores_per_instance = 4
        config.evaluation.performance.configs.num_of_instance = cores_per_socket // 4

        if config.evaluation.performance.dataloader:
            config.evaluation.performance.dataloader.batch_size = 1
