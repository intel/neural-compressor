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
"""Get available models from Examples."""

from typing import Any, Dict, List, Optional

from neural_compressor.ux.components.model.repository import ModelRepository
from neural_compressor.ux.utils.exceptions import ClientErrorException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.templates.workdir import Workdir
from neural_compressor.ux.utils.utils import (
    get_model_zoo_config_path,
    get_model_zoo_model_path,
    get_module_version,
    load_model_config,
)


def list_models(data: dict) -> List[Dict[str, Any]]:
    """Process download model request."""
    workspace_path = Workdir().get_active_workspace()
    model_list = get_available_models(workspace_path)
    return model_list


def get_available_models(workspace_path: Optional[str]) -> List[Dict[str, Any]]:
    """Get available models from Examples."""
    model_list = []
    full_list = load_model_config()

    supported_frameworks = ModelRepository.get_supported_frameworks()

    for framework in supported_frameworks:
        try:
            framework_version = get_module_version(framework)
        except Exception:
            log.debug(f"Framework {framework} not installed.")
            continue
        log.debug(f"{framework} version is {framework_version}")

        framework_dict = full_list[framework]
        for domain, domain_dict in framework_dict.items():
            if not isinstance(domain_dict, dict):
                continue
            for model, model_dict in domain_dict.items():
                if not isinstance(model_dict, dict):
                    continue
                if check_version(
                    framework_version,
                    model_dict.get("framework_version", []),
                ):
                    model_list.append(
                        {
                            "framework": framework,
                            "domain": domain,
                            "model": model,
                            "yaml": get_model_zoo_config_path(
                                workspace_path,
                                framework,
                                domain,
                                model,
                                model_dict,
                            ),
                            "model_path": get_model_zoo_model_path(
                                workspace_path,
                                framework,
                                domain,
                                model,
                                model_dict,
                            ),
                        },
                    )

    validate_model_list(model_list)
    return model_list


def check_version(framework_version: str, supported_versions: List[str]) -> bool:
    """Check if framework version is in supported versions list."""
    return any(framework_version.startswith(version) for version in supported_versions)


def validate_model_list(model_list: List[dict]) -> None:
    """Check if model list is valid."""
    if not model_list:
        raise ClientErrorException(
            "Examples require installed TensorFlow in specific version. "
            "Please install TensorFlow in one of following versions: 2.0.x, 2.3.x or 2.4.x.",
        )
