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

from typing import Any, Dict, List

from neural_compressor.ux.components.model.repository import ModelRepository
from neural_compressor.ux.utils.consts import Frameworks
from neural_compressor.ux.utils.exceptions import ClientErrorException, InternalException
from neural_compressor.ux.utils.logger import log
from neural_compressor.ux.utils.utils import get_module_version, load_model_config


def list_models(data: dict) -> List[Dict[str, Any]]:
    """Process download model request."""
    model_list = get_available_models()
    return model_list


def get_available_models() -> List[Dict[str, Any]]:
    """Get available models from Examples."""
    model_list = []
    full_list = load_model_config()
    installed_frameworks = get_installed_frameworks()
    for framework, framework_version in installed_frameworks.items():
        framework_dict = full_list.get(framework, {})

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
            "Please install TensorFlow in one of following versions: "
            "2.0.x or 2.3.x - 2.10.x ",
        )


def get_framework_module_name(framework_name: str) -> str:
    """Get name of python module."""
    modules_map: Dict[str, str] = {
        Frameworks.TF.value: "tensorflow",
        Frameworks.ONNX.value: "onnx",
        Frameworks.PT.value: "torch",
    }
    module_name = modules_map.get(framework_name, None)
    if module_name is None:
        raise InternalException(
            f"Could not find framework module name. Framework {framework_name} not recognized.",
        )
    return module_name


def get_installed_frameworks() -> dict:
    """Check environment for installed frameworks."""
    installed_frameworks = {}
    supported_frameworks = ModelRepository.get_supported_frameworks()
    for framework in supported_frameworks:
        try:
            framework_module_name = get_framework_module_name(framework)
            framework_version = get_module_version(framework_module_name)
        except ClientErrorException:
            log.debug(f"Framework {framework} not installed.")
            continue
        log.debug(f"{framework} version is {framework_version}")
        installed_frameworks.update({framework: framework_version})
    return installed_frameworks
