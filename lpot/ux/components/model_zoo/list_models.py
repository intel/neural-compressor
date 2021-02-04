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
"""Get available models from Model Zoo."""

import logging as log
from typing import Any, Dict, List

from lpot.ux.utils.utils import (
    get_model_zoo_config_path,
    get_module_version,
    load_model_config,
)

log.basicConfig(level=log.DEBUG)

SUPPORTED_FRAMEWORKS = ["tensorflow"]


def list_models(data: dict) -> List[Dict[str, Any]]:
    """Process download model request."""
    model_list = get_available_models()
    return model_list


def get_available_models() -> List[Dict[str, Any]]:
    """Get available models from Model Zoo."""
    model_list = []
    full_list = load_model_config()
    for framework, framework_dict in full_list.items():
        if framework not in SUPPORTED_FRAMEWORKS:
            continue
        framework_version = get_module_version(framework)
        log.debug(f"{framework} version is {framework_version}")

        for domain, domain_dict in framework_dict.items():
            for model, model_dict in domain_dict.items():
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
                                framework,
                                domain,
                                model_dict,
                            ),
                            "model_path": "",
                        },
                    )
    return model_list


def check_version(framework_version: str, supported_versions: List[str]) -> bool:
    """Check if framework version is in supported versions list."""
    return any(framework_version.startswith(version) for version in supported_versions)
