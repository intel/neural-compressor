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

import os
from pathlib import Path
from typing import Any, Dict, List, Union

from lpot.ux.utils.exceptions import ClientErrorException


def get_predefined_configuration(
    data: Dict[str, Any],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get configuration."""
    from lpot.ux.utils.utils import get_framework_from_path, get_predefined_config_path
    from lpot.ux.utils.workload.config import Config

    model_path = data.get("model_path", "")
    if not os.path.isfile(model_path):
        raise ClientErrorException(
            f"Could not find model in specified path: {model_path}.",
        )

    model_name = Path(model_path).stem

    domain = data.get("domain", None)

    if not domain:
        raise ClientErrorException("Domain is not defined!")

    framework = get_framework_from_path(model_path)
    if framework is None:
        raise ClientErrorException(
            f"Could not find framework for specified model {model_name} in path {model_path}.",
        )

    config = Config()
    predefined_config_path = get_predefined_config_path(framework, domain)
    config.load(predefined_config_path)

    return {
        "config": config.serialize(),
        "framework": framework,
        "name": model_name,
        "domain": domain,
    }
