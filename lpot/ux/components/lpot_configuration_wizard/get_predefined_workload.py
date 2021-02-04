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

from typing import Any, Dict, List, Union


def get_predefined_workload(
    data: Dict[str, Any],
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get configuration."""
    from lpot.ux.utils.workload.workload import Workload

    lpot_workload = Workload(data)
    lpot_workload.dump()
    lpot_workload.config.dump(lpot_workload.config_path)
    return lpot_workload.serialize()
