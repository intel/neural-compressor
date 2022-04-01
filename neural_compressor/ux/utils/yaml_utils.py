# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022 Intel Corporation
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

from typing import Any

from neural_compressor.conf.config import Pruner


def float_representer(dumper: Any, value: float) -> Any:
    """Float representer for yaml dumper."""
    if value.is_integer():
        float_out = f"{int(value)}.0"
    else:
        float_out = f"{value}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", float_out)


def pruner_representer(dumper: Any, pruner: Pruner) -> Any:
    """Pruner representer for yaml dumper."""
    pruner_data = {
        "start_epoch": pruner.start_epoch,
        "end_epoch": pruner.end_epoch,
        "update_frequency": pruner.update_frequency,
        "target_sparsity": pruner.target_sparsity,
        "initial_sparsity": pruner.initial_sparsity,
        "prune_type": pruner.prune_type,
        "method": pruner.method,
        "names": pruner.names,
        "parameters": pruner.parameters,
    }
    return dumper.represent_mapping(
        "!Pruner",
        pruner_data,
    )
