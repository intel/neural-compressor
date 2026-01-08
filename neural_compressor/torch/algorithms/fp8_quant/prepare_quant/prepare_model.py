# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
from typing import Optional



_local_rank = -1
_world_size = -1
_world_size_initialized = False
_local_rank_initialized = False

def get_world_size():
    global _world_size, _world_size_initialized
    if not _world_size_initialized:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _world_size = torch.distributed.get_world_size()
            _world_size_initialized = True
        elif os.getenv('WORLD_SIZE', None) is not None:
            _world_size = os.getenv('WORLD_SIZE')
            _world_size_initialized = True
    return _world_size

def get_local_rank():
    global _local_rank, _local_rank_initialized
    if not _local_rank_initialized:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _local_rank = torch.distributed.get_rank()
            _local_rank_initialized = True
        elif os.getenv('LOCAL_RANK', None) is not None:
            _local_rank = os.getenv('LOCAL_RANK')
            _local_rank_initialized = True
    return _local_rank

def _prep_model_with_predefined_config(model, *, config):
    from .._core.utils import prepare_model
    from .._quant_common.quant_config import set_hqt_config

    set_hqt_config(model, config)
    prepare_model(model)


def prep_model(model, config_path: Optional[str] = None):
    """Prepare this model with the given (absolute or relative) path of the json file containing the configuration.

    If `config_path` is not given or `None`,
    instead perform the legacy behavior of checking for env variable `QUANT_CONFIG`.
    """
    from .._quant_common.quant_config import Fp8cfg, _read_config_from_file

    if config_path is None:
        config_path = os.getenv("QUANT_CONFIG")
        if config_path is None:
            raise EnvironmentError(
                "Either pass config_path parameter explicitly (recommended), or set environment variable QUANT_CONFIG"
            )

    config = _read_config_from_file(config_path=config_path)
    config = Fp8cfg.parse(config)
    return _prep_model_with_predefined_config(model, config=config)


def finish_measurements(model):
    from .._core.save_measure import save_measurements

    save_measurements(model)
