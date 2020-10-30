#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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

from ..adaptor.pytorch import _cfg_to_qconfig, _propagate_qconfig
from . import logger
import torch
from torch.quantization import add_observer_, convert
import yaml
import os
import copy

def load(tune_cfg_file, weights_file, model):
    """Execute the quantize process on the specified model.

    Args:
        tune_cfg_file (file): the tune configure file.
        model (object): fp32 model need to do quantization.

    Returns:
        (object): quantized model
    """

    assert os.path.exists(os.path.expanduser(tune_cfg_file)), \
           "tune configure file %s didn't exist" % tune_cfg_file
    assert os.path.exists(os.path.expanduser(weights_file)), \
           "weight file %s didn't exist" % weights_file

    q_model = copy.deepcopy(model.eval())

    with open(os.path.expanduser(tune_cfg_file), 'r') as f:
        tune_cfg = yaml.load(f, Loader=yaml.UnsafeLoader)

    op_cfgs = _cfg_to_qconfig(tune_cfg)
    _propagate_qconfig(q_model, op_cfgs)
    # sanity check common API misusage
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in q_model.modules()):
        logger.warn("None of the submodule got qconfig applied. Make sure you "
                    "passed correct configuration through `qconfig_dict` or "
                    "by assigning the `.qconfig` attribute directly on submodules")
    add_observer_(q_model)
    q_model = convert(q_model, inplace=True)
    weights = torch.load(os.path.expanduser(weights_file))
    q_model.load_state_dict(weights)
    return q_model
