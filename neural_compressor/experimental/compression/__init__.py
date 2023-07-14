"""Experimental pruning API."""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

from os.path import dirname, basename, isfile, join
import glob
from .pruning import PRUNINGS
from .utils import torch
from typing import Optional
# modules = glob.glob(join(dirname(__file__), "*.py"))

# for f in modules:
#     if isfile(f) and not f.startswith('__') and not f.endswith('__init__.py'):
#         __import__(basename(f)[:-3], globals(), locals(), level=1)

FRAMEWORK = {
    'pytorch': 'pt',
    'keras': 'keras'
}

def get_pruning(config, model: torch.nn.Module, opt: Optional[torch.optim.Optimizer] = None,
                dataloader: Optional[torch.utils.data.DataLoader] = None,
                loss_func=None, framework='pytorch'):
    """Get registered pruner class.

    Get a Pruner object from PRUNINGS.

    Args:
        modules: A dict {"module_name": Tensor} that stores the pruning modules' weights.
        config: A config dict object that contains the pruner information.

    Returns:
        A Pruner object.

    Raises: AssertionError: Cuurently only support prunings that have been registered in PRUNINGS.
    """
    # do the ugly work here
    # check if it is doing self-multihead-attention pruning
    assert framework in FRAMEWORK.keys(), f"does not support {framework}, currently only support framework: {FRAMEWORK.keys()}"

    # if enable progressive pruning or not.
    if dataloader is None:
        assert opt is not None, f"Basic pruning type requires passing the 'opt' formal parameter."
        return PRUNINGS['basic_pruning'](config, model, opt)
    else:
        # assert 'gpt' in config["pruning_type"], f"Only sparseGPT pruning type are "
        return PRUNINGS['sparse_gpt_pruning'](config, model, dataloader, loss_func)

