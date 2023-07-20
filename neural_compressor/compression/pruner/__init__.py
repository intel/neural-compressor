"""prune init."""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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

# model slim related
from .model_slim.auto_slim import parse_auto_slim_config
from .model_slim.auto_slim import model_slim
from .pruning import PRUNINGS
from neural_compressor.compression.pruner.utils import process_config, torch, logger
from typing import Optional, Union

FRAMEWORK = {
    'pytorch': 'pt',
    'keras': 'keras'
}

def prepare_pruning(config, model: torch.nn.Module,
                    opt_or_dataloader: Union[torch.optim.Optimizer, torch.utils.data.DataLoader],
                    loss_func=None, framework='pytorch', device: str ='cuda:0'):
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
    pruning_list = []
    pruning_conf = process_config(config)
    # if enable progressive pruning or not.
    if isinstance(opt_or_dataloader, torch.optim.Optimizer):
        for pruner_info in pruning_conf:
            assert 'gpt' not in pruner_info["pruning_type"] or 'retrain' not in pruner_info["pruning_type"], \
                    f"Basic pruning only supports strategies that can be sparse during training/fine-tuning"
        return PRUNINGS['basic_pruning'](pruning_conf, model, opt_or_dataloader)
    else:
        sparse_gpt_conf = []
        retrain_free_conf = []
        for pruner_info in pruning_conf:
            assert 'gpt' in pruner_info["pruning_type"] or 'retrain' in pruner_info["pruning_type"], \
                    f"Two post training methods are currently supported: sparse_gpt and retrain_free."
            if 'gpt' in pruner_info["pruning_type"]:
                sparse_gpt_conf.append(pruner_info)
            else:
                retrain_free_conf.append(pruner_info)
        if len(sparse_gpt_conf) > 0:
            pruning_list.append(PRUNINGS['sparse_gpt_pruning'](sparse_gpt_conf, model, opt_or_dataloader, loss_func, device))
        if len(retrain_free_conf) > 0:
            pruning_list.append(PRUNINGS['retrain_free_pruning'](retrain_free_conf, model, opt_or_dataloader, loss_func))
        if len(pruning_list) > 1:
            logger.info(f"Note that two post-training pruning methods are used here.")
            return pruning_list
        return pruning_list[0]


