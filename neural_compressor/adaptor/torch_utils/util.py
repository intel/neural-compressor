#
# -*- coding: utf-8 -*-
#
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

import numpy as np
import re
from ...utils.utility import LazyImport

torch = LazyImport("torch")


def get_embedding_contiguous(model):
    """This is a helper function for nn.Embedding,
        and it will get input contiguous.

    Args:
        model (object): input model

    Returns:
        None
    """
    def contiguous_hook(module, input):
        embeddings = input[0].contiguous()
        modified_input = (embeddings, *input[1:])
        return modified_input

    for child in model.modules():
        child_type = child.__class__.__name__
        if child_type == 'Embedding':
            child.register_forward_pre_hook(contiguous_hook)

def collate_torch_preds(results):
    batch = results[0]
    if isinstance(batch, list):
        results = zip(*results)
        collate_results = []
        for output in results:
            output = [
                batch.numpy() if isinstance(batch, torch.Tensor) else batch
                for batch in output
            ]
            collate_results.append(np.concatenate(output))
    elif isinstance(batch, torch.Tensor):
        results = [
            batch.numpy() if isinstance(batch, torch.Tensor) else batch
            for batch in results
        ]
        collate_results = np.concatenate(results)
    return collate_results

def append_attr(fx_model, model):
    """a helper method to append attribution for the symbolic traced model.

    Args:
        fx_model(torch.fx.GraphModule): The symbolic traced model.
        model(torch.nn.Module): The original model.

    Returns:
        fx_model (dir): The symbolic traced model with additional attribution.
    """
    fx_attr = dir(fx_model)
    org_attr = dir(model)
    ignore_match_patterns = [r"_", r"quant", r"dequant", 
                            r'activation_post_process']
    ignore_search_patterns = [r"_scale_", r"_zero_point_", 
                            r'_activation_post_process_']
    attr_names = []
    for i in org_attr:
        if i not in fx_attr and \
          not any([re.match(p, i) for p in ignore_match_patterns]) and \
          not any([re.search(p, i) for p in ignore_search_patterns]) :
            attr_names.append(i)
    for name in attr_names:
        attr = getattr(model, name)
        setattr(fx_model, name, attr)
    return fx_model
