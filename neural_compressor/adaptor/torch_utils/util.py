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
import torch

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

