#!/usr/bin/env python
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

import torch
from typing import Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from optimum.intel.generation.modeling import TSModelForCausalLM
class TSModelCausalLMForOPTLLM(TSModelForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        model_type = self.config.model_type.replace("_", "-")
        if self.use_cache:
            if past_key_values is None:
                nb_pkv = 2
                num_layers = self.normalized_config.num_layers
                d_k = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
                batch_size = input_ids.shape[0]
                input_len = input_ids.shape[1]  
                num_attention_heads = self.normalized_config.num_attention_heads
                num_key_value_heads = num_attention_heads
                if hasattr(self.normalized_config, "num_key_value_heads"):
                    num_key_value_heads = self.normalized_config.num_key_value_heads
                if hasattr(self.normalized_config, "multi_query_group_num"):
                    num_key_value_heads = self.normalized_config.multi_query_group_num
                elif self.config.model_type == "qwen":
                    new_shape = [batch_size, 1, num_key_value_heads, d_k]
                elif self.config.model_type == "chatglm":
                    new_shape = [1, batch_size, num_key_value_heads, d_k]
                else:
                    new_shape = [batch_size, num_key_value_heads, 1, d_k]

                beam_idx_tmp = torch.zeros((2048, int(batch_size)), dtype=torch.long)
                past_key_values = [(torch.zeros(1, 0, 0, 1, dtype=torch.long).contiguous(),
                        torch.zeros(size=new_shape).contiguous(),
                        torch.zeros(size=new_shape).contiguous(),
                        beam_idx_tmp) for _ in range(num_layers)]
                past_key_values = tuple(past_key_values)
            inputs["past_key_values"] = past_key_values
        if model_type != "opt":
            if position_ids is not None:
                inputs["position_ids"] = position_ids
            else:
                inputs["position_ids"] = torch.arange(input_len).repeat(batch_size, 1)

        outputs = self.model(**inputs)

        if isinstance(outputs, (list, tuple)):
            logits = outputs[0]
            past_key_values = outputs[1] if self.use_cache else None
        else:
            logits = outputs["logits"]
            past_key_values = outputs["past_key_values"] if self.use_cache else None
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)
