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
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_compressor.common import logger

# model_id = "/models/opt-125m"
model_id = "/models/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()
input_str = "Hello HQQ.."
encoded_input = tokenizer(input_str, return_tensors="pt")
float_out = model(**encoded_input)
logger.info(float_out)


from neural_compressor.torch.algorithms.weight_only.hqq.quantizer import _hqq_entry, get_default_hqq_config_mapping

default_hqq_config_mapping = get_default_hqq_config_mapping(model)
default_hqq_config_mapping.pop("lm_head")
logger.info(default_hqq_config_mapping)


q_model = _hqq_entry(model, default_hqq_config_mapping)
encoded_input.to("cuda:0")
out_qdq = q_model(**encoded_input)
logger.info(out_qdq)

from .eval_wiki2 import eval_wikitext2

eval_wikitext2(q_model, tokenizer, verbose=True)
