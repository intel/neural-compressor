#!/usr/bin/env python
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

# Right now, we have upstreamed several int8 models to the huggingface model hub.
# For usage, just replace AutoModel class of transformers with our OptimizedModel

# 1. Intel/bert-base-uncased-mrpc-int8-qat 
# (https://huggingface.co/Intel/bert-base-uncased-mrpc-int8-qat)

# 2. Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static 
# (https://huggingface.co/Intel/distilbert-base-uncased-finetuned-sst-2-english-int8-static)


from neural_compressor.utils.load_huggingface import OptimizedModel

model_name_or_path = 'Intel/bert-base-uncased-mrpc-int8-qat'

model = OptimizedModel.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=None,
            cache_dir=None,
            revision=None,
            use_auth_token=None,
        )
