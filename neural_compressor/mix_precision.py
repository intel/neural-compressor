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


from .experimental.mixed_precision import MixedPrecision
from neural_compressor.conf.pythonic_config import Config, MixedPrecisionConfig, Options

def fit(model, config=None, eval_func=None, eval_dataloader=None, eval_metric=None, **kwargs):
    assert isinstance(config, MixedPrecisionConfig), "Please provide MixedPrecisionConfig!"
    options = Options() if "options" not in kwargs else kwargs["options"]
    conf = Config(quantization=config, options=options)
    converter = MixedPrecision(conf)
    converter.precisions = config.precisions
    converter.model = model
    if eval_func is not None:
        converter.eval_func = eval_func
    if eval_dataloader is not None:
        converter.eval_dataloader = eval_dataloader
    if eval_metric is not None:
        converter.metric = eval_metric
    return converter()
