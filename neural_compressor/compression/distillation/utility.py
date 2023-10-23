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
"""This is an utility file for PyTorch distillation."""

from neural_compressor.utils.utility import LazyImport

torch = LazyImport("torch")

STUDENT_FEATURES = {}
TEACHER_FEATURES = {}


# for adapting fx model
@torch.fx.wrap
def record_output(output, name, output_process, student=False):
    """Record layers output.

    It is a help function.
    """
    recorded_output = output
    if output_process != "":
        if isinstance(output, dict) and output_process in output:
            recorded_output = output[output_process]
        elif isinstance(output, (tuple, list)) and str.isnumeric(output_process):
            recorded_output = output[int(output_process)]
        elif callable(output_process):
            recorded_output = output_process(output)
        else:
            raise NotImplementedError(
                "Current only support get the data with "
                + "integer index in case the output is tuple or list and only "
                + "need one item or with key in case the output is dict,  "
                + "or output_process is a function."
            )
    if student:
        STUDENT_FEATURES[name].append(recorded_output)
    else:
        TEACHER_FEATURES[name].append(recorded_output)
    return output


def get_activation(name, output_process="", student=False):
    """Get a hook for getting activation."""

    def hook(model, input, output):
        if model.training or not student:
            return record_output(output, name, output_process, student=student)
        else:
            return output

    return hook
