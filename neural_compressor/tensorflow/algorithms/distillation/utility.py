#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Intel Corporation
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
"""This is an utility file for distillation."""

from neural_compressor.tensorflow.utils import LazyImport

STUDENT_FEATURES = {}
TEACHER_FEATURES = {}

def get_activation(name, output_process="", student=False):
    """Get a hook for getting activation."""

    def hook(model, input, output):
        if model.training or not student:
            return record_output(output, name, output_process, student=student)
        else:
            return output

    return hook
