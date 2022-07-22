# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convenience functions for logging ResNet topology.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mlperf_compliance import mlperf_log

_STACK_OFFSET = 2

def _get_shape(input_tensor):
  return "({})".format(", ".join(
      [str(i) for i in input_tensor.shape.as_list()[1:]]))


def _in_out_shape(input_tensor, output_tensor):
  return "{} -> {}".format( _get_shape(input_tensor), _get_shape(output_tensor))


def log_max_pool(input_tensor, output_tensor):
  mlperf_log.resnet_print(
      key=mlperf_log.MODEL_HP_INITIAL_MAX_POOL, value=_in_out_shape(
      input_tensor=input_tensor, output_tensor=output_tensor),
      stack_offset=_STACK_OFFSET)


def log_begin_block(input_tensor, block_type):
  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_BEGIN_BLOCK,
                          value={"block_type": block_type},
                          stack_offset=_STACK_OFFSET)
  mlperf_log.resnet_print(
      key=mlperf_log.MODEL_HP_RESNET_TOPOLOGY,
      value=" Block Input: {}".format(_get_shape(input_tensor)),
      stack_offset=_STACK_OFFSET)


def log_end_block(output_tensor):
  mlperf_log.resnet_print(
      key=mlperf_log.MODEL_HP_END_BLOCK,
      value=" Block Output: {}".format(_get_shape(output_tensor)),
      stack_offset=_STACK_OFFSET)


def log_projection(input_tensor, output_tensor):
  mlperf_log.resnet_print(
      key=mlperf_log.MODEL_HP_PROJECTION_SHORTCUT,
      value=_in_out_shape(input_tensor, output_tensor),
      stack_offset=_STACK_OFFSET)


def log_conv2d(input_tensor, output_tensor, stride, filters, initializer,
               use_bias):
  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_CONV2D_FIXED_PADDING,
                          value=_in_out_shape(input_tensor, output_tensor),
                          stack_offset=_STACK_OFFSET)
  mlperf_log.resnet_print(
      key=mlperf_log.MODEL_HP_CONV2D_FIXED_PADDING,
      value={"stride": stride, "filters": filters, "initializer": initializer,
             "use_bias": use_bias},
      stack_offset=_STACK_OFFSET)


def log_batch_norm(input_tensor, output_tensor, momentum, epsilon, center,
                   scale, training):
  assert _get_shape(input_tensor) == _get_shape(output_tensor)
  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_BATCH_NORM, value={
    "shape": _get_shape(input_tensor), "momentum": momentum, "epsilon": epsilon,
    "center": center, "scale": scale, "training": training},
                          stack_offset=_STACK_OFFSET)
