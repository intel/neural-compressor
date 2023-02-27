#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
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
#
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pkg_resources import parse_version

_inprecision = tf.float32
_rprecision = tf.float32
if parse_version(tf.version.VERSION) < parse_version('2.9.0'):
    _keras_policy = tf.keras.mixed_precision.experimental.Policy("float32")
else:
    _keras_policy = tf.keras.mixed_precision.Policy("float32")

_use_optimized_softmax = True
_use_experimental_gelu = True

def set_global_precision(dt):
  # Set Keras API precision
  global _keras_policy
  if dt == tf.bfloat16:
     if parse_version(tf.version.VERSION) < parse_version('2.9.0'):
        _keras_policy=tf.keras.mixed_precision.experimental.Policy("mixed_bfloat16")
     else:
         _keras_policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")

  # Set basic API precision
  set_rprecision(dt)

def set_rprecision(dt):
  global _rprecision
  _rprecision=dt

def get_keras_policy():
  return _keras_policy

def set_global_flags(optimized_softmax, experimental_gelu):
  global _use_optimized_softmax
  global _use_experimental_gelu
  _use_optimized_softmax = optimized_softmax
  _use_experimental_gelu = experimental_gelu

def i_cast(x) :
     return tf.cast(x, _inprecision)

def r_cast(x) :
     return tf.cast(x, _rprecision)

def multiply(x,y):
    x = r_cast(x)
    y = r_cast(y)
    return tf.multiply(x,y)

def mzip(x,y):
    if x.dtype== tf.bfloat16:
      x = r_cast(x)
      y = r_cast(y)
    return zip(x,y)

def tanh(x):
    x = i_cast(x)
    rval = tf.tanh(x)
    return r_cast(rval)

def softmax(scores, axis=None):
    if _use_optimized_softmax:
      return tf.nn.softmax(scores, axis)
    else:
      scores = i_cast(scores)
      rval = tf.nn.softmax(scores, axis)
      return r_cast(rval)

def layer_norm(inputs, begin_norm_axis, begin_params_axis, scope):
    lnorm = tf.keras.layers.LayerNormalization(dtype=get_keras_policy())
    return lnorm(inputs)

"Moved from modeling.py"
def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  #if _use_experial_gelu:
  if True:
    print("using experimental gelu")
    return tf.nn.gelu(x)
  else:
    x = i_cast(x)
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    rval = x * cdf
    return r_cast(rval)

def logTheLossHook(total_loss, n):
    return tf.compat.v1.train.LoggingTensorHook({"\t Loss " : total_loss}, every_n_iter=n)

