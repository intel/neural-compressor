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
"""Convenience function for extracting the values for logging calls.

Because TensorFlow generally defers computation of values to a session run call,
it is impractical to log the values of tensors when they are defined. Instead,
the definition of a tensor is logged as normal using the log function in
mlperf_log.py and a tf.print statement helper function can be used to report
the relevant values as they are computed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf


def log_deferred(op, log_id, every_n=1, first_n=None):
  """Helper method inserting compliance logging ops.

  Note: This helper is not guaranteed to be efficient, as it will insert ops
        and control dependencies. If this proves to be a bottleneck, submitters
        may wish to consider other methods such as extracting values from an
        .events file.

  Args:
    op: A tf op to be printed.
    log_id: a uuid provided by the logger in mlperf_log.py
    every_n: If repeat is True, with what frequency should the input op be '
             logged. If repeat is False, this argument is ignored.
    first_n: Only log this many values. This arg does not interact with every_n.
             The first_n refers to the first n that would have been logged.
  """

  prefix = ":::MLPv0.5.0 [{}]".format(log_id)
  if not first_n is not None and first_n == 1:
    return tf.compat.v1.Print(op, [tf.timestamp(), op], message=prefix, first_n=1)

  counter = tf.Variable(tf.zeros(shape=(), dtype=tf.int32) - 1,
                        aggregation=tf.VariableAggregation.MEAN)
  increment = tf.compat.v1.assign_add(counter, 1, use_locking=True)
  return tf.cond(
      pred=tf.equal(tf.math.mod(increment, every_n), 0),
      true_fn=lambda :tf.compat.v1.Print(op, [tf.timestamp(), op], message=prefix,
                       first_n=first_n),
      false_fn=lambda :op
  )


def sum_metric(tensor, name):
  sum_var = tf.compat.v1.Variable(
    initial_value=tf.zeros(shape=(), dtype=tensor.dtype),
    trainable=False,
    collections=[
      tf.compat.v1.GraphKeys.LOCAL_VARIABLES,
      tf.compat.v1.GraphKeys.METRIC_VARIABLES,
    ],
    name="{}_total".format(name),
    aggregation=tf.VariableAggregation.SUM
  )

  update_op = tf.identity(tf.compat.v1.assign_add(sum_var, tensor))
  return tf.identity(sum_var, name=name), update_op


def _example():
  for kwargs in [dict(first_n=1), dict(), dict(every_n=2),
                 dict(first_n=2, every_n=2)]:
    op = tf.compat.v1.assign_add(tf.Variable(tf.zeros(shape=(), dtype=tf.int32) - 1), 1)
    op = log_deferred(op, str(uuid.uuid4()), **kwargs)
    init = [tf.compat.v1.local_variables_initializer(), tf.compat.v1.global_variables_initializer()]
    print("-" * 5)
    with tf.compat.v1.Session().as_default() as sess:
      sess.run(init)
      for _ in range(6):
        sess.run(op)


if __name__ == "__main__":
  _example()
