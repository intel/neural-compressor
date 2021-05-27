# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
import generic_ops as bf


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, accum_steps=1, use_tpu=False, fine_tuning=True):
  """Creates an optimizer training op."""
  global_step = tf.compat.v1.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.compat.v1.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.compat.v1.trainable_variables()


  if accum_steps > 1 :
    #tf.compat.v1.logging.info("Accumulation Steps....")
    grads_and_vars = optimizer.compute_gradients(loss * 1.0 / accum_steps, tvars)

    current_step = tf.compat.v1.get_variable(name="current_step", shape=[], dtype=tf.int32, 
                                   trainable=False,
                                   initializer=tf.zeros_initializer)
    accum_vars = [tf.compat.v1.get_variable(
          name=tvar.name.split(":")[0] + "/agrads",
          shape=tvar.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer()) for tvar in tvars]

    apply_grads = tf.cast(tf.math.equal(current_step % accum_steps, 0), dtype=tf.bool)
    current_step = tf.cond(apply_grads, 
                           lambda:current_step.assign(tf.ones_like(current_step)), 
                           lambda:current_step.assign_add(1))
                           #lambda:inc_current_step(current_step, "Apply Grads:"), 
                           #lambda:inc_current_step(current_step, "Step:"))

    grads_and_vars_and_accums = [(gv[0],gv[1],accum_vars[i]) for i, gv in enumerate(grads_and_vars) if gv[0] is not None]
    grads, tvars, accum_vars = list(zip(*grads_and_vars_and_accums))

    (cgrads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    #accum_vars = update_accum_vars(accum_vars, apply_grads, cgrads, current_step)
    accum_vars = tf.cond(apply_grads,
              lambda: [accum_vars[i].assign(grad) for i, grad in enumerate(cgrads)],
              lambda: [accum_vars[i].assign_add(grad) for i, grad in enumerate(cgrads)])

    def applyGrads(accum_vars, current_step):
          #tf.compat.v1.logging.info("\t\t APPLYING GRADIENTS....:", global_step)
          return optimizer.apply_gradients(list(zip(accum_vars, tvars)), global_step=global_step)

    apply_step = tf.identity(tf.cast(tf.math.equal(current_step % accum_steps, 0), dtype=tf.bool), name="apply_step")
    update_op = tf.cond(apply_step, lambda: applyGrads(accum_vars, current_step), lambda: tf.no_op())

    new_global_step = tf.cond(apply_step, lambda: global_step+1, lambda: global_step)
    new_global_step = tf.identity(new_global_step, name='global_step_update')
    train_op = tf.group(update_op, [global_step.assign(new_global_step)])
  else :
    grads = tf.gradients(ys=loss, xs=tvars)

    # This is how the model was pre-trained.
    if fine_tuning :
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    else :
      gdtypes = [grad.dtype for grad in grads];
      mgrads = [tf.cast(grad, tf.float32) for grad in grads] 
      (grads, _) = tf.clip_by_global_norm(mgrads, clip_norm=1.0)
      grads = [tf.cast(grad, ddtype) for grad, ddtype in zip(grads, gdtypes)]

    train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step, fine_tuning=fine_tuning)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    new_global_step = tf.identity(new_global_step, name='global_step_update')
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

  return train_op


class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, fine_tuning=True, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())

      # Standard Adam update.
      if fine_tuning :
        next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
        next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))
      else :
        next_m = (
          bf.multiply(self.beta_1, m) + bf.multiply(1.0 - self.beta_1, grad))
        next_v = (
          bf.multiply(self.beta_2, v) + bf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        if fine_tuning :
          update += self.weight_decay_rate * param
        else :
          update += tf.cast(self.weight_decay_rate * param, update.dtype)

      if fine_tuning :
        update_with_lr = self.learning_rate * update

        next_param = param - update_with_lr

        assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
      else :
        update_with_lr = tf.cast(self.learning_rate, update.dtype) * update

        next_param = tf.cast(param, update_with_lr.dtype) - update_with_lr

      
        param, m, v = tf.cast(next_param, param.dtype), tf.cast(m, m.dtype), tf.cast(v, v.dtype)

        assignments.extend([param, m, v])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
