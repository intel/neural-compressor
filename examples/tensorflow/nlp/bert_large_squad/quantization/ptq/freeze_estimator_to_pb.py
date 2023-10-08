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
"""Freeze estimator to frozen pb for bert full pipeline tuning."""

import os
import modeling
import tensorflow as tf
import numpy as np
from absl import app
from absl import logging

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "input_model", None, "The input checkpoint path of model.")

flags.DEFINE_string(
    "output_model", None, "The output path frozen pb will be written.")

def write_graph(out_graph_def, out_graph_file):
    from tensorflow.python.platform import gfile
    if not isinstance(out_graph_def, tf.compat.v1.GraphDef):
        raise ValueError(
            'out_graph_def is not instance of TensorFlow GraphDef.')
    if out_graph_file and not os.path.exists(os.path.dirname(out_graph_file)):
        raise ValueError('"output_graph" directory does not exists.')
    f = gfile.GFile(out_graph_file, 'wb')
    f.write(out_graph_def.SerializeToString())

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.compat.v1.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.compat.v1.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.compat.v1.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(a=logits, perm=[2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.compat.v1.logging.info("*** Features ***")
    # for name in sorted(features.keys()):
    #   tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, "SQuAD")
      if use_tpu:

        def tpu_scaffold():
          tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.compat.v1.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"

    output_spec = None

    predictions = {
        "unique_ids": unique_ids,
        "start_logits": start_logits,
        "end_logits": end_logits,
    }
    output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.io.FixedLenFeature([], tf.int64),
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.io.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.io.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, dtype=tf.int32)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    input_file_placeholder = tf.compat.v1.placeholder(shape=[],
        name="input_file", dtype=tf.string)
    batch_size_placeholder = tf.compat.v1.placeholder(shape=[],
        name="batch_size", dtype=tf.int64)
    #batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    # d = tf.data.TFRecordDataset(input_file)
    d = tf.data.TFRecordDataset(input_file_placeholder)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size_placeholder,
            drop_remainder=drop_remainder))

    return d

  return input_fn

bert_config_dict = {'vocab_size': 30522, 'hidden_size': 1024, 'num_hidden_layers': 24, \
                    'num_attention_heads': 16, 'hidden_act': 'gelu', 'intermediate_size': 4096,\
                    'hidden_dropout_prob': 0.1, 'attention_probs_dropout_prob': 0.1, \
                    'max_position_embeddings': 512, 'type_vocab_size': 2, 'initializer_range': 0.02, \
                    'precision': 'fp32', 'new_bf16_scope': True, 'experimental_gelu': False, \
                    'optimized_softmax': False}

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  bert_config = modeling.BertConfig.from_dict(bert_config_dict)

  session_config = tf.compat.v1.ConfigProto(
      inter_op_parallelism_threads=2,
      intra_op_parallelism_threads=27,
      allow_soft_placement=True)

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2

  run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=None,
      master=None,
      model_dir='./',
      save_checkpoints_steps=1000,
      session_config=session_config,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=1000,
          num_shards=8,
          per_host_input_for_training=is_per_host))

  predict_input_fn = input_fn_builder(
      input_file='',
      seq_length=384,
      is_training=False,
      drop_remainder=False)

  from neural_compressor.adaptor.tf_utils.util import is_ckpt_format
  assert is_ckpt_format(FLAGS.input_model), 'invalid checkpoint path....'
  ckpt_model = [os.path.splitext(i)[0] for i in os.listdir(FLAGS.input_model) \
      if i.endswith('.meta')][0]
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=os.path.join(FLAGS.input_model, ckpt_model),
      learning_rate=5e-5,
      num_train_steps=None,
      num_warmup_steps=None,
      use_tpu=False,
      use_one_hot_embeddings=False)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=32,
      predict_batch_size=8)

  from neural_compressor.adaptor.tf_utils.util import get_estimator_graph
  graph = get_estimator_graph(estimator, predict_input_fn)
  write_graph(graph.as_graph_def(), FLAGS.output_model)

if __name__ == "__main__":
  tf.compat.v1.app.run()
