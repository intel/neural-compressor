from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from absl import app
from absl import logging
tf.compat.v1.disable_v2_behavior()

tf.compat.v1.flags.DEFINE_bool("saved_model",
                               False,
                               "whether export saved model or not")
FLAGS = tf.compat.v1.flags.FLAGS

# We just import classifier here for `create_model` and some processors such as
# MNLI or MRPC. Because of the flags defined in `run_classifier.py`, we need not
# to define the flags again.
from run_classifier import create_model_top
from run_classifier import ColaProcessor
from run_classifier import MnliProcessor
from run_classifier import MrpcProcessor
from run_classifier import XnliProcessor
from modeling import BertConfig

class ClassifierExporter:
  def __init__(self,
               output_dir: str,
               task_name: str,
               bert_config: str,
               max_seq_length: int):

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor
    }

    task_name = task_name.lower()
    if task_name not in processors:
      raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # create model for CPU/dGPU, not TPU
    use_one_hot_embeddings = False

    bert_config = BertConfig.from_json_file(bert_config)
    if FLAGS.precision:
      bert_config.precision = FLAGS.precision

    self.session = tf.compat.v1.Session()

    placeholder = tf.compat.v1.placeholder
    input_shape = [None, max_seq_length]
    self.label_ids = placeholder(tf.int32, [None], name='label_ids')
    self.input_ids = placeholder(tf.int32, input_shape, name='input_ids')
    self.input_mask = placeholder(tf.int32, input_shape, name='input_mask')
    self.segment_ids = placeholder(tf.int32, input_shape, name='segment_ids')

    self.loss, self.per_example_loss, self.logits, self.probabilities = \
      create_model_top(bert_config, False, # is training
                       self.input_ids, self.input_mask, self.segment_ids,
                       self.label_ids, num_labels, use_one_hot_embeddings,
                       None) # frozen graph path

    latest_model = tf.train.latest_checkpoint(FLAGS.output_dir)
    saver = tf.compat.v1.train.Saver()
    saver.restore(self.session, latest_model)

    self.output_dir = output_dir
    self.dest_dir = os.path.join(self.output_dir, "frozen")
    if not os.path.exists(self.dest_dir):
      os.mkdir(self.dest_dir)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    self.session.close()

  def export(self, saved_model: bool):
    if saved_model:
      self.export_saved_model()

    self.export_frozen_graph()

  def export_saved_model(self,
                         signature_def_name="eval",
                         tag=tf.compat.v1.saved_model.tag_constants.SERVING):
    build_tensor_info = tf.compat.v1.saved_model.build_tensor_info
    signature_def_utils = tf.compat.v1.saved_model.signature_def_utils
    inputs = {
        'label_ids': build_tensor_info(self.label_ids),
        'input_ids': build_tensor_info(self.input_ids),
        'input_mask': build_tensor_info(self.input_mask),
        'segment_ids': build_tensor_info(self.segment_ids)
    }

    outputs = {
        "loss": build_tensor_info(self.loss),
        "per_example_loss": build_tensor_info(self.per_example_loss),
        "logits": build_tensor_info(self.logits),
        "probabilities": build_tensor_info(self.probabilities)
    }

    signature = signature_def_utils.build_signature_def(inputs, outputs)
    signature_def_map = {signature_def_name: signature}

    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(self.dest_dir)
    builder.add_meta_graph_and_variables(self.session, [tag], signature_def_map)
    builder.save()

  def export_frozen_graph(self, frozen_graph_name="frozen_graph.pb"):
    # we should disable v2 behavior, at the same time, the bn norm has some op name difference
    # should be handled. Otherwise, it will throw exception when do import graph def.
    # https://www.bountysource.com/issues/36614355-unable-to-import-frozen-graph-with-batchnorm
    graph_def = self.session.graph.as_graph_def()
    for node in graph_def.node:
      if node.op == 'RefEnter':
        node.op = 'Enter'
        for index in range(len(node.input)):
          if 'moving_' in node.input[index]:
            node.input[index] = node.input[index] + '/read'
      if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
          if 'moving_' in node.input[index]:
            node.input[index] = node.input[index] + '/read'
      elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']
      elif node.op == 'AssignAdd':
        node.op = 'Add'
        if 'use_locking' in node.attr: del node.attr['use_locking']

    outputs_name = ['loss/Mean', 'loss/Sum', 'loss/BiasAdd', 'loss/Softmax']
    graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(self.session,
                                                                       graph_def,
                                                                       outputs_name)

    path = os.path.join(self.dest_dir, frozen_graph_name)
    with tf.compat.v1.gfile.GFile(path, 'wb') as pb_file:
      pb_file.write(graph_def.SerializeToString())

def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  with ClassifierExporter(FLAGS.output_dir,
                          FLAGS.task_name,
                          FLAGS.bert_config_file,
                          FLAGS.max_seq_length) as exporter:
    exporter.export(FLAGS.saved_model)

if __name__ == "__main__":
  tf.compat.v1.flags.mark_flag_as_required("task_name")
  tf.compat.v1.flags.mark_flag_as_required("bert_config_file")
  tf.compat.v1.flags.mark_flag_as_required("output_dir")
  tf.compat.v1.app.run()
