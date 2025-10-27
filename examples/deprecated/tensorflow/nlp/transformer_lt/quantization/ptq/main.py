#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
#
import re
import six
import sys
import time
import numpy as np
import unicodedata
import pandas as pd
from absl import app
import tensorflow as tf
from argparse import ArgumentParser

from utils import metrics
from utils import tokenizer
from utils.tokenizer import Subtokenizer
from neural_compressor.data import DataLoader

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 64,
                     "run batch size")

flags.DEFINE_string("input_graph", None,
                    "The path of input model file.")

flags.DEFINE_string("inputs_file", None,
                    "File saved to an output file.")

flags.DEFINE_string("reference_file", None,
                    "File containing reference translation.")

flags.DEFINE_string("vocab_file", None,
                    "Path to subtoken vocabulary file.")

flags.DEFINE_string("output_model", None,
                    "The output model of the quantized model.")

flags.DEFINE_bool('tune', False,
                    'whether to tune the model')

flags.DEFINE_bool('benchmark', False, 
                    'whether to benchmark the model')

flags.DEFINE_string("mode", 'performance',
                     "One of three options: 'performance'/'accuracy'.")

flags.DEFINE_integer("iters", 100,
                     "The iteration used for benchmark.")

class UnicodeRegex(object):
    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
        return "".join(six.unichr(x) for x in range(sys.maxunicode)
                    if unicodedata.category(six.unichr(x)).startswith(prefix))

uregex = UnicodeRegex()

def bleu_tokenize(string):
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()

class bleu(object):
    def __init__(self):
        self.translations = []
        self.labels = []

    def reset(self):
        self.translations = []
        self.labels = []

    def update(self, pred, label):
        if len(label) != len(pred):
            raise ValueError("Reference and translation files have different number "
                             "of lines. If training only a few steps (100-200), the "
                             "translation may be empty.")
        label = [x.lower() for x in label]
        pred = [x.lower() for x in pred]
        label = [bleu_tokenize(x) for x in label]
        pred = [bleu_tokenize(x) for x in pred]
        self.labels.extend(label)
        self.translations.extend(pred)

    def result(self):
        return metrics.compute_bleu(self.labels, self.translations) * 100

def collate_fn(batch):
    """Puts each data field into a pd frame with outer dimension batch size"""
    elem = batch[0]
    if isinstance(elem, tuple):
        batch = zip(*batch)
        return [collate_fn(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        return [list(elem) for elem in batch]
    elif isinstance(elem, str):
        return batch
    else:
        return pd.DataFrame(batch).fillna(0).values.astype(np.int32)

def load_graph(file_name):
    tf.compat.v1.logging.info('Loading graph from: ' + file_name)
    with tf.io.gfile.GFile(file_name, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def eval_func(infer_graph, iteration=-1):
    if isinstance(infer_graph, tf.compat.v1.GraphDef):
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(infer_graph, name='')
        infer_graph = graph

    subtokenizer = Subtokenizer(FLAGS.vocab_file)
    input_tensor = infer_graph.get_tensor_by_name('input_tensor:0')
    output_tensor = infer_graph.get_tensor_by_name(\
        'model/Transformer/strided_slice_19:0')

    ds = Dataset(FLAGS.inputs_file, FLAGS.reference_file, FLAGS.vocab_file)
    dataloader = DataLoader(framework='tensorflow', dataset=ds,
                                batch_size=FLAGS.batch_size, collate_fn=collate_fn)

    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=infer_graph, config=config)
    iteration=-1
    time_list = []
    bleu_eval = bleu()
    predictions = []
    labels = []
    warmup = 10
    if FLAGS.benchmark and FLAGS.mode == 'performance':
       iteration = FLAGS.iters
       assert iteration >= warmup, 'iteration must be larger than warmup'

    for idx, (input_data, label) in enumerate(dataloader):
        if idx < iteration or iteration == -1:
            time_start = time.time()
            out = sess.run([output_tensor], {input_tensor: input_data})
            duration = time.time() - time_start
            time_list.append(duration)
            predictions.append(out)
            labels.extend(label)
        else:
            break

    latency = np.array(time_list[warmup: ]).mean() / FLAGS.batch_size
    if FLAGS.benchmark and FLAGS.mode == 'performance':
        print('Batch size = {}'.format(FLAGS.batch_size))
        print('Latency: {:.3f} ms'.format(latency * 1000))
        print('Throughput: {:.3f} items/sec'.format(1./ latency))

    # only calculate accuracy when running out all predictions
    if iteration == -1:
        decode = []
        for i,tr in enumerate(predictions):
            for j,itr in enumerate(tr):
                for k, otr in enumerate(itr):
                    try:
                        index = list(otr).index(tokenizer.EOS_ID)
                        decode.append(subtokenizer.decode(otr[:index]))
                    except:
                        decode.append(subtokenizer.decode(otr))
        bleu_eval.update(decode, labels)
        print('Accuracy is {:.3f}'.format(bleu_eval.result()))
        return bleu_eval.result()

class Dataset(object):
    def __init__(self, inputs_file, reference_file, vocab_file):
        with tf.io.gfile.GFile(inputs_file) as f:
            records = f.read().split("\n")
            inputs = [record.strip() for record in records]
            if not inputs[-1]:
                inputs.pop()

        self.ref_lines = tokenizer.native_to_unicode(
            tf.io.gfile.GFile(reference_file).read()).strip().splitlines()

        subtokenizer = Subtokenizer(vocab_file)
        self.batch = []
        token_lens=[]
        for i, line in enumerate(inputs):
            enc = subtokenizer.encode(line, add_eos=True)
            token_lens.append((i, len(enc)))

        sorted_by_token_input_lens = sorted(token_lens, key=lambda x: x[1], reverse=True)

        sorted_inputs = [None] * len(sorted_by_token_input_lens)
        sorted_keys = [0] * len(sorted_by_token_input_lens)

        lines = []
        for i, (index, _) in enumerate(sorted_by_token_input_lens):
            sorted_inputs[i] = inputs[index]
            sorted_keys[index] = i
            enc=subtokenizer.encode(sorted_inputs[i], add_eos=True)
            lines.append([enc])
        for i in sorted_keys:
            self.batch.append(lines[i])

    def __getitem__(self, index):
        data = self.batch[index]
        label = self.ref_lines[index]
        return data[0], label

    def __len__(self):
        return len(self.batch)

def main(_):
    graph = load_graph(FLAGS.input_graph)
    if FLAGS.tune:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
        ds = Dataset(FLAGS.inputs_file, FLAGS.reference_file, FLAGS.vocab_file)
        calib_dataloader = DataLoader(framework='tensorflow', dataset=ds, \
                                        batch_size=FLAGS.batch_size, collate_fn=collate_fn,)										
        conf = PostTrainingQuantConfig(inputs=['input_tensor'],
                                        outputs=['model/Transformer/strided_slice_19'],
                                        calibration_sampling_size=[500])       
        q_model = quantization.fit(graph, conf=conf, calib_dataloader=calib_dataloader,
                    eval_func=eval_func)
        try:
            q_model.save(FLAGS.output_model)
        except Exception as e:
            print("Failed to save model due to {}".format(str(e)))

    if FLAGS.benchmark:
        assert FLAGS.mode == 'performance' or FLAGS.mode == 'accuracy', \
        "Benchmark only supports performance or accuracy mode."
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        if FLAGS.mode == 'performance':
            conf = BenchmarkConfig(cores_per_instance=28, num_of_instance=1)
            fit(graph, conf, b_func=eval_func)
        elif FLAGS.mode == 'accuracy':
            eval_func(graph)

if __name__ == "__main__":
    tf.compat.v1.app.run()
