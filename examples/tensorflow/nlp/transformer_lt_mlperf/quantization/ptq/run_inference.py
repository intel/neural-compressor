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

import os
import sys
import re
import time
import six
import tensorflow as tf
import numpy as np
import pandas as pd
import unicodedata
from tensorflow.python.platform import flags
from tensorflow.python.platform.flags import FLAGS
from tensorflow.python.platform import app
from google.protobuf import text_format
from utils import tokenizer
from utils.tokenizer import Subtokenizer
from utils import metrics
from neural_compressor.experimental import Quantization, common
from neural_compressor.data import DATALOADERS
from neural_compressor.utils.utility import dump_elapsed_time


INPUT_TENSOR_NAMES = ['input_tokens:0']
OUTPUT_TENSOR_NAMES = ["model/Transformer/strided_slice_15:0"]

flags.DEFINE_string(
    "config", "transformer_lt_mlperf.yaml",
    """Quantization configuration file to load.""")
flags.DEFINE_string(
    "input_graph", "transformer_mlperf_fp32.pb",
    """TensorFlow 'GraphDef' file to load.""")
flags.DEFINE_string(
    "output_model", "output_transformer_mlperf_int8.pb",
    """The output model of the quantized model.""")
flags.DEFINE_bool(
    "input_binary", True,
    """Whether the input files are in binary format.""")
flags.DEFINE_string(
    "vocab_file", "vocab.ende.32768",
    """Path to subtoken vocabulary file.""")
flags.DEFINE_string(
    "input_file", "newstest2014.en",
    """File containing text to translate.""")
flags.DEFINE_string(
    "reference_file", "newstest2014.de",
    """File containing reference translation.""")
flags.DEFINE_string(
    "file_out", "output_translation_result.txt",
    """Save latest translation to this file when using 'accuracy'/'tune' mode.""")
flags.DEFINE_integer(
    "batch_size", 64,
    """The validation batch size.""")
flags.DEFINE_integer(
    "num_inter", 2,
    """Number of inter op parallelism thread to use.""")
flags.DEFINE_integer(
    "num_intra", 56,
    """Number of intra op parallelism thread to use.""")
flags.DEFINE_integer(
    "warmup_steps", 5,
    """Number of warmup steps before benchmarking the model.""")
flags.DEFINE_integer(
    "iters", -1,
    "The iteration used for 'benchmark' mode.")
flags.DEFINE_string(
    "mode", "tune",
    """One of three options: 'benchmark'/'accuracy'/'tune'.""")
flags.DEFINE_string(
    "bleu_variant", "uncased",
    """One of two options: 'uncased'/'cased'.""")

def load_graph(file_name):
    with tf.io.gfile.GFile(file_name, "rb") as f:
        if FLAGS.input_binary:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), graph_def)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    tf.compat.v1.logging.info('Loaded graph from: ' + file_name)
    return graph

def _trim_and_decode(ids, subtokenizer):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
        index = list(ids).index(tokenizer.EOS_ID)
        return subtokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
        return subtokenizer.decode(ids)

class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols."""
    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
        return "".join(six.unichr(x) for x in range(sys.maxunicode) \
            if unicodedata.category(six.unichr(x)).startswith(prefix))

def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.

    See https://github.com/moses-smt/mosesdecoder/'
            'blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).

    Note that a numer (e.g. a year) followed by a dot at the end of sentence
    is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.

    Args:
        string: the input string

    Returns:
        a list of tokens
    """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()

def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
    """Compute BLEU for two files (reference and hypothesis translation)."""
    ref_lines = tf.io.gfile.GFile(ref_filename).read().strip().splitlines()
    hyp_lines = tf.io.gfile.GFile(hyp_filename).read().strip().splitlines()
    if len(ref_lines) != len(hyp_lines):
        raise ValueError("Reference and translation files have different number of lines.")
    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
    return metrics.compute_bleu(ref_tokens, hyp_tokens) * 100

class Dataset(object):
    def __init__(self, input_file, vocab_file):
        with tf.io.gfile.GFile(input_file) as f:
            records = f.read().split("\n")
            inputs = [record.strip() for record in records]
            if not inputs[-1]:
                inputs.pop()
        subtokenizer = Subtokenizer(vocab_file)
        self.lines = []
        token_lens=[]
        for i, line in enumerate(inputs):
            enc = subtokenizer.encode(line, add_eos=True)
            token_lens.append((i, len(enc)))
        sorted_by_token_input_lens = sorted(token_lens, key=lambda x: x[1], reverse=True)
        sorted_inputs = [None] * len(sorted_by_token_input_lens)
        self.sorted_keys = [0] * len(sorted_by_token_input_lens)
        for i, (index, _) in enumerate(sorted_by_token_input_lens):
            sorted_inputs[i] = inputs[index]
            self.sorted_keys[index] = i
            enc=subtokenizer.encode(sorted_inputs[i], add_eos=True)
            self.lines.append(enc)

    def __getitem__(self, index):
        return self.lines[index], 0

    def __len__(self):
        return len(self.lines)

def collate_fn(batch):
    """Puts each data field into a pd frame with outer dimension batch size."""
    elem = batch[0]
    if isinstance(elem, tuple):
        batch = zip(*batch)
        return [collate_fn(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        return [list(elem) for elem in batch]
    elif isinstance(elem, str) or isinstance(elem, int):
        return batch
    else:
        return pd.DataFrame(batch).fillna(0).values.astype(np.int32)

@dump_elapsed_time(customized_msg="Customized eval_func")
def eval_func(infer_graph):
    assert FLAGS.mode in ["benchmark", "accuracy", "tune"], \
            "'mode' must be one of three options: 'benchmark'/'accuracy'/'tune'."
    dataset = Dataset(FLAGS.input_file, FLAGS.vocab_file)
    sorted_keys = dataset.sorted_keys
    dataloader = DATALOADERS['tensorflow'] \
        (dataset, batch_size=FLAGS.batch_size, collate_fn=collate_fn)
    input_tensors = list(map(infer_graph.get_tensor_by_name, INPUT_TENSOR_NAMES))
    output_tensors = list(map(infer_graph.get_tensor_by_name, OUTPUT_TENSOR_NAMES))

    session_config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads = FLAGS.num_inter,
        intra_op_parallelism_threads = FLAGS.num_intra)
    with tf.compat.v1.Session(config=session_config, graph=infer_graph) as sess:
        time_list = []
        translations = []
        warmup = FLAGS.warmup_steps if FLAGS.warmup_steps > 0 else 0
        iteration = FLAGS.iters \
            if FLAGS.iters > -1 and FLAGS.mode == "benchmark" else len(dataloader)
        assert iteration != 0, \
            "'iteration' cannot be zero."
        assert iteration >= warmup, \
            "'iteration' must be greater than or equal to warmup."
        assert iteration <= len(dataloader), \
            "'iteration' must be less than or equal to len(dataloader)."
        if FLAGS.mode == "benchmark":
            tf.compat.v1.logging.info \
                ('******** Start to get performance of the model ********')
        else:
            tf.compat.v1.logging.info \
                ('******** Start to get accuracy and performance of the model ********')
        if warmup > 0:
            tf.compat.v1.logging.info \
                ('Start to do warm-up with {}/{} (steps/total_iterations) before getting performance.' \
                    .format(warmup, iteration))
        else:
            tf.compat.v1.logging.info \
                ('Start to get performance with {} iterations.'.format(iteration))
        for idx, (input_data, _) in enumerate(dataloader):
            if idx < iteration:
                if idx == warmup and warmup > 0:
                    tf.compat.v1.logging.info('The warm-up is over.')
                    tf.compat.v1.logging.info \
                        ('Start to get performance with {}/{} (steps/total_iterations).' \
                            .format(iteration - warmup, iteration))
                feed_dict = {input_tensors[0]: input_data}
                time_start = time.time()
                dec_tensor = sess.run(output_tensors, feed_dict)
                duration = time.time() - time_start
                time_list.append(duration)
                translations.append(dec_tensor)
            else:
                break
    latency = np.array(time_list[warmup:]).mean() / FLAGS.batch_size
    tf.compat.v1.logging.info('Batch-size = {}'.format(FLAGS.batch_size))
    tf.compat.v1.logging.info('Latency: {:.3f} ms'.format(latency * 1000))
    tf.compat.v1.logging.info('Throughput: {:.3f} items/sec'.format(1./ latency))

    if FLAGS.mode != "benchmark":
        """Write translations to file and calculate BLEU score."""
        translation_count = 0
        decoded_translations=[]
        subtokenizer = Subtokenizer(FLAGS.vocab_file)
        for i,tr in enumerate(translations):
            for j,itr in enumerate(tr):
                for k,otr in enumerate(itr):
                    translation_count += 1
                    decoded_translations.append(_trim_and_decode(otr, subtokenizer))
        tf.compat.v1.logging.info \
            ('Total number of sentences translated:%d' % (translation_count))
        tf.io.gfile.makedirs(os.path.dirname(FLAGS.file_out))
        with tf.io.gfile.GFile(FLAGS.file_out, "w") as f:
            for i in sorted_keys:
                f.write("%s\n" % decoded_translations[i])

        global uregex
        uregex = UnicodeRegex()
        score_uncased = bleu_wrapper(FLAGS.reference_file, FLAGS.file_out, False)
        tf.compat.v1.logging.info("Case-insensitive results: {:.8f}".format(score_uncased))
        score_cased = bleu_wrapper(FLAGS.reference_file, FLAGS.file_out, True)
        tf.compat.v1.logging.info("Case-sensitive results: {:.8f}".format(score_cased))
        assert FLAGS.bleu_variant in ["uncased", "cased"], \
             "'bleu_variant' must be one of two options: 'uncased'/'cased'."
        if FLAGS.bleu_variant == "uncased":
            return score_uncased
        else:
            return score_cased

def main(unused_args):
    graph = load_graph(FLAGS.input_graph)
    if FLAGS.mode == 'tune':
        quantizer = Quantization(FLAGS.config)
        dataset = Dataset(FLAGS.input_file, FLAGS.vocab_file)
        quantizer.calib_dataloader = common.DataLoader(dataset,
                                                       collate_fn = collate_fn,
                                                       batch_size = FLAGS.batch_size)
        quantizer.model = common.Model(graph)
        quantizer.eval_func = eval_func
        q_model = quantizer.fit()
        try:
            q_model.save(FLAGS.output_model)
        except Exception as e:
            tf.compat.v1.logging.error("Failed to save model due to {}".format(str(e)))
    else:
        eval_func(graph)


if __name__ == "__main__":
    app.run()
