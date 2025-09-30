
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

'''DistilBERT base inference, implementation adapted from Hugging Face Library https://huggingface.co/'''
import time
import tensorflow as tf
import numpy as np
import math
from argparse import ArgumentParser
from transformers import AutoTokenizer
from datasets import load_from_disk
from tensorflow.core.protobuf import saved_model_pb2
from neural_compressor.utils.utility import dump_elapsed_time
from neural_compressor.utils import logger
from tensorflow.python.client import timeline
import os

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

arg_parser = ArgumentParser(description="Distilbert inference")
arg_parser.add_argument("--task-name", type=str,
                        help="Name of the task to run benchmark.",
                        dest="task_name",
                        default="sst2"
                        )
arg_parser.add_argument("-c", "--config", type=str,
                        help="Quantization configuration file to load.",
                        dest="config",
                        default="distilbert_base.yaml"
                        )
arg_parser.add_argument("-g", "--in-graph", type=str,
                        help="Full path to the input graph.",
                        dest="input_graph",
                        default=None
                        )
arg_parser.add_argument("--data-location", type=str,
                        help="Path to the dataset.",
                        dest="data_location"
                        )
arg_parser.add_argument("-o", "--output-graph", type=str,
                        help="The output path of quantized graph.",
                        dest="output_graph",
                        default="output_distilbert_base_int8.pb"
                        )
arg_parser.add_argument("-m", "--mode", type=str,
                        choices=['performance', 'accuracy'],
                        help="One of two options: 'performance'/'accuracy'.",
                        dest="mode",
                        default="performance"
                        )
arg_parser.add_argument("--tune", type=boolean_string,
                        help="whether to apply quantization",
                        dest="tune",
                        default=False
                        )
arg_parser.add_argument('--sq', type=boolean_string, dest='sq', help='smooth quantization', default=False)
arg_parser.add_argument("--benchmark", type=boolean_string,
                        help="whether to do benchmark",
                        dest="benchmark",
                        default=False
                        )
arg_parser.add_argument('-e', "--num-inter-threads", type=int,
                        help="The number of inter-thread.",
                        dest="num_inter_threads",
                        default=2
                        )
arg_parser.add_argument('-a', "--num-intra-threads", type=int,
                        help="The number of intra-thread.",
                        dest="num_intra_threads",
                        default=28
                        )
arg_parser.add_argument("--pad-to-max-length", type=boolean_string,
                        help="Padding option.",
                        dest="pad_to_max_length",
                        default=True
                        )
arg_parser.add_argument("--warmup-steps", type=int,
                        help="Number of warmup steps.",
                        dest="warmup_steps",
                        default=10
                        )
arg_parser.add_argument("--max-seq-length", type=int,
                        help="Maximum total sequence length after tokenization.",
                        dest="max_seq_length",
                        default=128
                        )
arg_parser.add_argument("--steps", type=int,
                        help="Number of steps.",
                        dest="steps",
                        default=872
                        )
arg_parser.add_argument("--batch-size", type=int,
                        help="Inference batch-size.",
                        dest="batch_size",
                        default=128
                        )
arg_parser.add_argument("--profile", dest='profile',
                        type=boolean_string, help="profile",
                        default=False)

ARGS = arg_parser.parse_args()
MAX_STEPS = 872
MAX_WARMUP_STEPS = 22

def create_feed_dict_and_labels(dataset, batch_id= None, num_batch= None, idx= None):
    """Return the input dictionary for the given batch."""
    if idx is None:
        start_idx = batch_id * ARGS.batch_size
        if batch_id == num_batch - 1:
            end_idx = ARGS.steps
        else:
            end_idx = start_idx + ARGS.batch_size
        input_ids = np.array(dataset["input_ids"])[start_idx:end_idx, :]
        attention_mask = np.array(dataset["attention_mask"])[start_idx:end_idx, :]
        feed_dict = {"input_ids:0": input_ids,
                     "attention_mask:0": attention_mask,
        }
        labels = np.array(dataset["label"])[start_idx: end_idx]
    else:
        input_ids = np.array(dataset["input_ids"])[idx, :].reshape(1, -1)
        attention_mask = np.array(dataset["attention_mask"])[idx, :].reshape(1, -1)
        feed_dict = {"input_ids:0": input_ids,
                     "attention_mask:0": attention_mask,
        }
        labels = np.array(dataset["label"])[idx]
    return feed_dict, labels

def load_dataset(data_location):
    def preprocess_function(examples):
        """Tokenize the texts."""
        sentence1_key, sentence2_key = "sentence", None
        args = (
            (examples[sentence1_key],) if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding="max_length",
                            max_length=ARGS.max_seq_length,
                            truncation=True
        )
        return result

    # Load dataset (only validation split for inference)
    dataset = load_from_disk(data_location)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    # Set max sequence length
    if ARGS.max_seq_length > tokenizer.model_max_length:
        logger.info(f"The max sequence length passed ({ARGS.max_seq_length}) \
                    is larger than the max supported by model \
                   ({tokenizer.model_max_length}).Using max_seq_length =  \
                   {tokenizer.model_max_length}")
    ARGS.max_seq_length = min(ARGS.max_seq_length, tokenizer.model_max_length)
    # Tokenize the dataset
    dataset = dataset.map(preprocess_function, batched=True)
    return dataset

class Dataloader(object):
    def __init__(self, data_location, batch_size, steps):
        self.batch_size = batch_size
        self.data_location = data_location
        self.num_batch = math.ceil(steps / batch_size)

    def __iter__(self):
        return self.generate_dataloader(self.data_location).__iter__()

    def __len__(self):
        return self.num_batch

    def generate_dataloader(self, data_location):
        dataset = load_dataset(data_location)
        for batch_id in range(self.num_batch):
            feed_dict, labels = create_feed_dict_and_labels(dataset, batch_id, self.num_batch)
            yield feed_dict, labels

class Distilbert_base(object):
    def __init__(self):
        self.validate_args()
        self.dataset = load_dataset(ARGS.data_location)
        self.dataloader = Dataloader(ARGS.data_location, ARGS.batch_size, ARGS.steps)

    def validate_args(self):
        if ARGS.warmup_steps > MAX_WARMUP_STEPS:
            logger.warning("Warmup steps greater than max possible value of 22." + \
                           " Setting to max value of ", MAX_WARMUP_STEPS)
            ARGS.warmup_steps = MAX_WARMUP_STEPS
        if ARGS.tune or ARGS.sq or (ARGS.benchmark and ARGS.mode == "accuracy"):
            ARGS.steps = MAX_STEPS
        elif ARGS.benchmark:
            if ARGS.steps > (MAX_STEPS - MAX_WARMUP_STEPS):
                logger.warning("Steps greater than max possible value of {}.".format(MAX_STEPS - MAX_WARMUP_STEPS))
                logger.warning("Setting to max value of {}".format(MAX_STEPS - MAX_WARMUP_STEPS))
                ARGS.steps = MAX_STEPS - MAX_WARMUP_STEPS
        if not ARGS.data_location:
            raise SystemExit("Missing dataset path.")

    def load_graph(self):
        """Load the frozen model."""
        graph_def = tf.compat.v1.GraphDef()
        sm = saved_model_pb2.SavedModel()
        with tf.io.gfile.GFile(ARGS.input_graph, "rb") as f:
            try:
                content = f.read()
                graph_def.ParseFromString(content)
            except Exception:
                sm.ParseFromString(content)
                graph_def = sm.meta_graphs[0].graph_def
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        logger.info("Loaded graph from: " + ARGS.input_graph)
        return graph

    def get_correct_predictions(self, preds, label_ids):
        """Evaluate the predictions.

        return the total number of correct predictions.
        """
        preds = np.argmax(preds, axis=1)
        correct_preds = 0
        for pred, label in zip(preds, label_ids):
            if pred == label:
                correct_preds += 1
        return correct_preds

    @dump_elapsed_time(customized_msg="Customized eval_func")
    def eval_func(self, graph):
        # Set the config for running
        config = tf.compat.v1.ConfigProto()
        config.intra_op_parallelism_threads=ARGS.num_intra_threads
        config.inter_op_parallelism_threads=ARGS.num_inter_threads
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()

        output = graph.get_tensor_by_name('Identity:0')
        total_time = 0
        accuracy = 0
        logger.info("Started warmup for {} steps...".format(ARGS.warmup_steps))
        start_step_idx = MAX_STEPS - MAX_WARMUP_STEPS
        with tf.compat.v1.Session(graph=graph, config=config) as sess:
            # Warm up
            for step in range(start_step_idx, start_step_idx + ARGS.warmup_steps):
                feed_dict, _ = create_feed_dict_and_labels(self.dataset, idx=step)
                _ = sess.run(output, feed_dict= feed_dict)
            logger.info("Warmup completed.")
            # Inference
            logger.info("Starting inference for {} steps...".format(ARGS.steps))
            total_correct_predictions = 0
            iter = 0
            for feed_dict, labels in self.dataloader:
                iter += 1
                start_time = time.time()
                if ARGS.profile:
                    pred = sess.run(output, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                else:
                    pred = sess.run(output, feed_dict=feed_dict)
                run_time = time.time() - start_time
                if ARGS.tune or ARGS.sq or (ARGS.benchmark and ARGS.mode == "accuracy"):
                    total_correct_predictions += self.get_correct_predictions(pred, labels)
                total_time += run_time
                # save profiling file
                if ARGS.profile and iter == int(self.dataloader.num_batch / 2):
                        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                        model_dir = str(os.path.dirname(os.path.realpath(__file__))) + '/timeline'
                        if not os.path.exists(model_dir):
                            try:
                                os.makedirs(model_dir)
                            except:
                                pass
                        profiling_file = model_dir + '/timeline-' + str(iter + 1) + '-' + str(os.getpid()) + '.json'
                        with open(profiling_file, 'w') as trace_file:
                            trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
        time_per_batch = total_time / float(ARGS.steps / ARGS.batch_size)
        if ARGS.tune or ARGS.sq or (ARGS.benchmark and ARGS.mode == "accuracy"):
            accuracy = total_correct_predictions / ARGS.steps
            logger.info("Accuracy: {:.4f}".format(accuracy))
        if self.dataloader.batch_size == 1:
            logger.info("Latency: {:.4f} ms".format(time_per_batch * 1000))
        logger.info("Throughput: {:.4f} sentences/sec".format(self.dataloader.batch_size / time_per_batch))
        return accuracy

    def run(self):
        graph = self.load_graph()
        if ARGS.tune or ARGS.sq:
            from neural_compressor import quantization
            from neural_compressor.config import PostTrainingQuantConfig, AccuracyCriterion
            if ARGS.sq:
                config = PostTrainingQuantConfig(calibration_sampling_size=[500],
                                                quant_level=1,
                                                recipes={"smooth_quant": True, "smooth_quant_args": {'alpha': 0.6}})
            else:
                accuracy_criterion = AccuracyCriterion(tolerable_loss=0.02)
                config = PostTrainingQuantConfig(calibration_sampling_size=[500],
                                                 accuracy_criterion=accuracy_criterion)
            q_model = quantization.fit(model=graph, conf=config, calib_dataloader=self.dataloader,
                            eval_func=self.eval_func)
            try:
                q_model.save(ARGS.output_graph)
            except Exception as e:
                tf.compat.v1.logging.error("Failed to save model due to {}".format(str(e)))
        elif ARGS.benchmark:
            assert ARGS.mode == 'performance' or ARGS.mode == 'accuracy', \
            "Benchmark only supports performance or accuracy mode."
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            if ARGS.mode == 'performance':
                conf = BenchmarkConfig(cores_per_instance=28, num_of_instance=1)
                fit(graph, conf, b_func=self.eval_func)
            elif ARGS.mode == 'accuracy':
                self.eval_func(graph)


if __name__ == "__main__":
    distilbert_ob = Distilbert_base()
    distilbert_ob.run()
