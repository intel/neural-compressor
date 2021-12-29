"""
This is a sample stub of loadgen with multiple processes support.
Each process sets its affinity by a proc list.

Loadgen is a producer, which calls issue_queries(). issue_queries() gets query
from loadgen and puts query id/sample indices into an input queue.

Each Consumer(process)'s run() reads input queue, calls model_predict() to get
inference result, and put result into output queue.

A standalone thread's response_loadgen() reads output queue, and responds
inference result to loadgen.

Server and Offline scenario PerformanceOnly mode are verified.

Each Model needs to implement below
model_predict()
load_query_samples()
unload_query_samples()

For model_predict(), how to return data to loadgen is model specific, the
loadgen CPP API requires a data pointer and length, then it saves the data to
mlperf_log_accuracy.json, which is used to generate accuracy number offline.
"""

import multiprocessing
import threading
import subprocess
import time
import os
import sys
import argparse
import array
import logging
import numpy as np
import time
import tensorflow as tf
from utils import TF_BERTDataSet
import collections
from eval_util import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TensoFlow-BERT")

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Batch size")
    parser.add_argument("--input_model", default="build/data/model.pb",
                        type=str, help="Input model path")
    parser.add_argument("--output_model", default="ir/", type=str, help="Output model path")
    parser.add_argument("--vocab_file", default="build/data/vocab.txt",
                         type=str, help="vocab_file_path")
    parser.add_argument("--perf_count", default=None, help="perf count")
    parser.add_argument("--data_dir", default="build/data/dev-v1.1.json",
                        help="Path to validation data")
    parser.add_argument("--features_cache_file", default="eval_features.pickle",
                        help="Path to features' cache file")
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="vocab whether all lower case")
    parser.add_argument("--config", default="bert.yaml", type=str, help="yaml")
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true',
                        default=False, help="whether quantize the model")
    parser.add_argument('--mode', type=str, help="benchmark mode of performance or accuracy")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        ds = TF_BERTDataSet(args.data_dir, args.vocab_file, args.do_lower_case, args.perf_count)
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(args.input_model)
        evaluator.b_dataloader = common.DataLoader(ds, args.batch_size)
        evaluator(args.mode)

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        ds = TF_BERTDataSet(args.data_dir, args.vocab_file, args.do_lower_case, args.perf_count)
        quantizer = Quantization(args.config)
        quantizer.model = common.Model(args.input_model)
        quantizer.eval_dataloader = common.DataLoader(ds, args.batch_size)
        quantizer.calib_dataloader = common.DataLoader(ds, args.batch_size)
        q_model = quantizer.fit()
        q_model.save(args.output_model)

if __name__ == '__main__':
    main()

