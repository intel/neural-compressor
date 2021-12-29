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
from utils import TF_BERTDataSet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BERT")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1,
                        type=int, help="Batch size")
    parser.add_argument("--input_model", default="./model/bert_base_mrpc.pb",
                         type=str, help="input_model_path")
    parser.add_argument("--output_model", default="./ir/", type=str, help="output_model_path")
    parser.add_argument("--vocab_file", default="./data/vocab.txt", 
                            type=str, help="vocab_file_path")
    parser.add_argument("--do_lower_case", type=bool, default=False,
                        help="vocab whether all lower case")
    parser.add_argument("--data_dir", default="./data/MRPC/", type=str,
                        help="The input data dir. Should contain the .tsv files.")
    parser.add_argument("--config", default="./bert.yaml", type=str, help="yaml path")
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
        ds = TF_BERTDataSet(args.data_dir, args.vocab_file, args.do_lower_case)
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(args.input_model)
        evaluator.b_dataloader = common.DataLoader(ds, args.batch_size)
        evaluator(args.mode)

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        ds = TF_BERTDataSet(args.data_dir, args.vocab_file, args.do_lower_case)
        quantizer = Quantization(args.config)
        quantizer.model = common.Model(args.input_model)
        quantizer.eval_dataloader = common.DataLoader(ds, args.batch_size)
        quantizer.calib_dataloader = common.DataLoader(ds, args.batch_size)
        q_model = quantizer.fit()
        q_model.save(args.output_model)

if __name__ == '__main__':
    main()
