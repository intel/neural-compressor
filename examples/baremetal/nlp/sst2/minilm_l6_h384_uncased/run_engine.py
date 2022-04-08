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
from utils import SST2DataSet

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("MINILM")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--input_model", default="minilm_l6_h384_uncased_sst2.onnx", type=str, help="input_model_path")
    parser.add_argument("--output_model", default="./ir/", type=str, help="output_model_path")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir.")
    parser.add_argument("--tokenizer_dir", default= \
                        "philschmid/MiniLM-L6-H384-uncased-sst2", type=str,
                        help="pre-trained model tokenizer name or path")
    parser.add_argument("--config", default="./bert_static.yaml", type=str, help="yaml path")
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true', default=False, help="whether quantize the model")
    parser.add_argument('--mode', type=str, help="benchmark mode of performance or accuracy")
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        ds = SST2DataSet(args.data_dir, args.tokenizer_dir)
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(args.input_model)
        evaluator.b_dataloader = common.DataLoader(ds, args.batch_size)
        evaluator(args.mode)

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        ds = SST2DataSet(args.data_dir, args.tokenizer_dir)
        quantizer = Quantization(args.config)
        quantizer.model = common.Model(args.input_model)
        quantizer.eval_dataloader = common.DataLoader(ds, args.batch_size)
        quantizer.calib_dataloader = common.DataLoader(ds, args.batch_size)
        q_model = quantizer.fit()
        q_model.save(args.output_model)


if __name__ == '__main__':
    main()
