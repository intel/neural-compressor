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
from dlrm_data_pytorch import CriteoDataset, DLRM_DataLoader
import torch
from dlrm_data_pytorch import collate_wrapper_criteo
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32,
                        type=int, help="Batch size")
    parser.add_argument("--input_model", default="dlrm_s_pytorch.onnx",
                         type=str, help="input_model_path")
    parser.add_argument("--output_model", default="./ir/", type=str, help="output_model_path")
    parser.add_argument("--raw_path", default="./input/train.txt", type=str,
                        help="The input data set dir. Should contain the .txt files.")
    parser.add_argument("--pro_data", default="./input/kaggleAdDisplayChallenge_processed.npz", type=str,
                        help="The processed data set dir. Should contain the .npz files.")
    parser.add_argument("--config", default="./conf.yaml", type=str, help="yaml path")
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--tune', action='store_true',
                        default=False, help="whether quantize the model")
    parser.add_argument('--mode', type=str, help="benchmark mode of performance or accuracy")
    parser.add_argument('--dataset', default='kaggle', type=str, help="kaggle or terabyte")
    parser.add_argument('--mlperf_bin_loader', action='store_true', default=False)
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    raw_path = args.raw_path
    pro_data = args.pro_data
    if args.mlperf_bin_loader and args.dataset == 'terabyte':
        import criteo
        kwargs = {"randomize": 'total',  "memory_map": True}
        # --count-samples can be used to limit the number of samples used for testing
        terabyte_ds = criteo.Criteo(data_path=raw_path,
                            name=args.dataset,
                            pre_process=criteo.pre_process_criteo_dlrm,  # currently an identity function
                            use_cache=0,  # currently not used
                            batch_size=args.batch_size,
                            count=None,
                            samples_to_aggregate_fix=None,
                            samples_to_aggregate_min=None,
                            samples_to_aggregate_max=None,
                            samples_to_aggregate_quantile_file='dist_quantile.txt',
                            samples_to_aggregate_trace_file='aggregated_samples.txt',
                            test_num_workers=0,
                            max_ind_range=40000000,
                            sub_sample_rate=0.0,
                            mlperf_bin_loader=args.mlperf_bin_loader,
                            **kwargs)

        ds = terabyte_ds.test_loader
    else:
        kaggle_ds = CriteoDataset(dataset = args.dataset, max_ind_range=-1, \
              sub_sample_rate=0.0, randomize="total", split="test", \
              raw_path=raw_path, pro_data=pro_data, batch_size=args.batch_size)
        ds = torch.utils.data.DataLoader(
            kaggle_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_wrapper_criteo,
            pin_memory=False,
            drop_last=False,  # True
        )

    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(args.input_model)
        evaluator.b_dataloader = DLRM_DataLoader(ds)
        evaluator(args.mode)


    if args.tune:
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization(args.config)
        quantizer.model = common.Model(args.input_model)
        quantizer.eval_dataloader = DLRM_DataLoader(ds)
        quantizer.calib_dataloader = DLRM_DataLoader(ds)
        q_model = quantizer.fit()
        q_model.save(args.output_model)

if __name__ == '__main__':
    main()
