# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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

import array
import json
import os
import sys
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg
import numpy as np
import onnxruntime

from brats_QSL import get_brats_QSL

class _3DUNET_ONNXRuntime_SUT():
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        print("Loading ONNX model...")
        self.sess = onnxruntime.InferenceSession(model_path)

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        self.qsl = get_brats_QSL(preprocessed_data_dir, performance_count)
        print("Finished constructing SUT.")

    def issue_queries(self, query_samples):
        for i in range(len(query_samples)):
            data = self.qsl.get_features(query_samples[i].index)

            print("Processing sample id {:d} with shape = {:}".format(query_samples[i].index, data.shape))

            # Follow the PyTorch implementation.
            # The ONNX file has five outputs, but we only care about the one named "output".
            output = self.sess.run(["output"], {"input": data[np.newaxis, ...]})[0].squeeze(0).astype(np.float16)

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

def get_onnxruntime_sut(model_path, preprocessed_data_dir, performance_count):
    return _3DUNET_ONNXRuntime_SUT(model_path, preprocessed_data_dir, performance_count)