# coding=utf-8
# Copyright (c) 2020 INTEL CORPORATION. All rights reserved.
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

from brats_QSL import get_brats_QSL

from openvino.inference_engine import IECore
from scipy.special import softmax


class _3DUNET_OV_SUT():
    def __init__(self, model_path, preprocessed_data_dir, performance_count):
        print("Loading OV model...")

        model_xml = model_path
        model_bin = os.path.splitext(model_xml)[0] + '.bin'

        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)

        self.input_name = next(iter(net.inputs))

        # After model conversion output name could be any
        # So we are looking for output with max number of channels
        max_channels = 0
        for output in net.outputs:
            if max_channels < net.outputs[output].shape[-1]:
                _3DUNET_OV_SUT.output_name = output

        self.exec_net = ie.load_network(network=net, device_name='CPU')

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries,
                                   self.process_latencies)
        self.qsl = get_brats_QSL(preprocessed_data_dir, performance_count)
        print("Finished constructing SUT.")

    def issue_queries(self, query_samples):
        for i in range(len(query_samples)):
            data = self.qsl.get_features(query_samples[i].index)

            print("Processing sample id {:d} with shape = {:}".format(
                query_samples[i].index, data.shape))

            output = self.exec_net.infer(
                inputs={self.input_name: data[np.newaxis, ...]})[
                _3DUNET_OV_SUT.output_name].astype(np.float16)

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0],
                                              bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass


def get_ov_sut(model_path, preprocessed_data_dir, performance_count):
    return _3DUNET_OV_SUT(model_path, preprocessed_data_dir, performance_count)