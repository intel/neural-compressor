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

import os
import pickle
import sys
sys.path.insert(0, os.getcwd())

import mlperf_loadgen as lg

sys.path.insert(0, os.path.join(os.getcwd(), "nnUnet"))
from nnUnet.nnunet.inference.predict import preprocess_multithreaded

class BraTS_2019_QSL():
    def __init__(self, preprocessed_data_dir, perf_count):
        print("Constructing QSL...")
        self.preprocessed_data_dir = preprocessed_data_dir
        with open(os.path.join(self.preprocessed_data_dir, "preprocessed_files.pkl"), "rb") as f:
            self.preprocess_files = pickle.load(f)

        self.count = len(self.preprocess_files)
        self.perf_count = perf_count if perf_count is not None else self.count
        print("Found {:d} preprocessed files".format(self.count))
        print("Using performance count = {:d}".format(self.perf_count))

        self.loaded_files = {}
        self.qsl = lg.ConstructQSL(self.count, self.perf_count, self.load_query_samples, self.unload_query_samples)
        print("Finished constructing QSL.")

    def load_query_samples(self, sample_list):
        for sample_id in sample_list:
            file_name = self.preprocess_files[sample_id]
            print("Loading file {:}".format(file_name))
            with open(os.path.join(self.preprocessed_data_dir, "{:}.pkl".format(file_name)), "rb") as f:
                self.loaded_files[sample_id] = pickle.load(f)[0]

    def unload_query_samples(self, sample_list):
        for sample_id in sample_list:
            del self.loaded_files[sample_id]

    def get_features(self, sample_id):
        return self.loaded_files[sample_id]

def get_brats_QSL(preprocessed_data_dir="build/preprocessed_data", perf_count=None):
    return BraTS_2019_QSL(preprocessed_data_dir, perf_count)
