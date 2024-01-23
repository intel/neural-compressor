#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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

import glob
import os
import shutil
import unittest

import numpy as np
import onnx
from optimum.exporters.onnx import main_export

from neural_compressor.common import Logger
from neural_compressor.onnxrt import SmoohQuantQuantConfig, get_default_sq_config
from neural_compressor.onnxrt.quantization import CalibrationDataReader
from neural_compressor.onnxrt.quantization.quantize import _quantize

logger = Logger().get_logger()


class DataReader(CalibrationDataReader):
    def __init__(self, model):
        model = onnx.load(model)
        batch_size = 1
        past_sequence_length = 1
        self.data = {
            "input_ids": np.random.randint(10, size=(batch_size, 1)).astype("int64"),
            "attention_mask": np.zeros((batch_size, past_sequence_length + 1)).astype("int64"),
        }
        for inp in model.graph.input:
            if inp.name in self.data:
                continue
            self.data[inp.name] = np.random.random((batch_size, 4, past_sequence_length, 8)).astype("float32")
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([self.data])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class TestONNXRT3xSmoothQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        main_export(
            "hf-internal-testing/tiny-random-gptj",
            output="gptj",
        )
        self.gptj = glob.glob(os.path.join("./gptj", "*.onnx"))[0]
        self.data_reader = DataReader(self.gptj)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./gptj", ignore_errors=True)

    def test_sq_from_class_beginner(self):
        self.data_reader.rewind()
        config = get_default_sq_config()
        model = _quantize(self.gptj, config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 15)

    def test_sq_auto_tune_from_class_beginner(self):
        self.data_reader.rewind()
        config = SmoohQuantQuantConfig(alpha="auto")
        model = _quantize(self.gptj, config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 15)

    def test_sq_from_dict_beginner(self):
        config = {
            "smooth_quant": {
                "global": {
                    "alpha": 0.5,
                },
            }
        }
        self.data_reader.rewind()
        model = _quantize(self.gptj, config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 15)

    def test_sq_auto_tune_from_dict_beginner(self):
        config = {
            "smooth_quant": {
                "global": {
                    "alpha": "auto",
                },
            }
        }
        self.data_reader.rewind()
        model = _quantize(self.gptj, config, self.data_reader)
        num_muls = len([i for i in model.graph.node if i.name.endswith("_smooth_mul") and i.op_type == "Mul"])
        self.assertEqual(num_muls, 15)


if __name__ == "__main__":
    unittest.main()
