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
from typing import Callable, Dict, List, Optional, Union
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from functools import partial
from optimum.exporters.onnx import main_export

from neural_compressor_ort.common import Logger
from neural_compressor_ort.common.base_tuning import Evaluator, TuningConfig
from neural_compressor_ort.quantization import (
    AWQConfig,
    CalibrationDataReader,
    GPTQConfig,
    RTNConfig,
    SmoothQuantConfig,
    autotune,
)

logger = Logger().get_logger()


def fake_eval(model, eval_result_lst):
    acc = eval_result_lst.pop(0)
    return acc

def _create_evaluator_for_eval_fns(eval_fns: Optional[Union[Callable, Dict, List[Dict]]] = None) -> Evaluator:
    evaluator = Evaluator()
    evaluator.set_eval_fn_registry(eval_fns)
    return evaluator


class DataReader(CalibrationDataReader):
    def __init__(self, model):
        model = onnx.load(model)
        batch_size = 1
        sequence_length = 1
        self.data = {
            "input_ids": np.random.randint(10, size=(batch_size, sequence_length)).astype("int64"),
            "attention_mask": np.zeros((batch_size, sequence_length)).astype("int64"),
        }
        for inp in model.graph.input:
            if inp.name in self.data:
                continue
            if inp.name == "position_ids":
                # model is exported with optimum >= 1.14.0 with new input 'position_ids'
                self.data[inp.name] = np.random.randint(10, size=(batch_size, sequence_length)).astype("int64")

        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([self.data])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


class TestONNXRT3xAutoTune(unittest.TestCase):
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

    @patch("logging.Logger.warning")
    def test_auto_tune_warning(self, mock_warning):
        acc_data = iter([1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        def eval_acc_fn(model) -> float:
            session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            return next(acc_data)

        custom_tune_config = TuningConfig(config_set=[SmoothQuantConfig(alpha=0.5), SmoothQuantConfig(alpha=0.6)])
        with self.assertRaises(SystemExit):
            best_model = autotune(
                model_input=self.gptj,
                tune_config=custom_tune_config,
                eval_fn=eval_acc_fn,
                calibration_data_reader=self.data_reader,
            )
        call_args_list = mock_warning.call_args_list
        # There may be multiple calls to warning, so we need to check all of them
        self.assertIn(
            "Please refine your eval_fn to accept model path (str) as input.", [info[0][0] for info in call_args_list]
        )

    def test_sq_auto_tune(self):
        acc_data = iter([1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        perf_data = iter([1.0, 0.9, 0.99])

        def eval_perf_fn(model) -> float:
            return next(perf_data)

        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]

        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = TuningConfig(config_set=[SmoothQuantConfig(alpha=0.5), SmoothQuantConfig(alpha=0.6)])
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_acc_fn,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNotNone(best_model)

        custom_tune_config = TuningConfig(config_set=[SmoothQuantConfig(alpha=[0.5, 0.6])])
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calibration_data_reader=self.data_reader,
        )
        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)

    def test_rtn_auto_tune(self):
        acc_data = iter([1.0, 0.8, 0.6, 1.0, 0.99, 0.9])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        perf_data = iter([1.0, 0.99, 0.99])

        def eval_perf_fn(model) -> float:
            return next(perf_data)

        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]

        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = TuningConfig(config_set=[RTNConfig(weight_group_size=32), RTNConfig(weight_group_size=64)])
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_acc_fn,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNone(best_model)

        custom_tune_config = TuningConfig(config_set=[RTNConfig(weight_group_size=[32, 64])])
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calibration_data_reader=self.data_reader,
        )
        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)
        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(4, 32))
        ]
        self.assertTrue(len(op_names) > 0)

    def test_awq_auto_tune(self):
        acc_data = iter([1.0, 0.8, 0.6, 1.0, 0.99, 0.9])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        perf_data = iter([1.0, 0.99, 0.99])

        def eval_perf_fn(model) -> float:
            return next(perf_data)

        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]

        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = TuningConfig(config_set=[AWQConfig(weight_group_size=32), AWQConfig(weight_group_size=64)])
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_acc_fn,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNone(best_model)

        custom_tune_config = TuningConfig(config_set=[AWQConfig(weight_group_size=[32, 64])])
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calibration_data_reader=self.data_reader,
        )
        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)
        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(4, 32))
        ]
        self.assertTrue(len(op_names) > 0)

    def test_gptq_auto_tune(self):
        acc_data = iter([1.0, 0.8, 0.6, 1.0, 0.99, 0.9])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        perf_data = iter([1.0, 0.99, 0.99])

        def eval_perf_fn(model) -> float:
            return next(perf_data)

        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]
        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = TuningConfig(
            config_set=[GPTQConfig(weight_group_size=32), GPTQConfig(weight_group_size=64)]
        )
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_acc_fn,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNone(best_model)

        custom_tune_config = TuningConfig(config_set=[GPTQConfig(weight_group_size=[32, 64])])
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=eval_fn_wrapper,
            calibration_data_reader=self.data_reader,
        )
        self.assertEqual(len(evaluator.eval_fn_registry), 2)
        self.assertIsNotNone(best_model)
        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(4, 32))
        ]
        self.assertTrue(len(op_names) > 0)

    def test_woq_auto_tune(self):
        from neural_compressor_ort.quantization import RTNConfig, AWQConfig, GPTQConfig
        partial_fake_eval = partial(fake_eval, eval_result_lst=[1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        custom_tune_config = TuningConfig(config_set=[RTNConfig(weight_bits=4), AWQConfig(weight_bits=8)])
        best_model = autotune(
            model_input=self.gptj,
            tune_config=custom_tune_config,
            eval_fn=partial_fake_eval,
            calibration_data_reader=self.data_reader,
        )
        self.assertIsNotNone(best_model)
        op_names = [
            i.name
            for i in best_model.graph.node
            if i.op_type.startswith("MatMul") and i.input[1].endswith("_Q{}G{}".format(8, 32))
        ]
        self.assertTrue(len(op_names) > 0)

if __name__ == "__main__":
    unittest.main()
