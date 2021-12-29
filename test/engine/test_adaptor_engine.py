import sys
import os
import unittest
import numpy as np
from neural_compressor.experimental import Quantization, Benchmark, common
from neural_compressor.model.engine_model import EngineModel
from neural_compressor.data import DATASETS

def build_yaml():
    fake_yaml = """
        model:
          name: bert
          framework: engine
        quantization:
          calibration:
            sampling_size: 10
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
          op_wise: {
            'bert/encoder/layer_0/attention/self/query/BiasAdd':{
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32'], 'granularity': ['per_tensor']}
            }
          }
        evaluation:
          accuracy:
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
            postprocess:
              transform:
                LabelShift: -1
            metric:
              MSE:
                compare_label: False
          performance:
            iteration: 10
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
        tuning:
          exit_policy:
            max_trials: 1
    """
    with open("test.yaml",  "w", encoding="utf-8") as f:
        f.write(fake_yaml)

    fake_yaml_bf16 = """
        model:
          name: bert
          framework: engine
        quantization:
          dtype: bf16
          calibration:
            sampling_size: 10
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
          op_wise: {
            'bert/encoder/layer_0/attention/self/query/BiasAdd':{
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32'], 'granularity': ['per_tensor']}
            }
          }
        evaluation:
          accuracy:
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
            postprocess:
              transform:
                LabelShift: -1
            metric:
              MSE:
                compare_label: False
          performance:
            iteration: 10
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
        tuning:
          exit_policy:
            max_trials: 1
    """
    with open("test_bf16.yaml",  "w", encoding="utf-8") as f:
        f.write(fake_yaml_bf16)


class TestDeepengineAdaptor(unittest.TestCase):
    def setUp(self):
        build_yaml()

    def test_adaptor(self):
        quantizer = Quantization('test.yaml')
        quantizer.model = "/home/tensorflow/test-engine/bert_mlperf_2none.pb"
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

    def test_adaptor_bf16(self):
        quantizer = Quantization('test_bf16.yaml')
        quantizer.model = "/home/tensorflow/test-engine/bert_mlperf_2none.pb"
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

    def tearDown(self):
        os.remove('test.yaml')
        os.remove('test_bf16.yaml')

if __name__ == "__main__":
    unittest.main()
