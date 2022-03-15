import sys
import os
import unittest
from neural_compressor.experimental import Quantization

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
                  input_shape: [[128], [128], [128]]
                  label_shape: [128,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
        
        evaluation:
          accuracy:
            metric:
              GLUE:
                task: sst-2
          performance:
            warmup: 5
            iteration: 10
            configs:
              num_of_instance: 1
              cores_per_instance: 28
        
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
                  input_shape: [[128], [128], [128]]
                  label_shape: [128,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
       evaluation:
          accuracy:
            metric:
              GLUE:
                task: sst-2
          performance:
            warmup: 5
            iteration: 10
            configs:
              num_of_instance: 1
              cores_per_instance: 28
        
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
        quantizer.model = "/home/tensorflow/inc_ut/engine/bert_mini_sst2.onnx"
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

    def test_adaptor_bf16(self):
        quantizer = Quantization('test_bf16.yaml')
        quantizer.model = "/home/tensorflow/inc_ut/engine/bert_mini_sst2.onnx"
        q_model = quantizer.fit()
        self.assertNotEqual(q_model, None)

    def tearDown(self):
        os.remove('test.yaml')
        os.remove('test_bf16.yaml')

if __name__ == "__main__":
    unittest.main()