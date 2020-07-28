"""Tests for config file"""
import unittest
import numpy as np
import yaml
import os
from ilit.conf import config as conf

def helper(content):
    y = yaml.load(content, Loader=yaml.SafeLoader)
    with open('fake_conf.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

class TestConf(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        os.remove('fake_conf.yaml')

    def test_main_key(self):
        test = '''
        framework: 
          - name: pytorch
        test: cpu
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_framework(self):
        test = '''
        framework: 
          - name: pytorch, mxnet
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')
        
        test = '''
        device: cpu
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_device(self):
        test = '''
        framework: 
          - name: mxnet
        device: xpu
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: tensorflow
        device: cpu
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        device: cpu, gpu
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_calibration(self):
        test = '''
        framework: 
          - name: mxnet
        calibration:
          - iteration: 10
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        calibration:
          - iterations: 
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        calibration:
          - algorithm: kl
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        calibration:
        algorithm: 
        - weight: kl
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        calibration:
          - algorithm: 
          - activation: minmax
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        calibration:
          - algorithm: 
            activation: minmax
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        calibration:
        algorithm: 
        activation: minmax
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_quantization(self):
        test = '''
        framework: 
          - name: mxnet
        quantization: 
          - weights:
            - granularity: per_channel
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        quantization: 
          - approach:
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        quantization: 
          - approach: post_training_static_quant, quant_aware_training
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        quantization: 
          - activation: 
              scheme: asym
              dtype: int8
          - weight: 
            - scheme: asym
            - dtype: int8            
        '''
        helper(test)
        conf.Conf('fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        quantization: 
          - activation: 
              scheme: 
              dtype: int8
          - weight: 
              scheme: asym
              dtype: int8            
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_tuning(self):
        test = '''
        framework: 
          - name: mxnet
        tuning: 
          - accuracy_criterion:
              - relative: 0.01
          - strategy: basic, mse
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        tuning: 
          - accuracy_criterion:
            relative: 0.01
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        tuning: 
           accuracy_criterion:
           relative: 0.01
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        tuning: 
          - accuracy_criterion:
              - relative: 0.01
          - strategy: fake
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        tuning: 
          - accuracy_criterion:
              - relative: 
            strategy: basic
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        tuning: 
          - accuracy_criterion:
            timeout: 3
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        framework: 
          - name: mxnet
        tuning: 
          - accuracy_criterion:
              - relative: 0.01
            ops: {
              'test': {
                  'activation': [{'dtype': 'uint8'}, {'algorithm': 'minmax'}]
              }
            }
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_snapshot(self):
        test = '''
        framework: 
          - name: mxnet
        snapshot: 
          -path: /path/to/snapshot
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')


if __name__ == "__main__":
    unittest.main()
