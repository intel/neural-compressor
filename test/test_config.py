"""Tests for config file"""
import unittest
import numpy as np
import yaml
import os
from ilit.conf import config as conf

def helper(content):
    with open('fake_conf.yaml', 'w', encoding="utf-8") as f:
        f.write(content)

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

    def test_inputs_outputs(self):
        test = '''
        framework: 
          - name: mxnet
            inputs: x, y
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')
        self.assertEqual(config.usr_cfg.framework.inputs, ['x', 'y'])

    def test_modelwise_conf_merge(self):
        test = '''
        framework:
          - name: mxnet
        calibration:
          - algorithm:
              - weight:  minmax
                activation: minmax
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')

        framework_modelwise_capability = {
            'activation': {
                'dtype': ['uint8', 'fp32'],
                'scheme': ['asym', 'sym'],
                'granularity': ['per_tensor'],
                'algorithm': ['minmax', 'kl']
            },
            'weight': {
                'dtype': ['int8', 'fp32'],
                'scheme': [
                    'sym',
                ],
                'granularity': ['per_channel', 'per_tensor'],
                'algorithm': ['minmax']
            },
        }

        tune_space = config.modelwise_tune_space(framework_modelwise_capability)
        self.assertEqual(tune_space['activation']['algorithm'], ['minmax'])

    def test_ops_override(self):
        test = '''
        framework:
          - name: mxnet
        tuning:
          - ops: {
                'conv1': {
                  'activation':  {'dtype': ['uint8', 'fp32'], 'algorithm': ['minmax'], 'scheme':['sym']},
                  'weight': {'dtype': ['int8', 'fp32'], 'algorithm': ['kl']}
                },
                'conv2': {
                  'activation':  {'dtype': ['fp32']},
                  'weight': {'dtype': ['fp32']}
                }
            }
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')

        framework_modelwise_capability = {
            'activation': {
                'dtype': ['uint8', 'fp32'],
                'scheme': ['asym', 'sym'],
                'granularity': ['per_tensor'],
                'algorithm': ['minmax', 'kl']
            },
            'weight': {
                'dtype': ['int8', 'fp32'],
                'scheme': [
                    'sym',
                ],
                'granularity': ['per_channel', 'per_tensor'],
                'algorithm': ['minmax']
            },
        }

        config.modelwise_tune_space(framework_modelwise_capability)

        framework_opwise_capability = {
            ('conv1', 'CONV2D'): {
            'activation': {
                'dtype': ['uint8', 'fp32'],
                'scheme': ['asym', 'sym'],
                'granularity': ['per_tensor'],
                'algorithm': ['minmax', 'kl']
            },
            'weight': {
                'dtype': ['int8', 'fp32'],
                'scheme': [
                    'sym',
                ],
                'granularity': ['per_channel', 'per_tensor'],
                'algorithm': ['minmax']
            }},
            ('conv2', 'CONV2D'): {
            'activation': {
                'dtype': ['uint8', 'fp32'],
                'scheme': ['asym', 'sym'],
                'granularity': ['per_tensor'],
                'algorithm': ['minmax', 'kl']
            },
            'weight': {
                'dtype': ['int8', 'fp32'],
                'scheme': [
                    'sym',
                ],
                'granularity': ['per_channel', 'per_tensor'],
                'algorithm': ['minmax']
            }},
        }

        tune_space = config.opwise_tune_space(framework_opwise_capability)
        self.assertEqual(tune_space[('conv1', 'CONV2D')]['weight']['algorithm'], ['minmax'])
        self.assertEqual(tune_space[('conv2', 'CONV2D')]['activation']['dtype'], ['fp32'])

if __name__ == "__main__":
    unittest.main()
