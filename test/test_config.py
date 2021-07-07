"""Tests for config file"""
import unittest
import os
from lpot.conf import config as conf


def helper(content):
    with open('fake_conf.yaml', 'w', encoding="utf-8") as f:
        f.write(content)


class TestConf(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        os.remove('fake_conf.yaml')

    def test_main_key(self):
        test = '''
        model:
          name: main_key_yaml
          framework: pytorch
        test: cpu
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_framework(self):
        test = '''
        model:
          name: framework_yaml 
          framework: pytorch, mxnet
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
        model:
          name: device_yaml 
          framework: mxnet
        device: xpu
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: device_yaml 
          framework: mxnet
        device: cpu, gpu
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_version(self):
        test = '''
        version: 2.0

        model:
          name: version_yaml 
          framework: mxnet
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: version_yaml 
          framework: mxnet
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')
        self.assertEqual(config.usr_cfg.version, 1.0)

        test = '''
        version: 1.0

        model:
          name: version_yaml 
          framework: mxnet
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')
        self.assertEqual(config.usr_cfg.version, 1.0)

    def test_calibration(self):
        test = '''
        model:
          name: calib_yaml 
          framework: mxnet
        quantization:
          calibration:
            sampling_sizes: 10
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: calib_yaml 
          framework: mxnet
        quantization:
          calibration:
            sampling_size:
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: calib_yaml 
          framework: mxnet
        quantization:
          calibration:
            dataloader:
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: calib_yaml 
          framework: mxnet
        quantization:
          calibration:
          op_wise: {
            'test': {
                'activation': [{'dtype': 'uint8'}, {'algorithm': 'minmax'}]
            }
          }

        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_quantization(self):
        test = '''
        model:
          name: quant_yaml 
          framework: mxnet
        quantization:
          model_wise:
            weights:
            granularity: per_channel
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: quant_yaml 
          framework: mxnet
        quantization:
          model_wise:
          approach:
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: quant_yaml 
          framework: mxnet
        quantization:
          approach: post_training_static_quant, quant_aware_training
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: quant_yaml 
          framework: mxnet
        quantization:
          model_wise:
            activation:
              scheme: asym
              dtype: int8
            weight:
              scheme: asym
              dtype: int8
        '''
        helper(test)
        conf.Conf('fake_conf.yaml')

        test = '''
        model:
          name: quant_yaml 
          framework: mxnet
        quantization:
          model_wise:
            activation:
              scheme:
              dtype: int8
            weight:
              scheme: asym
              dtype: int8
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_tuning(self):
        test = '''
        model:
          name: tuning_yaml 
          framework: mxnet
        tuning:
          accuracy_criterion:
            relative: 0.01
          strategy:
            name: basic, mse
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: tuning_yaml 
          framework: mxnet
        tuning:
          accuracy_criterion:
          relative: 0.01
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: tuning_yaml 
          framework: mxnet
        tuning:
          accuracy_criterion:
          relative: 0.01
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: tuning_yaml 
          framework: mxnet
        tuning:
          accuracy_criterion:
            relative: 0.01
          strategy:
            name: fake
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: tuning_yaml 
          framework: mxnet
        tuning:
          accuracy_criterion:
            relative:
          strategy:
            name: basic
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: tuning_yaml 
          framework: mxnet
        tuning:
          accuracy_criterion:
          exit_policy:
            timeout: 3
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

        test = '''
        model:
          name: tuning_yaml 
          framework: mxnet
        tuning:
          accuracy_criterion:
            relative: 0.01
            absolute: 0.01
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_workspace(self):
        test = '''
        model:
          name: workspace_yaml 
          framework: mxnet
        tuning:
          workspace: 
            -path: ./workspace
        '''
        helper(test)
        self.assertRaises(RuntimeError, conf.Conf, 'fake_conf.yaml')

    def test_inputs_outputs(self):
        test = '''
        model:
          name: inout_yaml 
          framework: mxnet
          inputs: x, y
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')
        self.assertEqual(config.usr_cfg.model.inputs, ['x', 'y'])

    def test_modelwise_conf_merge(self):
        test = '''
        model:
          name: inout_yaml 
          framework: mxnet
        quantization:
          model_wise:
            weight:
              algorithm:  minmax
            activation:
              algorithm:  minmax
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')

        framework_modelwise_capability = {
            'CONV2D': {
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
            },
        }

        tune_space = config.modelwise_tune_space(framework_modelwise_capability)
        self.assertEqual(tune_space['CONV2D']['activation']['algorithm'], ['minmax'])

    def test_ops_override(self):
        test = '''
        model:
          name: ops_override_yaml 
          framework: mxnet
        quantization:
          op_wise: {
            'conv1': {
              'activation':  {'dtype': ['uint8', 'fp32'], 'algorithm': ['minmax'], 'scheme':['sym']},
              'weight': {'dtype': ['int8', 'fp32'], 'algorithm': ['minmax']}
            },
            'conv2': {
              'activation':  {'dtype': ['fp32']},
              'weight': {'dtype': ['fp32']}
            }
          }
        tuning:
          accuracy_criterion:
            relative: 0.01
          objective: performance
          
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')

        framework_modelwise_capability = {
            'CONV2D': {
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

    def test_prune(self):
        test = '''
        model:
          name: imagenet_prune
          framework: pytorch
        
        pruning:
          train:
            start_epoch: 0
            end_epoch: 4
            dataloader:
              batch_size: 30
              dataset:
                ImageFolder:
                  root: /path/to/training/dataset
            optimizer:
              SGD:
                learning_rate: 0.1
                momentum: 0.1   
                nesterov: True
                weight_decay: 0.1     
            criterion:
              CrossEntropyLoss:
                reduction: sum
          approach:
            weight_compression:
              initial_sparsity: 0.0
              target_sparsity: 0.97
              pruners:
                - !Pruner
                    start_epoch: 1
                    end_epoch: 3
                    names: ['layer1.0.conv1.weight']
        
                - !Pruner
                    start_epoch: 0
                    end_epoch: 4
                    target_sparsity: 0.6
                    update_frequency: 2
                    names: ['layer1.0.conv2.weight']
        '''
        helper(test)
        config = conf.Conf('fake_conf.yaml')

    def test_data_type(self):
        test = '''
        model:
          name: test
          framework: tensorflow

        quantization:
          calibration:
            sampling_size: 20
            dataloader:
              batch_size: 1
              dataset:
                dummy:
                  shape: [[224,224], [256,256]]
                  high: [128., 127]
                  low: 1
                  dtype: ['float32', 'int8']
        '''
        helper(test)
        cfg = conf.Conf('fake_conf.yaml').usr_cfg
        dataset = cfg['quantization']['calibration']['dataloader']['dataset']['dummy']
        self.assertTrue(isinstance(dataset['shape'][0], tuple))
        self.assertTrue(isinstance(dataset['shape'], list))
        self.assertTrue(isinstance(dataset['high'][1], float))
        self.assertTrue(isinstance(dataset['high'][0], float))
        self.assertTrue(isinstance(dataset['low'], float))

        test = '''
        model:
          name: test
          framework: tensorflow

        quantization:
          calibration:
            sampling_size: 20
            dataloader:
              batch_size: 1
              dataset:
                dummy:
                  shape: [224,224]
                  high: 128
                  low: 0.1
                  dtype: ['float32', 'int8']
        '''
        helper(test)
        cfg = conf.Conf('fake_conf.yaml').usr_cfg
        dataset = cfg['quantization']['calibration']['dataloader']['dataset']['dummy']
        self.assertTrue(isinstance(dataset['shape'], tuple))
        self.assertTrue(isinstance(dataset['high'], float)) 

        test = '''
        model:
          name: test
          framework: tensorflow

        quantization:
          calibration:
            sampling_size: 20
            dataloader:
              batch_size: 1
              dataset:
                style_transfer:
                  content_folder: test
                  style_folder: test
                  crop_ratio: 0.5
                  resize_shape: 10,10
              transform:
                RandomResizedCrop:
                  size: 10
                  scale: [0.07, 0.99]
                  ratio: [0.6, 0.8]
        '''
        helper(test)
        cfg = conf.Conf('fake_conf.yaml').usr_cfg
        shape_cfg = cfg['quantization']['calibration']['dataloader']['dataset']['style_transfer']['resize_shape']
        self.assertTrue(isinstance(shape_cfg, list)) 
        transform_cfg = cfg['quantization']['calibration']['dataloader']['transform']['RandomResizedCrop']
        self.assertTrue(isinstance(transform_cfg['scale'], list))
        self.assertTrue(isinstance(transform_cfg['ratio'], list))

        test = '''
        model:
          name: test
          framework: tensorflow

        quantization:
          calibration:
            sampling_size: 20
            dataloader:
              batch_size: 1
              dataset:
                style_transfer:
                  content_folder: test
                  style_folder: test
                  crop_ratio: 0.5
                  resize_shape: [10,10]
        '''
        helper(test)
        cfg = conf.Conf('fake_conf.yaml').usr_cfg
        shape_cfg = cfg['quantization']['calibration']['dataloader']['dataset']['style_transfer']['resize_shape']
        self.assertTrue(isinstance(shape_cfg, list)) 

        test = '''
        model:
          name: test
          framework: tensorflow

        quantization:
          calibration:
            sampling_size: 20
            dataloader:
              batch_size: 1
              dataset:
                dummy:
                  shape: [224,224]
              transform:
                BilinearImagenet:
                  height: 224
                  width: 224
                  mean_value: 123.68 116.78 103.94
        '''
        helper(test)
        cfg = conf.Conf('fake_conf.yaml').usr_cfg
        shape_cfg = cfg['quantization']['calibration']['dataloader']['dataset']['dummy']['shape']
        self.assertTrue(isinstance(shape_cfg, tuple)) 
        transform_cfg = cfg['quantization']['calibration']['dataloader']['transform']['BilinearImagenet']
        self.assertTrue(isinstance(transform_cfg['mean_value'], list))


if __name__ == "__main__":
    unittest.main()
