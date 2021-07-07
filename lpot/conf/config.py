#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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

import yaml
from schema import Schema, And, Use, Optional, Or, Hook
from ..adaptor import FRAMEWORKS
from ..strategy import STRATEGIES
from ..objective import OBJECTIVES
from ..pruners import PRUNERS
from ..utils import logger
from ..version import __version__
import re
import copy
import itertools
from collections import OrderedDict
from .dotdict import DotDict
import os, datetime

def constructor_register(cls):
    yaml_key = "!{}".format(cls.__name__)

    def constructor(loader, node):
        instance = cls.__new__(cls)
        yield instance

        state = loader.construct_mapping(node, deep=True)
        instance.__init__(**state)

    yaml.add_constructor(
        yaml_key,
        constructor,
        yaml.SafeLoader,
    )
    return cls

@constructor_register
class Pruner():
    def __init__(self, start_epoch=None, end_epoch=None, initial_sparsity=None,
                 target_sparsity=None, update_frequency=1, prune_type='basic_magnitude',
                 method='per_tensor', names=[], parameters=None):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.update_frequency = update_frequency
        self.target_sparsity = target_sparsity
        self.initial_sparsity = initial_sparsity
        self.update_frequency = update_frequency
        assert prune_type.replace('_', '') in [i.lower() for i in PRUNERS], \
                                         'now only support {}'.format(PRUNERS.keys())
        self.prune_type = prune_type
        self.method = method
        self.names= names
        self.parameters = parameters

# Schema library has different loading sequence priorities for different
# value types.
# To make sure the fields under dataloader.transform field of yaml file
# get loaded with written sequence, this workaround is used to convert
# None to {} in yaml load().
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:null', lambda loader, node: {})
# Add python tuple support because best_configure.yaml may contain tuple
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple',
                                lambda loader, node: tuple(loader.construct_sequence(node)))

def _valid_accuracy_field(key, scope, error):
    assert bool(
        'relative' in scope['accuracy_criterion']) != bool(
        'absolute' in scope['accuracy_criterion'])

def _valid_prune_epoch(key, scope, error):
    if "start_epoch" in scope and "end_epoch" in scope:
        assert scope["start_epoch"] <= scope["end_epoch"]

def _valid_prune_sparsity(key, scope, error):
    if "initial_sparsity" in scope and "target_sparsity" in scope:
        assert scope["initial_sparsity"] <= scope["target_sparsity"]
    elif "initial_sparsity" in scope:
        assert scope["initial_sparsity"] >= 0
    else:
        assert scope["target_sparsity"] < 1

# used for '123.68 116.78 103.94' style to float list
def input_to_list_float(data):
    if isinstance(data, str):
        return [float(s.strip()) for s in data.split()]

    if isinstance(data, float):
        return [data]

    assert isinstance(data, list)
    return [float(d) for d in data]

def input_int_to_float(data):
    if isinstance(data, str):
        # used for '123.68, 116.78, 103.94' style
        if ',' in data:
            data = data.split(',')
        # used for '123.68 116.78 103.94' style
        else:
            data = data.split()

        if len(data) == 1:
            return float(data[0].strip())
        else:
            return [float(s.strip()) for s in data]
    elif isinstance(data, list):
        return [float(s) for s in data]
    elif isinstance(data, int):
        return float(data)

def input_to_list_int(data):
    if isinstance(data, str):
        return [int(s.strip()) for s in data.split(',')]

    if isinstance(data, int):
        return [data]

    assert isinstance(data, list)
    return [int(d) for d in data]

def input_to_list(data):
    if isinstance(data, str):
        if ',' in data:
            return [s.strip() for s in data.split(',')]

        return [s.strip() for s in data.split()]

    if isinstance(data, int):
        return [data]

    assert isinstance(data, list)
    return data

def list_to_tuple(data):
    if isinstance(data, str):
        return tuple([int(s.strip()) for s in data.split(',')])

    elif isinstance(data, list):
        if isinstance(data[0], list):
            result = []
            for item in data:
                result.append(tuple([int(s) for s in item]))
            return result
        else:
            return tuple([int(s) for s in data])

def percent_to_float(data):
    if isinstance(data, str) and re.match(r'-?\d+(\.\d+)?%', data):
        data = float(data.strip('%')) / 100
    if isinstance(data, int):
        data = float(data)
    else:
        assert isinstance(data, float), 'This field should be float, int or percent string'
    return data

ops_schema = Schema({
    Optional('weight', default=None): {
        Optional('granularity', default=None): And(
            list,
            lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
        Optional('scheme', default=None): And(
            list,
            # asym_float and placeholder is only for PyTorch framework
            lambda s: all(i in ['asym', 'sym', 'asym_float', 'placeholder'] for i in s)),
        Optional('dtype', default=None): And(
            list,
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s)),
        Optional('algorithm', default=None): And(
            list,
            lambda s: all(i in ['minmax'] for i in s)),
        Optional('bit', default=None):  And(
            Or(float, list),
            Use(input_to_list_float),
            lambda s: all(0.0 < i <= 7.0 for i in s))
    },
    Optional('activation', default=None): {
        Optional('granularity', default=None): And(
            list,
            lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
        Optional('scheme', default=None): And(
            list,
            lambda s: all(i in ['asym', 'sym'] for i in s)),
        Optional('dtype', default=None): And(
            list,
            lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s)),
        Optional('algorithm', default=None): And(
            list,
            lambda s: all(i in ['minmax', 'kl'] for i in s))
    }
})

graph_optimization_schema = Schema({

    Optional('precisions', default={'precisions': ['fp32']}): And(
        Or(str, list),
        Use(input_to_list),
        lambda s: all(i in [ 'fp32', 'bf16'] for i in s)),

    Optional('op_wise', default={'weight': {}, 'activation': {}}): {
        Optional('weight', default=None): {
            Optional('dtype', default=None): And(
                Or(str, list),
                Use(input_to_list),
                lambda s: all(i in ['fp32', 'bf16'] for i in s)),
        },
        Optional('activation', default=None): {
            Optional('dtype', default=None): And(
                Or(str, list),
                Use(input_to_list),
                lambda s: all(i in ['fp32', 'bf16'] for i in s)),
            }
    }
})

model_conversion_schema = Schema({
    'source': And(str, lambda s: s.lower() == 'qat'),
    'destination': And(str, lambda s: s.lower() == 'default')
})

filter_schema = Schema({
    Optional('LabelBalance'): {
        'size': And(int, lambda s: s > 0)
    },
})

transform_schema = Schema({
    Optional('ResizeWithRatio'):{
        Optional('min_dim'): int,
        Optional('max_dim'): int,
        Optional('padding'): bool,
        Optional('constant_value'): int
    },
    Optional('CropToBoundingBox'): {
        'offset_height': int,
        'offset_width': int,
        'target_height': int,
        'target_width': int
    },
    Optional('Cast'): {
        Optional('dtype'): str
    },
    Optional('RandomResizedCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0)),
        Optional('scale'): And(list, lambda s: all(isinstance(i, float) for i in s)),
        Optional('ratio'): And(list, lambda s: all(isinstance(i, float) for i in s)),
        Optional('interpolation'): And(
            str,
            lambda s: s in ['nearest', 'bilinear', 'bicubic']),
    },
    Optional('AlignImageChannel'): {
        Optional('dim'): int
    },
    Optional('ToNDArray'): Or({}, None),
    Optional('CropResize'): {
        'x': int,
        'y': int,
        'width': int,
        'height': int,
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0)),
        Optional('interpolation'): And(
            str,
            lambda s: s in ['nearest', 'bilinear', 'bicubic']),
    },
    Optional('RandomHorizontalFlip'): Or({}, None),
    Optional('RandomVerticalFlip'): Or({}, None),
    Optional('ToTensor'): Or({}, None),
    Optional('ToPILImage'): Or({}, None),
    Optional('Normalize'): {
        Optional('mean'): And(list, lambda s: all(isinstance(i, float) for i in s)),
        Optional('std'): And(list, lambda s: all(isinstance(i, float) for i in s))
    },
    Optional('Resize'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0)),
        Optional('interpolation'): And(
            str,
            lambda s: s in ['nearest', 'bilinear', 'bicubic']),
    },
    Optional('RandomCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0))
    },
    Optional('Rescale'): Or({}, None),
    Optional('CenterCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0))
    },
    Optional('PaddedCenterCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0)),
        Optional('crop_padding'): And(int, lambda s: s > 0),
    },
    Optional('ToArray'): Or({}, None),
    Optional('QuantizedInput'): {
        Optional('dtype', default='int8'): And(str, lambda s: s in ['int8', 'uint8']),
        Optional('scale'): And(float, lambda s: s > 0),
    },
    Optional('Transpose'): {
        'perm': And(list, lambda s: all(isinstance(i, int) for i in s)),
    },
    # THIS API IS TO BE DEPRECATED!
    Optional('ParseDecodeImagenet'): Or({}, None),
    Optional('ParseDecodeCoco'): Or({}, None),
    Optional('ParseDecodeVoc'): Or({}, None),
    Optional('BilinearImagenet'): {
        'height': And(int, lambda s: s > 0),
        'width': And(int, lambda s: s > 0),
        Optional('central_fraction'): float,
        Optional('mean_value'): And(Or(str, list), Use(input_to_list_float)),
        Optional('scale'): float,
    },
    Optional('ResizeCropImagenet'): {
        'height': And(int, lambda s: s > 0),
        'width': And(int, lambda s: s > 0),
        Optional('random_crop'): bool,
        Optional('slice_crop'): bool,
        Optional('resize_side'): And(int, lambda s: s > 0),
        Optional('random_flip_left_right'): bool,
        Optional('mean_value'): And(Or(str, list), Use(input_to_list_float)),
        Optional('scale'): float
    },
    Optional('ResizeWithAspectRatio'):{
        'height': And(int, lambda s: s > 0),
        'width': And(int, lambda s: s > 0),
    },
    Optional('ParseDecodeImagenet'): Or({}, None),
    Optional('ToArray'): Or({}, None),
    Optional('QuantizedInput'): {
        Optional('dtype', default='int8'): And(str, lambda s: s in ['int8', 'uint8']),
        Optional('scale'): And(float, lambda s: s > 0),
    },
    Optional('Transpose'): {
        'perm': And(list, lambda s: all(isinstance(i, int) for i in s)),
    },
})

postprocess_schema = Schema({
    Optional('LabelShift'):  int,
    Optional('SquadV1'): {
        'label_file': str,
        'vocab_file': str
    },
})

dataset_schema = Schema({
    Optional('CIFAR10'): {
        'root': str,
        Optional('train'): bool,
        Optional('download'): bool,
    },
    Optional('CIFAR100'): {
        'root': str,
        Optional('train'): bool,
        Optional('download'): bool,
    },
    Optional('MNIST'): {
        'root': str,
        Optional('train'): bool,
        Optional('download'): bool,
    },
    Optional('FashionMNIST'): {
        'root': str,
        Optional('train'): bool,
        Optional('download'): bool,
    },
    Optional('ImageFolder'): {
        'root': str,
    },
    Optional('TFRecordDataset'): {
        'root': str,
    },
    Optional('ImageRecord'): {
        'root': str,
    },
    Optional('dummy_v2'): {
        'input_shape': And(Or(str, list), Use(list_to_tuple)),
        Optional('label_shape'): And(Or(str, list), Use(list_to_tuple)),
        Optional('low'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('high'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('dtype'): And(Or(str, list), Use(input_to_list)),
    },
    Optional('dummy'): {
        'shape': And(Or(str, list), Use(list_to_tuple)),
        Optional('low'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('high'): Or(
            float,
            And(int, Use(input_int_to_float)),
            And(list, Use(input_int_to_float)),
            And(str, Use(input_int_to_float))),
        Optional('dtype'): And(Or(str, list), Use(input_to_list)),
        Optional('label'): bool,
    },
    Optional('bert'): {
        'root': str,
        'label_file': str,
        Optional('task'): And(str, lambda s: s in ["classifier", "squad"]),
        Optional('model_type'): And(str, lambda s: s in ['bert', 'xlnet', 'xlm']),
    },
    Optional('VOCRecord'): {
        'root': str,
    },
    Optional('COCORecord'): {
        'root': str,
        Optional('num_cores'): int,
    },
    Optional('COCORaw'): {
        'root': str,
        Optional('img_dir'): str,
        Optional('anno_dir'): str,
        Optional('num_cores'): int,
    },
    Optional('COCONpy'): {
        'root': str,
        'npy_dir': str,
        Optional('anno_dir'): str,
        Optional('num_cores'): int,
    },
    Optional('ImagenetRaw'): {
        'data_path': str,
        Optional('image_list'): str,
    },
    Optional('style_transfer'): {
        'content_folder': str,
        'style_folder': str,
        Optional('crop_ratio'): float,
        Optional('resize_shape'): And(Or(str, list), Use(input_to_list_int)),
        Optional('image_format'): str,
    },
    # TO BE DEPRECATED!
    Optional('Imagenet'): {
        'root': str,
    },
})

dataloader_schema = Schema({
    Optional('last_batch', default='rollover'): And(str, lambda s: s in ['rollover', 'discard']),
    Optional('batch_size', default=1): And(int, lambda s: s > 0),
    'dataset': dataset_schema,
    Optional('filter'): filter_schema,
    Optional('transform'): transform_schema,
    Optional('shuffle'): bool,
})

configs_schema = Schema({
    Optional('cores_per_instance'): And(int, lambda s: s > 0),
    Optional('num_of_instance', default=1): And(int, lambda s: s > 0),
    Optional('inter_num_of_threads'): And(int, lambda s: s > 0),
    Optional('intra_num_of_threads'): And(int, lambda s: s > 0),
    Optional('kmp_blocktime'): And(int, lambda s: s >= 0),
    Optional('kmp_affinity', default='granularity=fine,verbose,compact,1,0'): str,
})

optimizer_schema = Schema({
    Optional('SGD'): {
        'learning_rate': float,
        Optional('momentum'): float,
        Optional('nesterov'): bool,
        Optional('weight_decay'): float
    }
})

criterion_schema = Schema({
    Optional('CrossEntropyLoss'): {
        Optional('reduction', default='mean'): And(str, lambda s: s in ['none', 'sum', 'mean'])
    }
})

train_schema = Schema({
    'optimizer': optimizer_schema,
    'criterion': criterion_schema,
    Optional('dataloader'): dataloader_schema,
    Optional('start_epoch', default=0): int,
    Optional('end_epoch'): int,
    Optional('iteration'): int,
    Optional('frequency'): int,
    # TODO reserve for multinode training support
    Optional('hostfile'): str
})

weight_compression_schema = Schema({
    Optional('initial_sparsity', default=0): And(float, lambda s: s < 1.0 and s >= 0.0),
    Optional('target_sparsity', default=0.97): float,
    Optional('start_epoch', default=0): int,
    Optional('end_epoch', default=4): int,
    Optional('pruners'): And(list, \
                               lambda s: all(isinstance(i, Pruner) for i in s))
})

approach_schema = Schema({
    Optional('weight_compression'): weight_compression_schema,
})

default_workspace = './lpot_workspace/{}/'.format(
                                           datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

schema = Schema({
    'model': {
        'name': str,
        'framework': And(str, lambda s: s in FRAMEWORKS),
        Optional('inputs', default=[]): And(Or(str, list), Use(input_to_list)),
        Optional('outputs', default=[]): And(Or(str, list), Use(input_to_list)),
    },
    Optional('version', default=float(__version__.split('.')[0])): And(
                                          Or(float,
                                             And(int, Use(input_int_to_float)),
                                             And(str, Use(input_int_to_float))),
                                          lambda s: s == float(__version__.split('.')[0])),
    Optional('device', default='cpu'): And(str, lambda s: s in ['cpu', 'gpu']),
    Optional('quantization', default={'approach': 'post_training_static_quant', \
                                      'calibration': {'sampling_size': [100]}, \
                                      'recipes': {'scale_propagation_max_pooling': True,
                                                      'scale_propagation_concat': True,
                                                      'first_conv_or_matmul_quantization': True},
                                      'model_wise': {'weight': {'bit': [7.0]},
                                                     'activation': {}}}): {
        Optional('approach', default='post_training_static_quant'): And(
            str,
            # TODO check if framework support dynamic quantize
            # Now only onnruntime and pytorch supoort
            lambda s: s in ['post_training_static_quant',
                            'post_training_dynamic_quant',
                            'quant_aware_training']),
        Optional('train', default=None): train_schema,
        Optional('advance', default=None): {
            Optional('bias_correction'): And(str, lambda s: s in ['weight_empirical']),
        },
        Optional('calibration', default={'sampling_size': [100]}): {
            Optional('sampling_size', default=[100]): And(Or(str, int, list), Use(input_to_list)),
            Optional('dataloader', default=None): dataloader_schema
        },
        Optional('recipes', default={'scale_propagation_max_pooling': True,
                                         'scale_propagation_concat': True,
                                         'first_conv_or_matmul_quantization': True}): {
            Optional('scale_propagation_max_pooling', default=True):
                    And(bool, lambda s: s in [True, False]),
            Optional('scale_propagation_concat', default=True):
                    And(bool, lambda s: s in [True, False]),
            Optional('first_conv_or_matmul_quantization', default=True):
                    And(bool, lambda s: s in [True, False]),
            Optional('fast_bias_correction', default=False):
                    And(bool, lambda s: s in [True, False]),
            Optional('weight_correction', default=False):
                    And(bool, lambda s: s in [True, False]),
        },
        Optional('model_wise', default={'weight': {'bit': [7.0]}, 'activation': {}}): {
            Optional('weight', default= {'bit': [7.0]}): {
                Optional('granularity', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
                Optional('scheme', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['asym', 'sym'] for i in s)),
                Optional('dtype', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s)),
                Optional('algorithm', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['minmax'] for i in s)),
                Optional('bit', default=[7.0]):  And(
                    Or(float, list),
                    Use(input_to_list_float),
                    lambda s: all(0.0 < i <= 7.0 for i in s))

            },
            Optional('activation', default=None): {
                Optional('granularity', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
                Optional('scheme', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['asym', 'sym'] for i in s)),
                Optional('dtype', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s)),
                Optional('algorithm', default=None): And(
                    Or(str, list),
                    Use(input_to_list),
                    lambda s: all(i in ['minmax', 'kl'] for i in s)),
            }
        },
        Optional('op_wise', default=None): {
            str: ops_schema
        },
    },

    Optional('graph_optimization'): graph_optimization_schema,

    Optional('model_conversion'): model_conversion_schema,

    Optional('tuning', default={
        'strategy': {'name': 'basic'},
        'accuracy_criterion': {'relative': 0.01, 'higher_is_better': True},
        'objective': 'performance',
        'exit_policy': {'timeout': 0, 'max_trials': 100, 'performance_only': False},
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': default_workspace}}): {
        Optional('strategy', default={'name': 'basic'}): {
            'name': And(str, lambda s: s in STRATEGIES), Optional('sigopt_api_token'): str,
            Optional('sigopt_project_id'): str,
            Optional('sigopt_experiment_name', default='lpot-tune'): str,
            Optional('accuracy_weight', default=1.0): float,
            Optional('latency_weight', default=1.0): float
        } ,
        Hook('accuracy_criterion', handler=_valid_accuracy_field): object,
        Optional('accuracy_criterion', default={'relative': 0.01}): {
            Optional('relative'): And(Or(str, float), Use(percent_to_float)),
            Optional('absolute'): And(Or(str, int, float), Use(percent_to_float)),
            Optional('higher_is_better', default=True): bool,
        },
        Optional('objective', default='performance'): And(str, lambda s: s in OBJECTIVES),
        Optional('exit_policy', default={'timeout': 0,
                                         'max_trials': 100,
                                         'performance_only': False}): {
            Optional('timeout', default=0): int,
            Optional('max_trials', default=100): int,
            Optional('performance_only', default=False): bool,
        },
        Optional('random_seed', default=1978): int,
        Optional('tensorboard', default=False): And(bool, lambda s: s in [True, False]),
        Optional('workspace', default={'path': default_workspace}): {
            Optional('path', default=None): str,
            Optional('resume'): str
        }
    },
    Optional('evaluation'): {
        Optional('accuracy'): {
            Optional('metric', default=None): {
                Optional('topk'): And(int, lambda s: s in [1, 5]),
                Optional('mAP'): {
                    Optional('anno_path'): str,
                    Optional('iou_thrs', default=0.5):
                            Or(And(str, lambda s: s in ['0.5:0.05:0.95']),
                               And(float, lambda s: s <= 1.0 and s >= 0.0)),
                    Optional('map_points', default=0): And(int, lambda s: s in [0, 11, 101])
                },
                Optional('COCOmAP'): {
                    Optional('anno_path'): str,
                    Optional('map_key', default='DetectionBoxes_Precision/mAP'): str
                },
                Optional('VOCmAP'): {
                    Optional('anno_path'): str
                },
                Optional('SquadF1'): Or({}, None),
                Optional('MSE'): {
                    Optional('compare_label'): bool
                },
                Optional('RMSE'): {
                    Optional('compare_label'): bool
                },
                Optional('MAE'): {
                    Optional('compare_label'): bool
                },
                Optional('Accuracy'): Or({}, None),
                Optional('Loss'): Or({}, None),
                Optional('BLEU'): Or({}, None),
                Optional('SquadF1'): Or({}, None),
                Optional('F1'): Or({}, None),
                Optional('mIOU'): {
                    Optional('num_classes'): int
                },
            },
            Optional('configs'): configs_schema,
            Optional('iteration', default=-1): int,
            Optional('dataloader'): dataloader_schema,
            Optional('postprocess'): {
                Optional('transform'): postprocess_schema
            },
        },
        Optional('performance'): {
            Optional('warmup', default=5): int,
            Optional('iteration', default=-1): int,
            Optional('configs'): configs_schema,
            Optional('dataloader'): dataloader_schema,
            Optional('postprocess'): {
                Optional('transform'): postprocess_schema
            }
        },
    },
    Optional('pruning'): {
        Optional("train"): train_schema,
        Optional("approach"): approach_schema
    }
})


class Conf(object):
    """config parser.

    Args:
        cfg_fname (string): The path to the configuration file.

    """

    def __init__(self, cfg_fname):
        assert cfg_fname is not None
        self.usr_cfg = DotDict(self._read_cfg(cfg_fname))
        self._model_wise_tune_space = None
        self._opwise_tune_space = None

    def _read_cfg(self, cfg_fname):
        """Load a config file following yaml syntax.

           Args:
               cfg_fname(string): The name of configuration yaml file
        """
        try:
            with open(cfg_fname, 'r') as f:
                content = f.read()
                cfg = yaml.safe_load(content)
                validated_cfg = schema.validate(cfg)

            # if user yaml doesn't include version field, lpot will write a supported version
            # into it.
            if 'version' not in cfg:
                leading_whitespace = re.search(r"[ \t]*model\s*:",
                                               content).group().split("model")[0]
                content = re.sub(r'model\s*:',
                                 'version: {}\n\n{}model:'.format(
                                                               float(__version__.split('.')[0]),
                                                               leading_whitespace
                                                           ),
                                 content)
                with open(cfg_fname, 'w') as f:
                    f.write(content)

            return validated_cfg

        except Exception as e:
            logger.error("{}".format(e))
            raise RuntimeError(
                "The yaml file format is not correct. Please refer to document."
            )

    def _merge_dicts(self, src, dst):
        """Helper function to merge src dict into dst dict.

           If the key in src doesn't exist in dst, then add this key and value
           pair to dst.
           If the key in src is in dst and the value intersects with the one in
           dst, then override the value in dst with the intersect value.

        Args:
            src (dict): The source dict merged from
            dst (dict): The source dict merged to

        Returns:
            dict: The merged dict from src to dst
        """
        for key in src:
            if key in dst:
                if isinstance(dst[key], dict) and isinstance(src[key], dict):
                    self._merge_dicts(src[key], dst[key])
                elif dst[key] == src[key] or src[key] is None:
                    pass  # same leaf value
                else:
                    value = [value for value in src[key]
                             if value in dst[key] or isinstance(value, float)]
                    if value != []:
                        dst[key] = value
            else:
                if not isinstance(src[key], dict):
                    dst[key] = src[key]

        return dst

    def modelwise_tune_space(self, model_wise_quant):
        cfg = self.usr_cfg
        self._model_wise_tune_space = OrderedDict()
        for optype in model_wise_quant.keys():
            self._model_wise_tune_space[optype] = self._merge_dicts(cfg.quantization.model_wise,
                                                                    model_wise_quant[optype])

        return self._model_wise_tune_space

    def _weight_compute(self, combined_cfg):
        temp_set = set()
        for _, config in combined_cfg.items():
            temp_str = ''
            for part, params in config.items():
                temp_str = temp_str + part
                for _, param in params.items():
                    temp_str += str(param)
                temp_str += '_'
            temp_set.add(temp_str)
        return len(temp_set)

    def _sort_cfgs(self, combined_cfgs):
        cfgs_num = len(combined_cfgs)
        for i in range(cfgs_num):
            for j in range(cfgs_num-i-1):
                weight_a = self._weight_compute(combined_cfgs[j])
                weight_b = self._weight_compute(combined_cfgs[j+1])
                if weight_a > weight_b:
                    temp = combined_cfgs[j]
                    combined_cfgs[j] = combined_cfgs[j+1]
                    combined_cfgs[j+1] = temp
        return combined_cfgs

    def _combine_optype_quant_cfgs(self, model_wise_quant_cfgs):
        if len(model_wise_quant_cfgs) == 0:
            return []
        temp_cfgs = OrderedDict()
        for optype, cfgs in model_wise_quant_cfgs.items():
            if len(cfgs) > 0:
                temp_cfgs[optype] = copy.deepcopy(cfgs)
        keys, values = zip(*temp_cfgs.items())
        return self._sort_cfgs([dict(zip(keys, v)) for v in itertools.product(*values)])

    def opwise_tune_space(self, opwise_quant):
        def _is_regex(pattern):
            if re.match("^[A-Za-z0-9.][A-Za-z0-9_.\\-/]*$", pattern):
                return False
            return True

        opwise = copy.deepcopy(opwise_quant)
        for k, v in opwise.items():
            opwise[k] = self._merge_dicts(self._model_wise_tune_space[k[1]], opwise[k])

        cfg = self.usr_cfg
        if cfg.quantization.op_wise:
            for k, v in cfg.quantization.op_wise.items():
                is_regex = _is_regex(k)
                for k_op, _ in opwise.items():
                    if not is_regex and k == k_op[0]:
                        opwise[k_op] = self._merge_dicts(v, opwise[k_op])

                    if is_regex and re.match(k, k_op[0]):
                        opwise[k_op] = self._merge_dicts(v, opwise[k_op])

        self._opwise_tune_space = opwise
        return self._opwise_tune_space

    def expand_tune_cfgs(self, tune_space):
        """generate all possible tuning combinations for each op or model wise tuning.

        Args:
            tune_space (dict): The tuning space to be expanded.

        Returns:
            dict: The expanded tuning configs
        """
        cfg_lists = self._expand_tune_cfgs_recursively(tune_space)

        # remove unreasonable tuning combinations
        valid_cfgs = []
        quant_dtype = ['int8', 'uint8', 'int4', 'uint4']

        for cfg in cfg_lists:
            cfg = DotDict(cfg)
            dtype = cfg.activation.dtype

            if dtype not in quant_dtype:
                cfg.activation.clear()
                cfg.activation.dtype = dtype

            if 'weight' in cfg:
                dtype = cfg.weight.dtype
                if dtype not in quant_dtype:
                    cfg.weight.clear()
                    cfg.weight.dtype = dtype
                if (cfg.weight.dtype != cfg.activation.dtype and
                    cfg.weight.dtype not in quant_dtype and
                    cfg.activation.dtype not in quant_dtype) or \
                    (cfg.weight.dtype != cfg.activation.dtype and
                     cfg.weight.dtype in quant_dtype and
                     cfg.activation.dtype not in quant_dtype) or \
                    (cfg.weight.dtype != cfg.activation.dtype and
                     cfg.weight.dtype not in quant_dtype and cfg.activation.dtype in quant_dtype):
                    continue

            valid_cfgs.append(cfg)

        # remove duplicated configurations
        valid_cfgs = [cfg[0] for cfg in itertools.groupby(valid_cfgs)]
        return valid_cfgs

    def _expand_tune_cfgs_recursively(self, cfg_dict):
        """Helper function of recursively generating all combinations.

        Args:
            cfg_dict (dict): The dict of conf space.

        Returns:
            list: List containing all combinations
        """
        assert isinstance(cfg_dict, dict)
        combinations = OrderedDict()
        for key in cfg_dict:
            if isinstance(cfg_dict[key], dict):
                lists = self._expand_tune_cfgs_recursively(cfg_dict[key])
                combinations[key] = lists

        if len(combinations) != 0:
            return self._expand_tune_cfgs_recursively(combinations)

        keys, values = zip(*cfg_dict.items())
        lists = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return lists
