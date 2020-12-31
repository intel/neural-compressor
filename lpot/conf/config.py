#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
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
from ..utils import logger
import re
import copy
import itertools
from collections import OrderedDict
from .dotdict import DotDict


# Schema library has different loading sequence priorities for different
# value types.
# To make sure the fields under dataloader.transform field of yaml file
# get loaded with written sequence, this workaround is used to convert
# None to {} in yaml load().
yaml.add_constructor('tag:yaml.org,2002:null', lambda loader, node: {})

def _valid_accuracy_field(key, scope, error):
    assert bool(
        'relative' in scope['accuracy_criterion']) != bool(
        'absolute' in scope['accuracy_criterion'])

def _valid_prune_epoch(key, scope, error):
    if "start_epoch" in scope and "end_epoch" in scope:
        assert scope["start_epoch"] <= scope["end_epoch"]

def _valid_prune_sparsity(key, scope, error):
    if "init_sparsity" in scope and "target_sparsity" in scope:
        assert scope["init_sparsity"] <= scope["target_sparsity"]
        if "start_epoch" in scope and "end_epoch" in scope:
            if scope["start_epoch"] == scope["end_epoch"]:
                assert scope["init_sparsity"] == scope["target_sparsity"]
    elif "init_sparsity" in scope:
        assert scope["init_sparsity"] >= 0
    else:
        assert scope["target_sparsity"] < 1

# used for '123.68 116.78 103.94' style to float list
def input_to_list_float(data):
    if isinstance(data, str):
        return [float(s.strip()) for s in data.split()]
    if isinstance(data, float):
        return [data]
    else:
        assert isinstance(data, list)
        return [float(d) for d in data]

def input_to_list(data):
    if isinstance(data, str):
        return [s.strip() for s in data.split(',')]
    if isinstance(data, int):
        return [data]
    else:
        assert isinstance(data, list)
        return data

def percent_to_float(data):
    if isinstance(data, str) and re.match(r'-?\d+(\.\d+)?%', data):
        data = float(data.strip('%')) / 100
    else:
        assert isinstance(data, float), 'This field should be float or percent string'
    return data

policy_schema = Schema({
    Optional('weights', default=None): list,
    Optional('method', default=None): And(str, lambda s: s in ["per_channel", "per_tensor"]),
    Optional('init_sparsity', default=0): And(float, lambda s: s < 1.0 and s >= 0.0),
    Hook('target_sparsity', handler=_valid_prune_sparsity): object,
    Optional('target_sparsity', default=0.5): float,
    Optional("start_epoch", default=0): int,
    Hook('end_epoch', handler=_valid_prune_epoch): object,
    Optional('end_epoch', default=4): int
})

ops_schema = Schema({
    Optional('weight', default=None): {
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

filter_schema = Schema({
    Optional('LabelBalance'): {
        'size': And(int, lambda s: s > 0)
    },
})

transform_schema = Schema({
    Optional('RandomResizedCrop'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0))
    },
    Optional('CropResize'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0))
    },
    Optional('RandomHorizontalFlip'): Or({}, None),
    Optional('ToTensor'): Or({}, None),
    Optional('ToPILImage'): Or({}, None),
    Optional('Normalize'): {
        'mean': And(list, lambda s: all(isinstance(i, float) for i in s)),
        'std': And(list, lambda s: all(isinstance(i, float) for i in s))
    },
    Optional('Resize'): {
        'size': Or(And(list, lambda s: all(isinstance(i, int) for i in s)),
                    And(int, lambda s: s > 0))
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
    Optional('BilinearImagenet'): {
        'height': And(int, lambda s: s > 0),
        'width': And(int, lambda s: s > 0),
        Optional('central_fraction'): bool,
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
    Optional('ParseDecodeImagenet'): Or({}, None),
    Optional('ParseDecodeCoco'): Or({}, None),
    Optional('ImageTypeParse'): Or({}, None),
    Optional('QuantizedInput'): {
        Optional('dtype', default='int8'): And(str, lambda s: s in ['int8', 'uint8']),
        Optional('scale'): And(float, lambda s: s > 0),
    },
    Optional('Transpose'): {
        'perm': And(list, lambda s: all(isinstance(i, int) for i in s)),
    },
})

postprocess_schema = Schema({
    Optional('LabelShift'):  And(int, lambda s: s > 0),
})

dataset_schema = Schema({
    str: object,
})

dataloader_schema = Schema({
    Optional('last_batch', default='rollover'): And(str, lambda s: s in ['rollover', 'discard']),
    Optional('batch_size', default=1): And(int, lambda s: s > 0),
    'dataset': dataset_schema,
    Optional('filter'): filter_schema,
    Optional('transform'): transform_schema,
})

configs_schema = Schema({
    Optional('cores_per_instance'): And(int, lambda s: s > 0),
    Optional('num_of_instance'): And(int, lambda s: s > 0),
    Optional('inter_num_of_threads'): And(int, lambda s: s > 0),
    Optional('intra_num_of_threads'): And(int, lambda s: s > 0),
    Optional('kmp_blocktime'): And(int, lambda s: s >= 0),
})

schema = Schema({
    'model': {
        'name': str,
        'framework': And(str, lambda s: s in FRAMEWORKS),
        Optional('inputs', default=[]): And(Or(str, list), Use(input_to_list)),
        Optional('outputs', default=[]): And(Or(str, list), Use(input_to_list)),
    },
    Optional('device', default='cpu'): And(str, lambda s: s in ['cpu', 'gpu']),
    Optional('quantization', default={'approach': 'post_training_static_quant', \
                                      'calibration': {'sampling_size': [100]}, \
                                      'model_wise': {'weight': {}, 'activation': {}}}): {
        Optional('approach', default='post_training_static_quant'): And(
            str,
            # TODO check if framework support dynamic quantize
            # Now only onnruntime and pytorch supoort
            lambda s: s in ['post_training_static_quant', 
                            'post_training_dynamic_quant',
                            'quant_aware_training']),
        Optional('advance', default=None): {
            Optional('bias_correction'): And(str, lambda s: s in ['weight_empirical']),
        },
        Optional('calibration', default={'sampling_size': [100]}): {
            Optional('sampling_size', default=[100]): And(Or(str, int, list), Use(input_to_list)),
            Optional('dataloader', default=None): dataloader_schema
        },
        Optional('model_wise', default={'weight': {}, 'activation': {}}): {
            Optional('weight', default=None): {
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
    Optional('tuning', default={
        'strategy': {'name': 'basic'},
        'accuracy_criterion': {'relative': 0.01},
        'objective': 'performance',
        'exit_policy': {'timeout': 0, 'max_trials': 100},
        'random_seed': 1978, 'tensorboard': False,
        'workspace': {'path': None}}): {
        Optional('strategy', default={'name': 'basic'}): {
            'name': And(str, lambda s: s in STRATEGIES),
            Optional('accuracy_weight', default=1.0): float,
            Optional('latency_weight', default=1.0): float
        } ,
        Hook('accuracy_criterion', handler=_valid_accuracy_field): object,
        Optional('accuracy_criterion', default={'relative': 0.01}): {
            Optional('relative'): And(Or(str, float), Use(percent_to_float)),
            Optional('absolute'): And(Or(str, float), Use(percent_to_float)),
        },
        Optional('objective', default='performance'): And(str, lambda s: s in OBJECTIVES),
        Optional('exit_policy', default={'timeout': 0, 'max_trials': 100}): {
            Optional('timeout', default=0): int,
            Optional('max_trials', default=100): int,
        },
        Optional('random_seed', default=1978): int,
        Optional('tensorboard', default=False): And(bool, lambda s: s in [True, False]),
        # workspace default value is ./lpot_workspace/$framework/$module_name/, set by code
        Optional('workspace', default={'path': None}): {
            Optional('path', default=None): str,
            Optional('resume'): str
        }
    },
    Optional('evaluation'): {
        Optional('accuracy'): {
            Optional('metric', default=None): {
                Optional('topk'): And(int, lambda s: s in [1, 5]),
                Optional('COCOmAP'): {
                    Optional('anno_path'): str
                }
            },
            Optional('configs'): configs_schema,
            Optional('dataloader'): dataloader_schema,
            Optional('postprocess'): {
                Optional('transform'): postprocess_schema
            },
        },
        Optional('performance'): {
            Optional('warmup', default=10): int,
            Optional('iteration', default=-1): int,
            Optional('configs'): configs_schema,
            Optional('dataloader'): dataloader_schema,
            Optional('postprocess'): {
                Optional('transform'): postprocess_schema
            }
        },
    },
    Optional('pruning'): {
        Optional("magnitude"): {
            str: policy_schema
        },
        Optional('start_epoch', default=0): int,
        Hook('end_epoch', handler=_valid_prune_epoch): object,
        Optional('end_epoch', default=4): int,
        Optional('frequency', default=2): int,
        Optional('init_sparsity', default=0.0): And(float, lambda s: s < 1.0 and s >= 0.0),
        Hook("target_sparsity", handler=_valid_prune_sparsity): object,
        Optional('target_sparsity', default=0.5): And(float, lambda s: s < 1.0 and s >= 0.0)
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
        # set lpot workspace default path
        if self.usr_cfg.tuning.workspace.path is None:
            self.usr_cfg.tuning.workspace.path = './lpot_workspace/{}/{}/'.format(
                                                        self.usr_cfg.model.framework,
                                                        self.usr_cfg.model.name)

    def _read_cfg(self, cfg_fname):
        """Load a config file following yaml syntax.

           Args:
               cfg_fname(string): The name of configuration yaml file
        """
        try:
            with open(cfg_fname, 'r') as f:
                # remove '- ' sign from yaml, it's to avoid the side effect
                # of the syntax as user may not quite familiar with this and
                # arbitrarily add it or not.
                content = f.read().replace('- ', '  ')
                cfg = yaml.load(content, yaml.Loader)
                return schema.validate(cfg)
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
                    value = [value for value in src[key] if value in dst[key]]
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
                    temp_str += param
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
        opwise = copy.deepcopy(opwise_quant)
        for k, v in opwise.items():
            opwise[k] = self._merge_dicts(self._model_wise_tune_space[k[1]], opwise[k])

        cfg = self.usr_cfg
        if cfg.quantization.op_wise:
            for k, v in cfg.quantization.op_wise.items():
                for k_op, _ in opwise.items():
                    if k == k_op[0]:
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
