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

# Schema library has different loading sequence priorities for different
# value types.
# To make sure the fields under dataloader.transform field of yaml file
# get loaded with written sequence, this workaround is used to convert
# None to {} in yaml load().
yaml.add_constructor('tag:yaml.org,2002:null', lambda loader, node: {})

class DotDict(dict):
    """access yaml using attributes instead of using the dictionary notation.

    Args:
        value (dict): The dict object to access.

    """

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __getitem__(self, key):
        value = self.get(key, None)
        return value

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        if isinstance(value, list) and len(value) == 1 and isinstance(
                value[0], dict):
            value = DotDict(value[0])
        if isinstance(value, list) and len(value) > 1 and all(isinstance(
                v, dict) for v in value):
            value = DotDict({k: v for d in value for k, v in d.items()})
        super(DotDict, self).__setitem__(key, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    __setattr__, __getattr__ = __setitem__, __getitem__

def _valid_framework_field(key, scope, error):
    if scope['name'] == 'tensorflow':
        assert 'inputs' in scope and 'outputs' in scope

def _valid_type_field(key, scope, error):
    if scope['type'] == 'style_transfer':
        assert 'content_folder' in scope and 'style_folder' in scope

def _valid_accuracy_field(key, scope, error):
    assert bool('relative' in scope['accuracy_criterion']) != bool('absolute' in scope['accuracy_criterion'])

def input_to_list(data):
    if isinstance(data, str):
        return [s.strip() for s in data.split(',')]
    if isinstance(data, int):
        return [data]
    else:
        assert isinstance(data, list)
        return data

def percent_to_float(data):
    if isinstance(data, str) and re.match('-?\d+(\.\d+)?%', data):
        data = float(data.strip('%'))/100
    else:
        assert isinstance(data, float), 'This field should be float or percent string'
    return data

ops_schema = Schema({
    Optional('weight', default=None): {
        Optional('granularity', default=None): And(list, lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
        Optional('scheme', default=None): And(list, lambda s: all(i in ['asym', 'sym'] for i in s)),
        Optional('dtype', default=None): And(list, lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s)),
        Optional('algorithm', default=None): And(list, lambda s: all(i in ['minmax', 'kl'] for i in s))
    },
    Optional('activation', default=None): {
        Optional('granularity', default=None): And(list, lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
        Optional('scheme', default=None): And(list, lambda s: all(i in ['asym', 'sym'] for i in s)),
        Optional('dtype', default=None): And(list, lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s)),
        Optional('algorithm', default=None): And(list, lambda s: all(i in ['minmax', 'kl'] for i in s))
    }
})

transform_schema = Schema({
    Optional('RandomResizedCrop'): {
        'size': And(int, lambda s: s > 0)
    },
    Optional('RandomHorizontalFlip'): Or({}, None),
    Optional('ToTensor'): Or({}, None),
    Optional('Normalize'): {
        'mean': And(list, lambda s: all(isinstance(i, float) for i in s)),
        'std': And(list, lambda s: all(isinstance(i, float) for i in s))
    },
    Optional('Resize'): {
        'size': And(int, lambda s: s > 0)
    },
    Optional('CenterCrop'): {
        'size': And(int, lambda s: s > 0)
    },
    Optional('Reshape'): {
        'shape': And(list, lambda s: all(isinstance(i, int) for i in s)),
    }
})

dataloader_schema = Schema({
    Optional('batch_size', default=1): And(int, lambda s: s > 0),
    Optional('dataset', default=None): {
        Hook('type', handler=_valid_type_field): object,
        Optional('type'): str,
        Optional('root'): str,
        Optional('content_folder'): str,
        Optional('style_folder'): str,
    },
    Optional('transform', default=None): transform_schema
})

schema = Schema({
    'framework': {
        Hook('name', handler=_valid_framework_field): object,
        'name': And(str, lambda s: s in FRAMEWORKS),
        Optional('inputs', default=None): And(Or(str, list), Use(input_to_list)),
        Optional('outputs', default=None): And(Or(str, list), Use(input_to_list))
    },
    Optional('device', default='cpu'): And(str, lambda s: s in ['cpu', 'gpu']),
    Optional('quantization', default={'approach':'post_training_static_quant'}): {
        Optional('approach', default='post_training_static_quant'): And(str, 
                                                                        lambda s: s in ['post_training_static_quant', 'quant_aware_training']),
        Optional('weight', default=None): {
            Optional('granularity', default=None): And(Or(str, list), Use(input_to_list), lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
            Optional('scheme', default=None): And(Or(str, list), Use(input_to_list), lambda s: all(i in ['asym', 'sym'] for i in s)),
            Optional('dtype', default=None): And(Or(str, list), Use(input_to_list), lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s))
        },
        Optional('activation', default=None): {
            Optional('granularity', default=None): And(Or(str, list), Use(input_to_list), lambda s: all(i in ['per_channel', 'per_tensor'] for i in s)),
            Optional('scheme', default=None): And(Or(str, list), Use(input_to_list), lambda s: all(i in ['asym', 'sym'] for i in s)),
            Optional('dtype', default=None): And(Or(str, list), Use(input_to_list), lambda s: all(i in ['int8', 'uint8', 'fp32', 'bf16'] for i in s))
        }
    },
    Optional('calibration', default={'iterations': [1]}): {
        Optional('iterations', default=[1]): And(Or(str, int, list), Use(input_to_list)),
        Optional('algorithm', default=None): {
            Optional('weight', default=None): And(Or(str, list), Use(input_to_list), lambda s: all(i in ['minmax', 'kl'] for i in s)),
            Optional('activation', default=None): And(Or(str, list), Use(input_to_list), lambda s: all(i in ['minmax', 'kl'] for i in s))
        },
        Optional('dataloader', default=None): dataloader_schema
    },
    Optional('tuning', default={'strategy':'basic', 'accuracy_criterion': {'relative': 0.01}, 'objective': 'performance', 'timeout': 0, 'random_seed': 1978}): {
        Optional('strategy', default='basic'): And(str, lambda s: s in STRATEGIES),
        Optional('objective', default='performance'): And(str, lambda s: s in OBJECTIVES),
        Optional('timeout', default=0): int, 
        Optional('random_seed', default=1978): int, 
        Hook('accuracy_criterion', handler=_valid_accuracy_field): object,
        Optional('accuracy_criterion', default={'relative': 0.01}): {
            Optional('relative'): And(Or(str, float), Use(percent_to_float)),
            Optional('absolute'): And(Or(str, float), Use(percent_to_float)),
        }, 
        Optional('metric', default=None): {
            Optional('topk'): And(int, lambda s: s in [1, 5]),
        }, 
        Optional('ops', default=None): {
            str: ops_schema
        }
    },
    Optional('evaluation', default=None): {
        Optional('dataloader', default=None): dataloader_schema,
        Optional('postprocess', default=None): {
            Optional('transform', default=None): transform_schema
        }
    },
    Optional('snapshot', default={'path': '~/.ilit/snapshot/'}): {
        Optional('path', default='~/.ilit/snapshot/'): str
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
        self._modelwise_tune_space = None
        self._opwise_tune_space = None

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
                elif dst[key] == src[key] or src[key] == None:
                    pass  # same leaf value
                else:
                    value = [value for value in src[key] if value in dst[key]]
                    if value != []:
                        dst[key] = value
            else:
                if not isinstance(src[key], dict):
                    dst[key] = src[key]

        return dst 

    def modelwise_tune_space(self, modelwise_quant):
        src = DotDict({'weight': dict(), 'activation': dict()})

        cfg = self.usr_cfg
        if cfg.calibration and cfg.calibration.algorithm and cfg.calibration.algorithm.weight:
            src.weight.algorithm = cfg.calibration.algorithm.weight

        if cfg.quantization and cfg.quantization.weight and cfg.quantization.weight.granularity:
            src.weight.granularity = cfg.quantization.weight.granularity

        if cfg.quantization and cfg.quantization.weight and cfg.quantization.weight.scheme:
            src.weight.scheme = cfg.quantization.weight.scheme

        if cfg.quantization and cfg.quantization.weight and cfg.quantization.weight.dtype:
            src.weight.dtype = cfg.quantization.weight.dtype

        if cfg.calibration and cfg.calibration.algorithm and cfg.calibration.algorithm.activation:
            src.activation.algorithm = cfg.calibration.algorithm.activation

        if cfg.quantization and cfg.quantization.activation and cfg.quantization.activation.granularity:
            src.activation.granularity = cfg.quantization.activation.granularity

        if cfg.quantization and cfg.quantization.activation and cfg.quantization.activation.scheme:
            src.activation.scheme = cfg.quantization.activation.scheme

        if cfg.quantization and cfg.quantization.activation and cfg.quantization.activation.dtype:
            src.activation.dtype = cfg.quantization.activation.dtype

        self._modelwise_tune_space = self._merge_dicts(src, modelwise_quant)
        return self._modelwise_tune_space

    def opwise_tune_space(self, opwise_quant):
        opwise = copy.deepcopy(opwise_quant)
        for k, v in opwise.items():
            opwise[k] = self._merge_dicts(self._modelwise_tune_space, opwise[k])

        cfg = self.usr_cfg
        if cfg.tuning.ops:
            for k, v in cfg.tuning.ops.items():
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
                    cfg.weight.dtype not in quant_dtype and cfg.activation.dtype not in quant_dtype) or \
                    (cfg.weight.dtype != cfg.activation.dtype and
                     cfg.weight.dtype in quant_dtype and cfg.activation.dtype not in quant_dtype) or \
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
