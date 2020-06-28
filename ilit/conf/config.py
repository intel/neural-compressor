from ..utils.utility import cfg_from_file
from ..adaptor import FRAMEWORKS
from ..strategy import STRATEGIES


class YamlAttr(dict):
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
        if isinstance(value, dict) and not isinstance(value, YamlAttr):
            value = YamlAttr(value)
        if isinstance(value, list) and len(value) == 1 and isinstance(
                value[0], dict):
            value = YamlAttr(value[0])
        if isinstance(value, list) and len(value) > 1 and all(isinstance(
                v, dict) for v in value):
            value = YamlAttr({k: v for d in value for k, v in d.items()})
        super(YamlAttr, self).__setitem__(key, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    __setattr__, __getattr__ = __setitem__, __getitem__


class Conf(object):
    """config parser.

    Args:
        cfg_fname (string): The path to the configuration file.

    """    
    def __init__(self, cfg_fname):
        assert cfg_fname is not None
        self.cfg = self._read_cfg(cfg_fname)

    def _read_cfg(self, cfg_fname):
        try:
            cfg = YamlAttr(cfg_from_file(cfg_fname))
            self._sanity_check(cfg)
            return cfg
        except Exception as e:
            raise RuntimeError(
                "The yaml file format is not correct. Please refer to document."
            )

    def _sanity_check(self, cfg):
        for key in cfg.keys():
            assert key in [
                'framework', 'device', 'calibration', 'quantization', 'tuning',
                'snapshot'
            ]

        assert 'framework' in cfg and 'name' in cfg.framework and isinstance(
            cfg.framework.name, str)
        assert cfg.framework.name.lower(
        ) in FRAMEWORKS, "The framework {} specified in yaml file is NOT supported".format(
            cfg.framework.name)

        if cfg.framework.name.lower() == 'tensorflow':
            assert cfg.framework.inputs and cfg.framework.outputs, "TensorFlow backend requires user to specify the graph inputs and outputs"
        else:
            cfg.framework.inputs = None
            cfg.framework.outputs = None

        if 'calibration' in cfg.keys():
            for key in cfg.calibration.keys():
                assert key in ['iterations', 'algorithm']
                if key == 'algorithm':
                    for algo_key in cfg.calibration.algorithm.keys():
                        assert algo_key in ['weight', 'activation']

        if 'device' in cfg.keys():
            assert cfg.device in ['cpu', 'gpu']
        else:
            cfg.device = 'cpu'

        if 'quantization' in cfg.keys():
            for key in cfg.quantization.keys():
                assert key in ['approach', 'weight', 'activation']
                if key == 'approach':
                    assert cfg.quantization.approach.lower() in [
                        'post_training_static_quant'
                    ], "post_training_dynamic_quant and quant_aware_training are not supported yet."
                if key == 'weight':
                    for w_key in cfg.quantization.weight.keys():
                        assert w_key in ['granularity', 'scheme', 'dtype']
                if key == 'activation':
                    for a_key in cfg.quantization.activation.keys():
                        assert a_key in ['granularity', 'scheme', 'dtype']
        else:
            cfg.quantization = {}

        if not cfg.quantization.approach:
            cfg.quantization.approach = 'post_training_static_quant'

        if 'tuning' in cfg.keys():
            assert 'accuracy_criterion' in {
                key.lower()
                for key in cfg.tuning.keys()
            }
            for key in cfg.tuning.keys():
                assert key in [
                    'strategy', 'metric', 'accuracy_criterion', 'objective',
                    'timeout', 'random_seed', 'ops'
                ]
                if key == 'strategy':
                    assert cfg.tuning.strategy.lower(
                    ) in STRATEGIES, "The strategy {} specified in yaml file is NOT supported".format(
                        cfg.tuning.strategy)
                if key == 'ops':
                    assert isinstance(cfg.tuning.ops, dict)
                    for op in cfg.tuning.ops.keys():
                        op_value = cfg.tuning.ops[op]
                        assert isinstance(op_value, dict)
                        for attr in op_value.keys():
                            assert attr in ['activation', 'weight']
                            assert isinstance(op_value[attr], dict)
                            for attr_key in op_value[attr].keys():
                                assert attr_key in [
                                    'granularity', 'scheme', 'dtype',
                                    'algorithm'
                                ]
        else:
            cfg.tuning = {}

        if not cfg.tuning.strategy:
            cfg.tuning.strategy = 'basic'

        if cfg.tuning.strategy.lower() == 'mse':
            assert cfg.framework.name.lower(
            ) != 'pytorch', "The MSE strategy doesn't support PyTorch framework"

        if not cfg.tuning.timeout:
            cfg.tuning.timeout = 0

        if not cfg.tuning.random_seed:
            cfg.tuning.random_seed = 1978

        if not cfg.tuning.objective:
            cfg.tuning.objective = 'performance'

        if not cfg.tuning.accuracy_criterion:
            cfg.tuning.accuracy_criterion = {'relative': 0.01}

        if 'snapshot' in cfg.keys():
            assert 'path' in cfg.snapshot.keys() and isinstance(
                cfg.snapshot.path, str)
