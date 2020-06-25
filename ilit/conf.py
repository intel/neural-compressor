from .utils import cfg_from_file
from .adaptor import FRAMEWORKS
from .strategy import STRATEGIES

class YamlAttr(dict):
    ''' access yaml using attributes instead of using the dictionary notation.
    '''
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
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
            value = YamlAttr(value[0])
        super(YamlAttr, self).__setitem__(key, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    __setattr__, __getattr__  = __setitem__, __getitem__

class Conf(object):
    ''' config parser.

        Args:
            cfg_fname (string): the name of configuration file to parse
    '''
    def __init__(self, cfg_fname):
        assert cfg_fname is not None
        self.cfg = self._read_cfg(cfg_fname)

    def _read_cfg(self, cfg_fname):
        try:
            cfg = YamlAttr(cfg_from_file(cfg_fname))
            self._sanity_check(cfg)
            return cfg
        except Exception as e:
            raise RuntimeError("The yaml file format is not correct. Please refer to document.")

    def _sanity_check(self, cfg):
        for key in cfg.keys():
            assert key in ['framework', 'device', 'calibration', 'quantization', 'tuning', 'snapshot']

        assert 'framework' in cfg.keys() and isinstance(cfg.framework, str)
        assert cfg.framework in FRAMEWORKS, "The framework {} specified in yaml file is NOT supported".format(cfg.framework)

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
                    assert cfg.quantization.approach.lower() in ['post_training_static_quant', 'post_training_dynamic_quant'], "quant_aware_training is not supported yet."
                if key == 'weight':
                    for w_key in cfg.quantization.weight.keys():
                        assert w_key in ['granularity', 'scheme', 'dtype']
                if key == 'activation':
                    for w_key in cfg.quantization.activation.keys():
                        assert w_key in ['granularity', 'scheme', 'dtype']
        else:
            cfg.quantization = {}

        if not cfg.quantization.approach:
            cfg.quantization.approach = 'post_training_static_quant'

        if 'tuning' in cfg.keys():
            assert 'accuracy_criterion' in {key.lower() for key in cfg.tuning.keys()}
            for key in cfg.tuning.keys():
                assert key in ['strategy', 'metric', 'accuracy_criterion', 'objective', 'timeout', 'random_seed']
                if key == 'strategy':
                    assert cfg.tuning.strategy in STRATEGIES, "The strategy {} specified in yaml file is NOT supported".format(cfg.tuning.strategy)
        else:
            cfg.tuning = {}

        if not cfg.tuning.strategy:
            cfg.tuning.strategy = 'basic'
        if not cfg.tuning.timeout:
            cfg.tuning.timeout = 0
        if not cfg.tuning.random_seed:
            cfg.tuning.random_seed = 1978
        if not cfg.tuning.objective:
            cfg.tuning.objective = 'performance'
        if not cfg.tuning.accuracy_criterion:
            cfg.tuning.accuracy_criterion = {'relative': 0.01}

        if 'snapshot' in cfg.keys():
            assert 'path' in cfg.snapshot.keys() and isinstance(cfg.snapshot.path, str)

