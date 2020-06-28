from abc import abstractmethod
import copy
import itertools
from collections import OrderedDict
from ..adaptor import FRAMEWORKS
from ..objective import OBJECTIVES
from ..metric import METRICS
from ..utils.utility import Timeout

"""The tuning strategies supported by iLiT, including basic, random, bayesian and mse.

   User could add new strategies by implementing new TuneStrategy subclass under this directory.
   The naming convention of new strategy subclass should be something like ABCTuneStrategy, user
   could choose this strategy by setting "abc" string in tuning.strategy field of yaml.

   STRATEGIES variable is used to store all implelmented TuneStrategy subclasses to support
   different tuning strategies.
"""
STRATEGIES = {}

def strategy_registry(cls):
    """The class decorator used to register all TuneStrategy subclasses.

    Args:
        cls (class): The class of register.

    Returns:
        cls: The class of register.
    """    
    assert cls.__name__.endswith('TuneStrategy'), "The name of subclass of TuneStrategy should end with \'TuneStrategy\' substring."
    if cls.__name__[:-len('TuneStrategy')].lower() in STRATEGIES:
        raise ValueError('Cannot have two strategies with the same name')
    STRATEGIES[cls.__name__[:-len('TuneStrategy')].lower()] = cls
    return cls

class TuneStrategy(object):
    """The base class of tuning strategy.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        cfg (YamlAttr):                        The tuning configuration user specified.
        q_dataloader (generator):              Data loader for calibration, mandatory for post-training quantization.
                                               It is iterable and should yield a tuple (input, label) for calibration
                                               dataset containing label, or yield (input, _) for label-free calibration
                                               dataset. The input could be a object, list, tuple or dict, depending on
                                               user implementation, as well as it can be taken as model input.
        q_func (function, optional):           Reserved for future use.
        eval_dataloader (generator, optional): Data loader for evaluation. It is iterable and should yield a tuple
                                               of (input, label). The input could be a object, list, tuple or dict,
                                               depending on user implementation, as well as it can be taken as model
                                               input. The label should be able to take as input of supported
                                               metrics. If this parameter is not None, user needs to specify
                                               pre-defined evaluation metrics through configuration file and should
                                               set "eval_func" paramter as None. Tuner will combine model,
                                               eval_dataloader and pre-defined metrics to run evaluation process.
        eval_func (function, optional):        The evaluation function provided by user. This function takes model
                                               as parameter, and evaluation dataset and metrics should be encapsulated
                                               in this function implementation and outputs a higher-is-better accuracy
                                               scalar value.

                                               The pseudo code should be something like:

                                               def eval_func(model):
                                                    input, label = dataloader()
                                                    output = model(input)
                                                    accuracy = metric(output, label)
                                                    return accuracy
        dicts (dict, optional):                The dict containing resume information. Defaults to None.
    """        
    def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):
        self.model = model
        self.cfg = cfg
        framework_specific_info = {}
        if cfg.framework.name.lower() == 'tensorflow':
            framework_specific_info = {"inputs": cfg.framework.inputs, "outputs": cfg.framework.outputs}
        if cfg.framework.name.lower() == 'mxnet':
            framework_specific_info = {"q_dataloader": q_dataloader}

        framework = cfg.framework.name.lower()
        self.adaptor = FRAMEWORKS[framework](framework_specific_info)

        self.calib_dataloader = q_dataloader
        self.q_func = q_func
        self.eval_dataloader = eval_dataloader
        self.eval_func = eval_func

        self.baseline = None
        self.last_tune_result = None
        self.last_qmodel = None
        self.best_tune_result = None
        self.best_qmodel = None

        objective = 'performance'
        if cfg.tuning.objective:
            objective = cfg.tuning.objective.lower()
        self.objective = OBJECTIVES[objective](cfg.tuning.accuracy_criterion)

        self.modelwise_tune_space = self._modelwise_tune_space(model)
        self.opwise_tune_space = self._opwise_tune_space(model)
        self.modelwise_tune_cfgs = self._tune_cfgs(self.modelwise_tune_space)
        self.opwise_tune_cfgs = OrderedDict()
        for key in self.opwise_tune_space:
            self.opwise_tune_cfgs[key] = self._tune_cfgs(self.opwise_tune_space[key])

        self.calib_iter = cfg.calibration.iterations if cfg.calibration and cfg.calibration.iterations else None
        if self.calib_iter and isinstance(self.calib_iter, str):
            self.calib_iter = self.calib_iter.split(',')
        elif self.calib_iter and isinstance(self.calib_iter, int):
            self.calib_iter = [self.calib_iter]
        else:
            self.calib_iter = [1]

        self.modelwise_quant_cfgs = []
        for cfg in self.modelwise_tune_cfgs:
            if cfg['activation']['dtype'] not in ['fp32']:
                self.modelwise_quant_cfgs.append(cfg)

        self.opwise_quant_cfgs = OrderedDict()
        for key in self.opwise_tune_cfgs:
            cfg_list = self.opwise_tune_cfgs[key]
            new_list = []
            for cfg in cfg_list:
                if cfg['activation']['dtype'] not in ['fp32']:
                    new_list.append(cfg)
            self.opwise_quant_cfgs[key] = new_list

        self.evaluated_cfgs = []

        if dicts is not None:
            self.__dict__.update(dicts)

    @abstractmethod
    def next_tune_cfg(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        Yields:
            tune_config (dict): It's a dict containing the tuning configuration to run.
        """        
        raise notimplementederror

    def traverse(self):
        """The main traverse logic, which could be override by some concrete strategy which needs more hooks.
        """
        with Timeout(self.cfg.tuning.timeout) as t:
            # get fp32 model baseline
            if self.baseline is None:
                print('Getting FP32 model baseline...')
                self.baseline = self._evaluate(self.model, True)
            print('FP32 baseline is: [{:.4f}, {:.4f}]'.format(*self.baseline))

            for tune_cfg in self.next_tune_cfg():
                evaluated = False
                for cfg in self.evaluated_cfgs:
                    if tune_cfg == cfg[0]:
                        self.last_tune_result = cfg[1]
                        evaluated = True
                if evaluated:
                    continue

                self.last_qmodel = self.adaptor.quantize(tune_cfg, self.model, self.calib_dataloader)
                self.last_tune_result = self._evaluate(self.last_qmodel)

                saved_tune_cfg = copy.deepcopy(tune_cfg)
                saved_last_tune_result = copy.deepcopy(self.last_tune_result)
                self.evaluated_cfgs.append([saved_tune_cfg, saved_last_tune_result])

                if self.stop(t):
                    break

    def _intersect(self, src_list, dst_list):
        """Get the intersection result from two lists

        Args:
            src_list (list): The source list intersected from
            dst_list (list): The dest list intersected to

        Returns:
            list: The list containing the intersection result of two lists 
        """        
        if src_list is None:
            return dst_list

        assert isinstance(src_list, list) and isinstance(dst_list, list)
        intersect = [value for value in src_list if value in dst_list]
        if intersect != []:
            dst_list = intersect

        return dst_list

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
                elif dst[key] == src[key]:
                    pass # same leaf value
                else:
                    value = [value for value in src[key] if value in dst[key]]
                    if value != []:
                        dst[key] = value
            else:
                if not isinstance(src[key], dict):
                    dst[key] = src[key]

        return dst

    def _modelwise_tune_space(self, model):
        """Merge user yaml config with framework model wise capability.

        Args:
            model (object): The FP32 model to tune.

        Returns:
            dict: The override model wise tunining space
        """
        capability = self.adaptor.query_fw_capability(model)
        dst = capability['modelwise']

        src = {'weight': OrderedDict(), 'activation': OrderedDict()}

        if self.cfg.calibration and self.cfg.calibration.algorithm and self.cfg.calibration.algorithm.weight:
            src['weight']['algorithm'] = [self.cfg.calibration.algorithm.weight]
        if self.cfg.quantization and self.cfg.quantization.weight and self.cfg.quantization.weight.granularity:
            src['weight']['granularity'] = [self.cfg.quantization.weight.granularity]
        if self.cfg.quantization and self.cfg.quantization.weight and self.cfg.quantization.weight.scheme:
            src['weight']['scheme'] = [self.cfg.quantization.weight.scheme]
        if self.cfg.quantization and self.cfg.quantization.weight and self.cfg.quantization.weight.dtype:
            src['weight']['dtype'] = [self.cfg.quantization.weight.dtype]

        if self.cfg.calibration and self.cfg.calibration.algorithm and self.cfg.calibration.algorithm.activation:
            src['activation']['algorithm'] = [self.cfg.calibration.algorithm.activation]
        if self.cfg.quantization and self.cfg.quantization.activation and self.cfg.quantization.activation.granularity:
            src['activation']['granularity'] = [self.cfg.quantization.activation.granularity]
        if self.cfg.quantization and self.cfg.quantization.activation and self.cfg.quantization.activation.scheme:
            src['activation']['scheme'] = [self.cfg.quantization.activation.scheme]
        if self.cfg.quantization and self.cfg.quantization.activation and self.cfg.quantization.activation.dtype:
            src['activation']['dtype'] = [self.cfg.quantization.activation.dtype]

        return self._merge_dicts(src, dst)

    def _opwise_tune_space(self, model):
        """Generate all tuning spaces for op wise.

        Args:
            model (object): The FP32 model to tune.

        Returns:
            dict: The opwise tunining space
        """
        capability = self.adaptor.query_fw_capability(model)
        opwise = capability['opwise']

        for k, v in opwise.items():
            opwise[k] = self._merge_dicts(self.modelwise_tune_space, opwise[k])

        if self.cfg.tuning.ops:
            for k, v in self.cfg.tuning.ops.items():
                for k_op, _ in opwise.items():
                    if k == k_op[0]:
                        opwise[k_op] = self._merge_dicts(v, opwise[k_op])

        return opwise

    def _tune_cfgs(self, tune_space):
        """generate all possible tuning combinations for each op or model wise tuning.

        Args:
            tune_space (dict): The tuning space to be expanded.

        Returns:
            dict: The expanded tuning configs
        """        
        cfg_lists = self._tune_cfgs_recursively(tune_space)

        # remove unreasonable tuning combinations
        valid_cfgs = []
        quant_dtype = ['int8', 'uint8', 'int4', 'uint4']
        for cfg in cfg_lists:
            dtype = cfg['activation']['dtype']
            if dtype not in quant_dtype:
                cfg['activation'].clear()
                cfg['activation']['dtype'] = dtype

            if 'weight' in cfg:
                dtype = cfg['weight']['dtype']
                if dtype not in quant_dtype:
                    cfg['weight'].clear()
                    cfg['weight']['dtype'] = dtype
                if (cfg['weight']['dtype'] != cfg['activation']['dtype'] and \
                    cfg['weight']['dtype'] not in quant_dtype and cfg['activation']['dtype'] not in quant_dtype) or \
                    (cfg['weight']['dtype'] != cfg['activation']['dtype'] and \
                    cfg['weight']['dtype'] in quant_dtype and cfg['activation']['dtype'] not in quant_dtype) or \
                    (cfg['weight']['dtype'] != cfg['activation']['dtype'] and \
                   cfg['weight']['dtype'] not in quant_dtype and cfg['activation']['dtype'] in quant_dtype):
                    continue

            valid_cfgs.append(cfg)

        # remove duplicated configurations
        valid_cfgs = [cfg[0] for cfg in itertools.groupby(valid_cfgs)]
        return valid_cfgs

    def _tune_cfgs_recursively(self, cfg_dict):
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
                lists = self._tune_cfgs_recursively(cfg_dict[key])
                combinations[key] = lists

        if len(combinations) != 0:
            return self._tune_cfgs_recursively(combinations)

        keys, values = zip(*cfg_dict.items())
        lists = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return lists

    def _evaluate(self, model, baseline=False):
        """The interface of evaluating model.

        Args:
            model (object): The model to be evaluated.
            baseline (bool, optional): TRUE if the evaluated model is FP32 baseline model.

        Returns:
            Objective: The objective value evaluated
        """
        if self.eval_func:
            val = self.objective.evaluate(self.eval_func, model, baseline)
        else:
            # eval_func being None means user will provide dataloader and metric info in config yaml file
            assert self.eval_dataloader and self.cfg.tuning.metric, \
                   "tuning dataloader and tuning metric should NOT be empty when eval_func is None"
            dataloader = self.eval_dataloader
            metric = self.cfg.tuning.metric
            assert len(metric) == 1, "Only one metric should be specified!"
            metric = METRICS[list(metric.keys())[0]](metric)
            def eval_func(model):
                return self.adaptor.evaluate(model, dataloader, metric)
            val = self.objective.evaluate(eval_func, model, baseline)
        return val

    def __getstate__(self):
        """Magic method for pickle saving.

        Returns:
            dict: Saved dict for resuming
        """        
        save_dict = {
            'baseline': self.baseline,
            'cfg': self.cfg,
            'last_tune_result': self.last_tune_result,
            'best_tune_result': self.best_tune_result,
            'modelwise_tune_space': self.modelwise_tune_space,
            'opwise_tune_space': self.opwise_tune_space,
            'modelwise_tune_cfgs': self.modelwise_tune_cfgs,
            'opwise_tune_cfgs': self.opwise_tune_cfgs,
            'modelwise_quant_cfgs': self.modelwise_quant_cfgs,
            'opwise_quant_cfgs': self.opwise_quant_cfgs,
            'evaluated_cfgs': self.evaluated_cfgs
            }
        return save_dict

    def __setstate__(self, d):
        """Magic method for pickle loading.

        Args:
            d (dict): The dict to load.
        """        
        self.__dict__.update(d)

    def stop(self, timeout):
        """Check if need to stop traversing the tuning space, either accuracy goal is met or timeout is reach.

        Args:
            timeout (Timeout): The timeout object instantiated in utils.py

        Returns:
            bool: True if need stop, otherwise False
        """        
        need_stop = False

        if self.objective.compare(self.best_tune_result):
            del self.best_tune_result
            del self.best_qmodel
            self.best_tune_result = self.last_tune_result
            self.best_qmodel = self.last_qmodel
        else:
            del self.last_qmodel

        print('Tune result is: ', '[{:.4f}, {:.4f}]'.format(*self.last_tune_result) if self.last_tune_result else None, 'Best tune result is: ', '[{:.4f}, {:.4f}]'.format(*self.best_tune_result) if self.best_tune_result else None)

        if timeout.seconds != 0 and timeout.timed_out:
            need_stop = True
        elif timeout.seconds == 0 and self.best_tune_result:
            need_stop = True
        else:
            need_stop = False

        return need_stop
