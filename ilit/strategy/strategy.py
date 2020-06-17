from abc import abstractmethod
import time
import copy
import itertools
from collections import OrderedDict
from ..adaptor import FRAMEWORKS
from ..objective import OBJECTIVES
from ..metric import METRICS
from ..utils import Timeout

'''The tuning strategies supported by iLit, including basic, random, bayesian and mse.

   User could add new strategies by implementing new TuneStrategy subclass under this directory.
   The naming convention of new strategy subclass should be something like ABCTuneStrategy, user
   could choose this strategy by setting "abc" string in tuning.strategy field of yaml.

   STRATEGIES variable is used to store all implelmented TuneStrategy subclasses to support
   different tuning strategies.
'''
STRATEGIES = {}

def strategy_registry(cls):
    '''The class decorator used to register all TuneStrategy subclasses.

       Args:
           cls (class): The class of register.
    '''
    assert cls.__name__.endswith('TuneStrategy'), "The name of subclass of TuneStrategy should end with \'TuneStrategy\' substring."
    if cls.__name__[:-len('TuneStrategy')].lower() in STRATEGIES:
        raise ValueError('Cannot have two strategies with the same name')
    STRATEGIES[cls.__name__[:-len('TuneStrategy')].lower()] = cls
    return cls

class TuneStrategy(object):
    '''The base class of tuning strategy.

       Args:
           model (object): The model user specified.
           cfg (object): The configuration user specified.
           q_dataloder (object): The class object of framework adaptor.
           q_func (object): The class object of framework adaptor.
           baseline (tuple): The baseline of fp32 model.
           calibration_loader (optional): The calibration data feeder provided by user.
           eval_func (optional): The test function provided by user.
    '''
    def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):
        self.model = model
        self.cfg = cfg
        input_output_info = {}
        if "inputs" in cfg:
            input_output_info["inputs"] = cfg["inputs"]
        if "outputs" in cfg:
            input_output_info["outputs"] = cfg["outputs"]
        if q_dataloader is not None:
            input_output_info["q_dataloader"] = q_dataloader

        framework = cfg.framework.lower()
        self.adaptor = FRAMEWORKS[framework](input_output_info)

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

        self.customized_ops = cfg.customized_ops
        # inputs and outputs attributes are specifically used by tensorflow adaptor.
        self.inputs = cfg.inputs
        self.outputs = cfg.outputs

        self.modelwise_tune_space = self._modelwise_tune_space(model)
        self.opwise_tune_space = self._opwise_tune_space(model)
        self.modelwise_tune_cfgs = self._tune_cfgs(self.modelwise_tune_space)
        self.opwise_tune_cfgs = OrderedDict()
        for key in self.opwise_tune_space:
            self.opwise_tune_cfgs[key] = self._tune_cfgs(self.opwise_tune_space[key])

        self.calib_iter = cfg.calibration.iterations if cfg.calibration and cfg.calibration.iterations else None
        if self.calib_iter:
            self.calib_iter = self.calib_iter.split(',')
        else:
            self.calib_iter = [1]

        self.modelwise_quant_cfgs = []
        for cfg in self.modelwise_tune_cfgs:
            if cfg['activation']['data_type'] not in ['fp32']:
                self.modelwise_quant_cfgs.append(cfg)

        self.opwise_quant_cfgs = OrderedDict()
        for key in self.opwise_tune_cfgs:
            cfg_list = self.opwise_tune_cfgs[key]
            new_list = []
            for cfg in cfg_list:
                if cfg['activation']['data_type'] not in ['fp32']:
                    new_list.append(cfg)
            self.opwise_quant_cfgs[key] = new_list

        self.evaluated_cfgs = []

        if dicts is not None:
            self.__dict__.update(dicts)

    @abstractmethod
    def next_tune_cfg(self):
        '''The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

           Args:
               qmodel (quantized model): It's the quanitzed model by last tuning result. If None, it means initial state.
               current_tune_result (tuple): It's the return objective value by last tuning result.

           Return:
               tune_config (dict) it's a dict containing the tuning configuration to run.
        '''
        raise notimplementederror

    def traverse(self):
        '''The main traverse logic, which could be override by some concrete strategy which needs more hook.

        '''
        with Timeout(self.cfg.tuning.timeout) as t:
            # get fp32 model baseline
            if self.baseline is None:
                self.baseline = self._evaluate(self.model, True)

            for tune_cfg in self.next_tune_cfg():
                evaluated = False
                for cfg in self.evaluated_cfgs:
                    if tune_cfg == cfg[0]:
                        self.last_tune_result = cfg[1]
                        evaluated = True
                if evaluated:
                    continue

                self.last_qmodel = self.adaptor.quantize(tune_cfg, self.model, self.calib_dataloader)
                print('cfg', tune_cfg)
                # print('eval', self.last_qmodel)
                self.last_tune_result = self._evaluate(self.last_qmodel)

                saved_tune_cfg = copy.deepcopy(tune_cfg)
                saved_last_tune_result = copy.deepcopy(self.last_tune_result)
                self.evaluated_cfgs.append([saved_tune_cfg, saved_last_tune_result])

                if self.stop(t):
                    print("Specified timeout is reached! Exit...")
                    break

    def _intersect(self, src_list, dst_list):
        if src_list is None:
            return dst_list

        assert isinstance(src_list, list) and isinstance(dst_list, list)
        intersect = [value for value in src_list if value in dst_list]
        if intersect != []:
            dst_list = intersect

        return dst_list

    def _merge_dicts(self, src, dst):
        '''Merges src dict into dst dict'''

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
        '''Merge user yaml config with framework model wise capability.

           Return:
               modelwise_tune_space (dict) The override model wise tunining configs
        '''
        capability = self.adaptor.query_fw_capability(model)
        dst = capability['modelwise']

        src = {'weight': OrderedDict(), 'activation': OrderedDict()}

        if self.cfg.calibration and self.cfg.calibration.algo and self.cfg.calibration.algo.weight:
            src['weight']['algo'] = [self.cfg.calibration.algo.weight]
        if self.cfg.quantization and self.cfg.quantization.weight and self.cfg.quantization.weight.granularity:
            src['weight']['granularity'] = [self.cfg.quantization.weight.granularity]
        if self.cfg.quantization and self.cfg.quantization.weight and self.cfg.quantization.weight.mode:
            src['weight']['mode'] = [self.cfg.quantization.weight.mode]
        if self.cfg.quantization and self.cfg.quantization.weight and self.cfg.quantization.weight.data_type:
            src['weight']['data_type'] = [self.cfg.quantization.weight.data_type]

        if self.cfg.calibration and self.cfg.calibration.algo and self.cfg.calibration.algo.activation:
            src['activation']['algo'] = [self.cfg.calibration.algo.activation]
        if self.cfg.quantization and self.cfg.quantization.activation and self.cfg.quantization.activation.granularity:
            src['activation']['granularity'] = [self.cfg.quantization.activation.granularity]
        if self.cfg.quantization and self.cfg.quantization.activation and self.cfg.quantization.activation.mode:
            src['activation']['mode'] = [self.cfg.quantization.activation.mode]
        if self.cfg.quantization and self.cfg.quantization.activation and self.cfg.quantization.activation.data_type:
            src['activation']['data_type'] = [self.cfg.quantization.activation.data_type]

        return self._merge_dicts(src, dst)

    def _opwise_tune_space(self, model):
        '''Generate all tuning spaces for op wise.
        '''
        capability = self.adaptor.query_fw_capability(model)
        opwise = capability['opwise']

        for k, v in opwise.items():
            opwise[k] = self._merge_dicts(self.modelwise_tune_space, opwise[k])

        if self.customized_ops:
            for k, v in self.customized_ops.items():
                for k_op, _ in opwise.items():
                    if k == k_op[0]:
                        opwise[k_op] = self._merge_dicts(v, opwise[k_op])

        return opwise

    def _tune_cfgs(self, tune_space):
        # generate all possible tuning combinations for each op or model wise tuning.
        cfg_lists = self._tune_cfgs_recursively(tune_space)

        # remove unreasonable tuning combinations
        valid_cfgs = []
        quant_dtype = ['int8', 'uint8', 'int4', 'uint4']
        for cfg in cfg_lists:
            dtype = cfg['activation']['data_type']
            if dtype not in quant_dtype:
                cfg['activation'].clear()
                cfg['activation']['data_type'] = dtype

            if 'weight' in cfg:
                dtype = cfg['weight']['data_type']
                if dtype not in quant_dtype:
                    cfg['weight'].clear()
                    cfg['weight']['data_type'] = dtype
                if (cfg['weight']['data_type'] != cfg['activation']['data_type'] and \
                    cfg['weight']['data_type'] not in quant_dtype and cfg['activation']['data_type'] not in quant_dtype) or \
                    (cfg['weight']['data_type'] != cfg['activation']['data_type'] and \
                    cfg['weight']['data_type'] in quant_dtype and cfg['activation']['data_type'] not in quant_dtype) or \
                    (cfg['weight']['data_type'] != cfg['activation']['data_type'] and \
                   cfg['weight']['data_type'] not in quant_dtype and cfg['activation']['data_type'] in quant_dtype):
                    continue

            valid_cfgs.append(cfg)

        # remove duplicated configurations
        valid_cfgs = [cfg[0] for cfg in itertools.groupby(valid_cfgs)]
        return valid_cfgs

    def _tune_cfgs_recursively(self, cfg_dict):
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
        '''The interface of evaluating model.

           Return:
               model (object) it's the model to evaluate.
               baseline (bool) it's TRUE if the evaluated model is FP32 baseline model.
        '''
        if self.eval_func:
            val = self.objective.evaluate(self.eval_func, model, baseline)
        else:
            # eval_func being None means user will provide dataloader and metric info in config yaml file
            assert self.eval_dataloader and self.cfg.tuning.metric, \
                   "tuning dataloader and tuning metric should NOT be empty when eval_func is None"
            dataloader = self.eval_dataloader
            metric = self.cfg.tuning.metric
            assert len(metric) == 1, "Only one metric should be specified!"
            #print(metric)
            metric = METRICS[list(metric.keys())[0]](metric)
            def eval_func(model):
                return self.adaptor.evaluate(model, dataloader, metric)
            val = self.objective.evaluate(eval_func, model, baseline)
        return val

    def __getstate__(self):
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
        self.__dict__.update(d)

    def stop(self, timeout):
        '''Check if need to stop traversing the tuning space, either accuracy goal is met or timeout is reach.

           Args:
               timeout (Timeout) The timeout instantiate object in utils.py

        '''
        if timeout.timed_out:
            if self.best_tune_result is None:
                self.best_tune_result = self.last_tune_result
                self.best_qmodel = self.last_qmodel
            return True
        elif self.objective.compare(self.best_tune_result) and timeout.seconds == 0:
            self.best_tune_result = self.last_tune_result
            self.best_qmodel = self.last_qmodel
            return True
        elif self.objective.compare(self.best_tune_result):
            self.best_tune_result = self.last_tune_result
            self.best_qmodel = self.last_qmodel
            return False
        else:
            return False
