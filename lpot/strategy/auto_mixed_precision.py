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

import copy
from collections import OrderedDict
from .strategy import strategy_registry, TuneStrategy
from ..utils import logger

@strategy_registry
class AutoMixedPrecisionTuneStrategy(TuneStrategy):
    """The graph optimization strategy which tunes the mixed precision model with below order.

    1. modelwise tuning for all tunable ops.
    2. fallback tuning from bottom to top to decide the priority of which op has biggest impact
       on accuracy.
    3. incremental fallback tuning by fallbacking multiple ops with the order got from #2.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        conf (Class):                          The Conf class instance initialized from user yaml
                                               config file.
        eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                               and should yield a tuple of (input, label).
                                               The input could be a object, list, tuple or dict,
                                               depending on user implementation, as well as it can
                                               be taken as model input. The label should be able
                                               to take as input of supported metrics. If this
                                               parameter is not None, user needs to specify
                                               pre-defined evaluation metrics through configuration
                                               file and should set "eval_func" parameter as None.
                                               Tuner will combine model, eval_dataloader and
                                               pre-defined metrics to run evaluation process.
        eval_func (function, optional):        The evaluation function provided by user.
                                               This function takes model as parameter, and
                                               evaluation dataset and metrics should be
                                               encapsulated in this function implementation and
                                               outputs a higher-is-better accuracy scalar value.

                                               The pseudo code should be something like:

                                               def eval_func(model):
                                                    input, label = dataloader()
                                                    output = model(input)
                                                    accuracy = metric(output, label)
                                                    return accuracy
        dicts (dict, optional):                The dict containing resume information.
                                               Defaults to None.

    """

    def __init__(self, model, conf, q_dataloader=None, q_func=None,
                eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):
        super().__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts,
            q_hooks)

    def next_tune_cfg(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
            according to last tuning result.

        """
        # Model wise tuning
        op_cfgs = {}
        best_cfg = None
        best_acc = 0

        logger.debug("Start AutoMixedPrecision strategy by model-wise tuning")
        for iterations in self.calib_iter:
            op_cfgs['calib_iteration'] = int(iterations)

            for combined_cfg in self.combined_model_wise_quant_cfgs:
                op_cfgs['op'] = OrderedDict()
                for op, op_cfg in self.opwise_quant_cfgs.items():
                    if op[1] in combined_cfg.keys() and len(op_cfg) > 0:
                        op_cfgs['op'][op] = copy.deepcopy(
                            self._get_common_cfg(combined_cfg[op[1]], op_cfg))
                    elif op[1] not in combined_cfg.keys() or not op_cfg:
                        pass
                    else:
                        op_cfgs['op'][op] = copy.deepcopy(
                            self.opwise_tune_cfgs[op][0])

                yield op_cfgs
                acc, _ = self.last_tune_result
                # if acc >= best_acc or self.eval_dataloader is None:
                if acc >= best_acc:
                    best_acc = acc
                    best_cfg = copy.deepcopy(op_cfgs)

        if best_cfg is not None:
            fallback_dtypes = []
            for data_type in ["bf16", "fp32"]:
                for _, tune_space in self.modelwise_tune_space.items():
                    if (data_type in tune_space["activation"]["dtype"] and
                        data_type not in fallback_dtypes):
                        fallback_dtypes.append(data_type)

            for fallback_dtype in fallback_dtypes:
                logger.debug(
                    "Continue basic strategy by sorting opwise {} fallback priority".format
                    (fallback_dtype))
                ops_acc = OrderedDict()
                for op, configs in reversed(self.opwise_tune_cfgs.items()):
                    op_cfgs = copy.deepcopy(best_cfg)
                    for cfg in configs:
                        if fallback_dtype == cfg['activation']['dtype']:
                            op_cfgs['op'][op]['activation'].clear()
                            op_cfgs['op'][op]['activation']['dtype'] = fallback_dtype
                            if 'weight' in cfg:
                                assert cfg['weight']['dtype'] == fallback_dtype
                                op_cfgs['op'][op]['weight'].clear()
                                op_cfgs['op'][op]['weight']['dtype'] = fallback_dtype
                    yield op_cfgs
                    acc, _ = self.last_tune_result
                    ops_acc[op] = acc

                op_cfgs = copy.deepcopy(best_cfg)
                if ops_acc is not None:
                    ordered_ops = sorted(ops_acc.keys(), key=lambda key: ops_acc[key],
                                         reverse=True)
                    for op in ordered_ops:
                        old_cfg = copy.deepcopy(op_cfgs['op'][op])
                        for cfg in self.opwise_tune_cfgs[op]:
                            if fallback_dtype == cfg['activation']['dtype']:
                                op_cfgs['op'][op]['activation'].clear()
                                op_cfgs['op'][op]['activation']['dtype'] = fallback_dtype
                                if 'weight' in cfg:
                                    assert cfg['weight']['dtype'] == fallback_dtype
                                    op_cfgs['op'][op]['weight'].clear()
                                    op_cfgs['op'][op]['weight']['dtype'] = fallback_dtype
                        yield op_cfgs
                        acc, _ = self.last_tune_result
                        if acc <= best_acc:
                            op_cfgs['op'][op] = copy.deepcopy(old_cfg)
                        else:
                            best_acc = acc

                    op_cfgs = copy.deepcopy(best_cfg)
                    for op in ordered_ops:
                        for cfg in self.opwise_tune_cfgs[op]:
                            if fallback_dtype == cfg['activation']['dtype']:
                                op_cfgs['op'][op]['activation'].clear()
                                op_cfgs['op'][op]['activation']['dtype'] = fallback_dtype
                                if 'weight' in op_cfgs['op'][op]:
                                    op_cfgs['op'][op]['weight'].clear()
                                    op_cfgs['op'][op]['weight']['dtype'] = fallback_dtype
                        yield op_cfgs
        else:
            logger.debug(self.opwise_tune_cfgs)
            op_cfgs['op'] = OrderedDict()
            for op in self.opwise_tune_cfgs.keys():
                op_cfgs['op'][op] = copy.deepcopy(self.opwise_tune_cfgs[op][0])
            yield op_cfgs

        return

    def traverse(self):
        # get fp32 model baseline
        if self.baseline is None and self.eval_dataloader:
            logger.info("Get FP32 model baseline.")
            self.baseline = self._evaluate(self.model)
            # record the FP32 baseline
            self._add_tuning_history()
            baseline_msg = '[accuracy: {:.4f}, {}: {:.4f}]'.format(self.baseline[0],
                                                        str(self.objective.measurer),
                                                        self.baseline[1]) \
                                                        if self.baseline else 'n/a'
            logger.info("FP32 baseline is: {}".format(baseline_msg))

        trials_count = 0
        for tune_cfg in self.next_tune_cfg():
            # add tune_cfg here as quantize use tune_cfg
            tune_cfg['advance'] = self.cfg.quantization.advance
            trials_count += 1
            tuning_history = self._find_tuning_history(tune_cfg)
            if tuning_history and trials_count < self.cfg.tuning.exit_policy.max_trials:
                self.last_tune_result = tuning_history['last_tune_result']
                self.best_tune_result = tuning_history['best_tune_result']
                logger.warn("Find evaluated tuning config, skip.")
                continue

            logger.debug("Dump current graph optimization configuration:")
            logger.debug(tune_cfg)
            self.last_qmodel = self.adaptor.quantize(
                tune_cfg, self.model, self.calib_dataloader, self.q_func)
            assert self.last_qmodel
            if self.eval_dataloader:
                self.last_tune_result = self._evaluate(self.last_qmodel)

                need_stop = self.stop(self.cfg.tuning.exit_policy.timeout, trials_count)
                # record the tuning history
                saved_tune_cfg = copy.deepcopy(tune_cfg)
                saved_last_tune_result = copy.deepcopy(self.last_tune_result)
                self._add_tuning_history(saved_tune_cfg, saved_last_tune_result)
            else:
                # If the eval_dataloader was not specified under the config yaml file,
                # We only converted the model with customized precisions.
                self.best_qmodel = self.last_qmodel
                need_stop = True

            if need_stop:
                break
