
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

"""The SigOpt Tuning Strategy provides support for the quantization process."""
import copy
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport
from neural_compressor.strategy.strategy import strategy_registry, TuneStrategy
from collections import OrderedDict
from neural_compressor.strategy.utils.tuning_sampler import OpWiseTuningSampler
from neural_compressor.strategy.utils.tuning_structs import OpTuningConfig

sigopt = LazyImport('sigopt')

@strategy_registry
class SigOptTuneStrategy(TuneStrategy):
    """The tuning strategy using SigOpt HPO search in tuning space.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        conf (Conf):                           The Conf class instance initialized from user yaml
                                               config file.
        q_dataloader (generator):              Data loader for calibration, mandatory for
                                               post-training quantization.
                                               It is iterable and should yield a tuple (input,
                                               label) for calibration dataset containing label,
                                               or yield (input, _) for label-free calibration
                                               dataset. The input could be a object, list, tuple or
                                               dict, depending on user implementation, as well as
                                               it can be taken as model input.
        q_func (function, optional):           Reserved for future use.
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

    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):
        """Initialize the SigOpt tuning strategy if the user specified to use it."""
        super().__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts,
            q_hooks)
        strategy_name = conf.usr_cfg.tuning.strategy.name
        if strategy_name.lower() == "sigopt":
            try:
                import sigopt
            except ImportError:
                try:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "sigopt"])
                    import sigopt # pylint: disable=import-error
                except:
                    assert False, "Unable to import sigopt from the local environment."
        else:
            pass
        # SigOpt init
        client_token = conf.usr_cfg.tuning.strategy.sigopt_api_token
        self.project_id = conf.usr_cfg.tuning.strategy.sigopt_project_id
        self.experiment_name = conf.usr_cfg.tuning.strategy.sigopt_experiment_name
        try:
            assert client_token != None
        except(AssertionError):
            logger.error("`sigopt_api_token` field in yaml file is required. " \
                         "Please refer to details in /docs/sigopt_strategy.md.")
            exit(0)
        try:
            assert self.project_id != None
            logger.warning('Project id is {}, ' \
                           'Please check whether it is created in the sigopt account.'\
                           .format(self.project_id))
        except(AssertionError):
            logger.error("`sigopt_project_id` field in yaml file is required. " \
                         "Please refer to details in /docs/sigopt_strategy.md.")
            exit(0)
        if self.experiment_name == 'nc-tune':
           logger.info("Default experiment name `nc-tune` is used, " \
                       "Please refer to details in /docs/sigopt_strategy.md " \
                       "if user wants to modify it.")
        else:
           logger.info("Experiment name is {}.".format(self.experiment_name))

        self.conn = sigopt.Connection(client_token)
        self.experiment = None

    def params_to_tune_configs(self, params):
        """Get the parameters of the tuning strategy."""
        op_tuning_cfg = {}
        calib_sampling_size_lst = self.tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        for op_name_type, configs in self.op_configs.items():
            if len(configs) == 1:
                op_tuning_cfg[op_name_type] = configs[0]
            else:
                op_tuning_cfg[op_name_type] = configs[min(len(configs) - 1, int(params[op_name_type[0]]))]
        calib_sampling_size = calib_sampling_size_lst[min(len(configs) - 1, int(params['calib_sampling_size']))]
        op_tuning_cfg['calib_sampling_size'] = calib_sampling_size
        return op_tuning_cfg

    def next_tune_cfg(self):
        """Yielding the tuning config to traverse by concreting strategies according to last tuning result."""
        while self.experiment.progress.observation_count < self.experiment.observation_budget:
            suggestion = self.conn.experiments(self.experiment.id).suggestions().create()
            yield self.params_to_tune_configs(suggestion.assignments)
            values = [
                dict(name='accuracy', value=self.last_tune_result[0]),
                dict(name='latency', value=self.last_tune_result[1])
            ]
            obs = self.conn.experiments(self.experiment.id).observations().create(
                suggestion=suggestion.id, values=values)
            logger.debug("`suggestion_id` is {}, `observation_id` is {}.".
                format(suggestion.id, obs.id))
            self.experiment = self.conn.experiments(self.experiment.id).fetch()

    def get_acc_target(self, base_acc):
        """Get the tuning target of the accuracy ceiterion."""
        if self.cfg.tuning.accuracy_criterion.relative:
            return base_acc * (1. - self.cfg.tuning.accuracy_criterion.relative)
        else:
            return base_acc - self.cfg.tuning.accuracy_criterion.absolute

    def traverse(self):
        """The main traverse logic, which could be override by some concrete strategy which needs more hooks.

        This is SigOpt version of traverse -- with additional constraints setting to HPO.
        """
        #get fp32 model baseline
        if self.baseline is None:
            logger.info("Get FP32 model baseline.")
            self.baseline = self._evaluate(self.model)
            # record the FP32 baseline
            self._add_tuning_history()

        baseline_msg = '[Accuracy: {:.4f}'.format(self.baseline[0]) + \
            ''.join([', {}: {:.4f}'.format(x,y) for x,y in zip( \
            self.objectives.representation, self.baseline[1]) if x != 'Accuracy']) + ']' \
            if self.baseline else 'n/a'
        logger.info("FP32 baseline is: {}".format(baseline_msg))
        self.experiment = self.create_exp(acc_target=self.get_acc_target(self.baseline[0]))
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

            logger.debug("Dump current tuning configuration:")
            logger.debug(tune_cfg)
            self.last_qmodel = self.adaptor.quantize(
                tune_cfg, self.model, self.calib_dataloader, self.q_func)
            assert self.last_qmodel
            self.last_tune_result = self._evaluate(self.last_qmodel)

            need_stop = self.stop(self.cfg.tuning.exit_policy.timeout, trials_count)

            # record the tuning history
            saved_tune_cfg = copy.deepcopy(tune_cfg)
            saved_last_tune_result = copy.deepcopy(self.last_tune_result)
            self._add_tuning_history(saved_tune_cfg, saved_last_tune_result)

            if need_stop:
                break

    def create_exp(self, acc_target):
        """Set the config for the experiment."""
        params = []
        from copy import deepcopy
        tuning_space = self.tuning_space
        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == 'op':
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, 'fp32', tuning_space)
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options
        # step1. collect the ops that support static and dynamic
        quant_mode_wise_items = OrderedDict()
        query_order = ['static', 'dynamic', 'bf16', 'fp32']
        pre_items = set()
        for quant_mode in query_order:
            items = tuning_space.query_items_by_quant_mode(quant_mode)
            filtered_items = [item for item in items if item not in pre_items]
            pre_items = pre_items.union(set(items))
            quant_mode_wise_items[quant_mode] = filtered_items

        def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict):
            """Initialize the op tuning mode."""
            for item in items_lst:
                op_item_dtype_dict[item.name] = target_quant_mode

        op_item_dtype_dict = OrderedDict()
        for quant_mode, quant_mode_items in quant_mode_wise_items.items():
            initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict)

        op_wise_pool = OpWiseTuningSampler(tuning_space, [], [], 
                                           op_item_dtype_dict, initial_op_tuning_cfg)
        self.op_configs = op_wise_pool.get_opwise_candidate()
        for op, configs in self.op_configs.items():
            if len(configs) > 1:
                params.append(dict(name=op[0], type='int',
                    bounds=dict(min=0, max=len(configs) - 1)))
        params.append(dict(name='calib_sampling_size', type='int',
                      bounds=dict(min=0, max=len(calib_sampling_size_lst) - 1)))
        experiment = self.conn.experiments().create(
            name=self.experiment_name,
            parameters=params,
            metrics=[
                dict(name='accuracy', objective='maximize', strategy='constraint', \
                     threshold=acc_target),
                dict(name='latency', objective='minimize', strategy='optimize'),
            ],
            parallel_bandwidth=1,
            # Define an Observation Budget for your experiment
            observation_budget=100,
            project=self.project_id,
        )

        logger.debug("Create experiment at https://app.sigopt.com/experiment/{}".
                     format(experiment.id))

        return experiment
