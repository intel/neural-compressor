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
"""Fefine the tuning strategy that uses tpe search in tuning space."""
import copy
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np

from neural_compressor.strategy.strategy import TuneStrategy, strategy_registry
from neural_compressor.strategy.utils.tuning_sampler import OpWiseTuningSampler
from neural_compressor.strategy.utils.tuning_structs import OpTuningConfig
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

hyperopt = LazyImport("hyperopt")

try:
    import pandas as pd
except ImportError:
    pd = None
    logger.info("Pandas package is required for best result and CSV files generation.")


@strategy_registry
class TpeTuneStrategy(TuneStrategy):
    """The tuning strategy using tpe search in tuning space.

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

    def __init__(
        self,
        model,
        conf,
        q_dataloader=None,
        q_func=None,
        eval_func=None,
        eval_dataloader=None,
        eval_metric=None,
        resume=None,
        q_hooks=None,
    ):
        """Initialize the tpe tuning strategy if the user specified to use it."""
        self.config = self._initialize_config(conf)
        assert (
            self.config.approach == "post_training_static_quant"
        ), "TPE strategy is only for post training static quantization!"
        """Initialize the tpe tuning strategy if the user specified to use it."""
        strategy_name = self.config.tuning_criterion.strategy
        if strategy_name.lower() == "tpe":
            try:
                import hyperopt
            except ImportError:
                try:
                    import subprocess
                    import sys

                    subprocess.check_call([sys.executable, "-m", "pip", "install", "hyperopt"])
                    import hyperopt  # pylint: disable=import-error
                except:
                    assert False, "Unable to import hyperopt from the local environment."
        else:
            pass
        self.hpopt_search_space = None
        self.warm_start = False
        self.cfg_evaluated = False
        self.hpopt_trials = hyperopt.Trials()
        self.max_trials = 200
        if self.config.tuning_criterion.max_trials:
            self.max_trials = self.config.tuning_criterion.max_trials

        self.loss_function_config = {"acc_th": 0.01, "acc_weight": 1.0, "lat_weight": 1.0}
        accuracy_criterion = self.config.accuracy_criterion
        if accuracy_criterion.criterion == "relative":
            self.loss_function_config["acc_th"] = accuracy_criterion.tolerable_loss

        self.tpe_params = {"n_initial_point": 10, "gamma": 0.3, "n_EI_candidates": 100, "prior_weight": 1.0}
        self.best_result = {"best_loss": float("inf"), "best_acc_loss": float("inf"), "best_lat_diff": 0.0}
        self._algo = None

        super().__init__(
            model=model,
            conf=conf,
            q_dataloader=q_dataloader,
            q_func=q_func,
            eval_func=eval_func,
            eval_dataloader=eval_dataloader,
            eval_metric=eval_metric,
            resume=resume,
            q_hooks=q_hooks,
        )

    def __getstate__(self):
        """Magic method for pickle saving.

        Returns:
            dict: Saved dict for resuming
        """
        for history in self.tuning_history:
            if self._same_conf(history["cfg"], self.conf):
                history["warm_start"] = True
                history["hpopt_trials"] = self.hpopt_trials
                history["loss_function_config"] = self.loss_function_config
                history["tpe_params"] = self.tpe_params
                history["hpopt_search_space"] = self.hpopt_search_space
                history["_algo"] = self._algo
        save_dict = super().__getstate__()
        return save_dict

    def _configure_hpopt_search_space_and_params(self, search_space):
        """Set the configuration of hpopt searching strategy."""
        self.hpopt_search_space = {}
        for param, configs in search_space.items():
            self.hpopt_search_space[(param)] = hyperopt.hp.choice((param[0]), configs)
        # Find minimum number of choices for params with more than one choice
        multichoice_params = [len(configs) for param, configs in search_space.items() if len(configs) > 1]
        if not multichoice_params:
            return False
        min_param_size = min(multichoice_params) if len(multichoice_params) > 0 else 1
        self.tpe_params["n_EI_candidates"] = min_param_size
        self.tpe_params["prior_weight"] = 1 / min_param_size
        self._algo = partial(
            hyperopt.tpe.suggest,
            n_startup_jobs=self.tpe_params["n_initial_point"],
            gamma=self.tpe_params["gamma"],
            n_EI_candidates=self.tpe_params["n_EI_candidates"],
            prior_weight=self.tpe_params["prior_weight"],
        )
        return True

    def traverse(self):
        """Tpe traverse logic."""
        logger.info("Start to run tpe strategy.")
        self._prepare_tuning()
        # prepare log file
        trials_file = os.path.join(os.path.dirname(self.history_path), "tpe_trials.csv")
        best_result_file = os.path.join(os.path.dirname(self.history_path), "tpe_best_result.csv")
        logger.debug("trials_file: {} ".format(trials_file) + "best_result_file: {}".format(best_result_file))
        if Path(trials_file).exists():
            os.remove(trials_file)
        status = True
        tuning_history = self._find_self_tuning_history()

        from copy import deepcopy

        tuning_space = self.tuning_space
        initial_op_tuning_cfg = {}
        for item in tuning_space.root_item.options:
            if item.item_type == "op":
                op_name, op_type = item.name
                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, "fp32", tuning_space)
        calib_sampling_size_lst = tuning_space.root_item.get_option_by_name("calib_sampling_size").options
        # step1. collect the ops that support static and dynamic
        quant_mode_wise_items = OrderedDict()
        query_order = ["static", "dynamic", "bf16", "fp16", "fp32"]
        pre_items = set()
        for quant_mode in query_order:
            items = tuning_space.query_items_by_quant_mode(quant_mode)
            filtered_items = [item for item in items if item not in pre_items]
            pre_items = pre_items.union(set(items))
            quant_mode_wise_items[quant_mode] = filtered_items

        def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict):
            for item in items_lst:
                op_item_dtype_dict[item.name] = target_quant_mode

        op_item_dtype_dict = OrderedDict()
        for quant_mode, quant_mode_items in quant_mode_wise_items.items():
            initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict)
        op_wise_pool = OpWiseTuningSampler(tuning_space, [], [], op_item_dtype_dict, initial_op_tuning_cfg)
        self.op_configs = op_wise_pool.get_opwise_candidate()
        self.opwise_tune_cfgs = {}
        for key, val in self.op_configs.items():
            self.opwise_tune_cfgs[key[0]] = val
        self.opwise_tune_cfgs["calib_sampling_size"] = self.tuning_space.root_item.get_option_by_name(
            "calib_sampling_size"
        ).options

        if tuning_history and not self.warm_start:
            # prepare loss function scaling (best result from basic can be used)
            best_lat, worse_acc_loss = 0, 0
            for history in tuning_history["history"]:
                acc_loss, lat_diff = self._calculate_acc_lat_diff(history["tune_result"][0], history["tune_result"][1])
                if lat_diff > best_lat:
                    best_lat = lat_diff
                if acc_loss > worse_acc_loss:
                    worse_acc_loss = acc_loss
            self._calculate_loss_function_scaling_components(worse_acc_loss, best_lat, self.loss_function_config)
            first_run_cfg = self.add_loss_to_tuned_history_and_find_best(tuning_history["history"])
            # Prepare hpopt config with best cfg from history
            self._configure_hpopt_search_space_and_params(first_run_cfg)
            # Run first iteration with best result from history
            trials_count = len(self.hpopt_trials.trials) + 1
            hyperopt.fmin(
                partial(self.object_evaluation, model=self.model),
                space=self.hpopt_search_space,
                algo=self._algo,
                max_evals=trials_count,
                trials=self.hpopt_trials,
                show_progressbar=False,
            )
            if pd is not None:
                self._save_trials(trials_file)
                self._update_best_result(best_result_file)
            # Prepare full hpopt search space
            new_tune_cfgs = self._prepare_final_searchspace(first_run_cfg, self.opwise_tune_cfgs)
            status = self._configure_hpopt_search_space_and_params(new_tune_cfgs)
        elif not self.warm_start:
            self._calculate_loss_function_scaling_components(0.01, 2, self.loss_function_config)
            status = self._configure_hpopt_search_space_and_params(self.opwise_tune_cfgs)

        if status:
            trials_count = len(self.hpopt_trials.trials) + 1
            # get fp32 model baseline
            if self.baseline is None:
                logger.info("Get FP32 model baseline.")
                self.baseline = self._evaluate(self.model)
                self._add_tuning_history()

            baseline_msg = (
                "[Accuracy: {:.4f}".format(self.baseline[0])
                + "".join(
                    [
                        ", {}: {:.4f}".format(x, y)
                        for x, y in zip(self.objectives.representation, self.baseline[1])
                        if x != "Accuracy"
                    ]
                )
                + "]"
                if self.baseline
                else "n/a"
            )
            logger.info("FP32 baseline is: {}".format(baseline_msg))

            if not self.objectives.relative:
                self.loss_function_config["acc_th"] = (self.baseline[0] - self.objectives.acc_goal) / self.baseline[0]
            # start trials
            exit = False
            while not exit:
                self.cfg_evaluated = False
                logger.debug("Trial iteration start: {} / {}.".format(trials_count, self.max_trials))
                hyperopt.fmin(
                    partial(self.object_evaluation, model=self.model),
                    space=self.hpopt_search_space,
                    algo=self._algo,
                    max_evals=trials_count,
                    trials=self.hpopt_trials,
                    show_progressbar=False,
                )
                trials_count += 1
                if pd is not None:
                    self._save_trials(trials_file)
                    self._update_best_result(best_result_file)
                self._save()
                if self.stop(self.config.tuning_criterion.timeout, trials_count):
                    exit = True
        else:
            logger.warn("Can't create search space for input model.")

    def _prepare_final_searchspace(self, first, second):
        """Set the final search space."""
        for key, cfgs in second.items():
            new_cfg = []
            for cfg in cfgs:
                if cfg != first[key][0]:
                    new_cfg.append(cfg)
            first[key] = first[key] + new_cfg
        return first

    def add_loss_to_tuned_history_and_find_best(self, tuning_history_list):
        """Find the best tuned history."""
        logger.debug("Number of resumed configs is {}.".format(len(tuning_history_list)))
        best_loss = None
        first_run_cfg = None
        for history in tuning_history_list:
            result = self._compute_metrics(
                history["tune_cfg"]["op"], history["tune_result"][0], history["tune_result"][1]
            )
            if best_loss is None or result["loss"] < best_loss:
                best_loss = result["loss"]
                first_run_cfg = history["tune_cfg"]["op"].copy()
            result["source"] = "finetune"
            history["result"] = result
            logger.debug(
                "Resumed iteration loss is {}, acc_loss is {}, lat_diff is {}, "
                "quantization_ratio is {}.".format(
                    result["loss"], result["acc_loss"], result["lat_diff"], result["quantization_ratio"]
                )
            )
        for op, cfg in first_run_cfg.items():
            first_run_cfg[op] = [
                cfg,
            ]
        return first_run_cfg

    def object_evaluation(self, tune_cfg, model):
        """Check if config was already evaluated."""
        for k, v in self.op_configs.items():
            tune_cfg.update({k: tune_cfg.pop(k[0])})
        op_cfgs = self._tune_cfg_converter(tune_cfg)
        self.last_qmodel = self.adaptor.quantize(op_cfgs, self.model, self.calib_dataloader)
        self.last_tune_cfg = copy.deepcopy(tune_cfg)
        self.last_tune_result = self._evaluate(self.last_qmodel)
        logger.info("The last tune result is {}.".format((self.last_tune_result[0], self.last_tune_result[1][0])))
        saved_tune_cfg = copy.deepcopy(op_cfgs)
        saved_last_tune_result = copy.deepcopy(self.last_tune_result)
        # prepare result
        result = self._compute_metrics(op_cfgs["op"], self.last_tune_result[0], self.last_tune_result[1][0])
        result["source"] = "tpe"
        self._add_tuning_history(saved_tune_cfg, saved_last_tune_result, result=result)
        logger.info(
            "Current iteration loss is {}, acc_loss is {}, lat_diff is {}, "
            "quantization_ratio is {}.".format(
                result["loss"], result["acc_loss"], result["lat_diff"], result["quantization_ratio"]
            )
        )
        return result

    def _compute_metrics(self, tune_cfg, acc, lat):
        quantization_ratio = 1 - len(
            [param for param in tune_cfg.values() if param["activation"]["dtype"] == "fp32"]
        ) / len(tune_cfg)
        acc_diff, lat_diff = self._calculate_acc_lat_diff(acc, lat)
        return {
            "loss": self.calculate_loss(acc_diff, lat_diff, self.loss_function_config),
            "acc": acc,
            "lat": lat,
            "acc_loss": acc_diff,
            "lat_diff": lat_diff,
            "quantization_ratio": quantization_ratio,
            "status": hyperopt.STATUS_OK,
        }

    def _calculate_acc_lat_diff(self, acc, lat):
        int8_acc = acc
        int8_lat = lat
        fp32_acc = self.baseline[0]
        fp32_lat = self.baseline[1][0]
        acc_diff = (fp32_acc - int8_acc) / fp32_acc
        lat_diff = fp32_lat / int8_lat
        return acc_diff, lat_diff

    def calculate_loss(self, acc_diff, lat_diff, config):
        """Calculate the accuracy loss."""
        gamma_penalty = 40  # penalty term
        acc_loss_component = self._calculate_acc_loss_component(acc_diff)
        lat_loss_component = self._calculate_lat_diff_component(lat_diff)
        acc_weight = config["acc_weight"] if acc_diff > config["acc_th"] else 0.0
        if acc_weight == 0 and config["lat_weight"] == 0:
            acc_weight = 1.0
        loss = acc_weight * (config["acc_scale"] * (acc_loss_component - config["acc_min"])) + config["lat_weight"] * (
            config["lat_scale"] * (lat_loss_component - config["lat_min"])
        )
        if acc_diff > config["acc_th"]:
            loss += 2 * gamma_penalty
        return loss

    def _calculate_acc_loss_component(self, acc_loss):
        return np.exp(acc_loss)

    def _calculate_lat_diff_component(self, lat_diff):
        return np.log(np.power((1 / (1000 * lat_diff)), 8))

    def _calculate_loss_function_scaling_components(self, acc_loss, lat_diff, config):
        acc_min = self._calculate_acc_loss_component(0)
        acc_max = self._calculate_acc_loss_component(acc_loss)
        if acc_max == acc_min:
            acc_max = self._calculate_acc_loss_component(config["acc_th"])
        config["acc_min"] = acc_min
        config["acc_scale"] = 10 / np.abs(acc_max - acc_min)

        lat_min = self._calculate_lat_diff_component(lat_diff)
        lat_max = self._calculate_lat_diff_component(1)
        if lat_min == lat_max:
            lat_min = self._calculate_lat_diff_component(2)
        config["lat_min"] = lat_min
        config["lat_scale"] = 10 / np.abs(lat_max - lat_min)

    def _save_trials(self, trials_log):
        """Save the trial result to the log file."""
        tpe_trials_results = pd.DataFrame(self.hpopt_trials.results)
        csv_file = trials_log
        tpe_trials_results.to_csv(csv_file)

    def _update_best_result(self, best_result_file):
        if not self.hpopt_trials:
            raise Exception("No trials loaded to get best result")
        trials_results = pd.DataFrame(self.hpopt_trials.results)

        if not trials_results[trials_results.acc_loss <= self.loss_function_config["acc_th"]].empty:
            # If accuracy threshold reached, choose best latency
            best_result = (
                trials_results[trials_results.acc_loss <= self.loss_function_config["acc_th"]]
                .reset_index(drop=True)
                .sort_values(by=["lat_diff", "acc_loss"], ascending=[False, True])
                .reset_index(drop=True)
                .loc[0]
            )
        else:
            # If accuracy threshold is not reached, choose based on loss function
            best_result = trials_results.sort_values("loss", ascending=True).reset_index(drop=True).loc[0]

        update_best_result = False
        if not self.best_result["best_loss"]:
            update_best_result = True
        elif self.best_result["best_acc_loss"] <= self.loss_function_config["acc_th"]:
            if (
                best_result["acc_loss"] <= self.loss_function_config["acc_th"]
                and best_result["lat_diff"] > self.best_result["best_lat_diff"]
            ):
                update_best_result = True
        else:
            if (
                best_result["acc_loss"] <= self.loss_function_config["acc_th"]
                or best_result["loss"] < self.best_result["best_loss"]
            ):
                update_best_result = True

        if update_best_result:
            best_result.to_csv(best_result_file, header=False)
            self.best_result["best_loss"] = best_result["loss"]
            self.best_result["best_acc_loss"] = best_result["acc_loss"]
            self.best_result["best_lat_diff"] = best_result["lat_diff"]
            self.best_result["quantization_ratio"] = best_result["quantization_ratio"]

        logger.info(
            "Trial iteration end is {} / {}, best loss is {}, acc_loss is {}, "
            "lat_diff is {}, quantization_ratio is {}.".format(
                len(self.hpopt_trials.trials),
                self.max_trials,
                self.best_result["best_loss"],
                self.best_result["best_acc_loss"],
                self.best_result["best_lat_diff"],
                self.best_result["quantization_ratio"],
            )
        )

    def stop(self, timeout, trials_count):
        """Check if need to stop traversing the tuning space, either accuracy goal is met or timeout is reach.

        Returns:
            bool: True if need stop, otherwise False.
        """
        need_stop = False
        if not self.cfg_evaluated:
            if self.objectives.compare(self.best_tune_result, self.baseline):
                del self.best_tune_result
                del self.best_qmodel
                self.best_tune_result = self.last_tune_result
                self.best_qmodel = self.last_qmodel
                self.adaptor.save(self.best_qmodel, os.path.dirname(self.deploy_path))
            else:
                del self.last_qmodel

        last_tune_msg = (
            "[Accuracy ({}|fp32): {:.4f}|{:.4f}".format("int8", self.last_tune_result[0], self.baseline[0])
            + "".join(
                [
                    ", {} ({}|fp32): {:.4f}|{:.4f}".format(x, "int8", y, z)
                    for x, y, z in zip(self.objectives.representation, self.last_tune_result[1], self.baseline[1])
                    if x != "Accuracy"
                ]
            )
            + "]"
            if self.last_tune_result
            else "n/a"
        )

        best_tune_msg = (
            "[Accuracy: {:.4f}".format(self.best_tune_result[0])
            + "".join(
                [
                    ", {}: {:.4f}".format(x, y)
                    for x, y in zip(self.objectives.representation, self.best_tune_result[1])
                    if x != "Accuracy"
                ]
            )
            + "]"
            if self.best_tune_result
            else "n/a"
        )

        logger.info("Tune {} result is: {}, Best tune result is: {}".format(trials_count, last_tune_msg, best_tune_msg))

        if timeout == 0 and self.best_tune_result:
            need_stop = True
        elif trials_count >= self.config.tuning_criterion.max_trials:
            need_stop = True
        else:
            need_stop = False

        return need_stop
