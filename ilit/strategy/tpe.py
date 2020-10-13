from .strategy import strategy_registry, TuneStrategy
import numpy as np
import os
import copy
from pathlib import Path
from ..utils.utility import Timeout
from ..utils import logger
import hyperopt as hpo
from hyperopt import fmin, hp, STATUS_OK, Trials
from functools import partial
try:
    import pandas as pd
except ImportError:
    pd = None
    logger.info('Pandas package is required for best result and CSV files generation.')


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
                                               file and should set "eval_func" paramter as None.
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
                 eval_dataloader=None, eval_func=None, dicts=None):
        assert conf.usr_cfg.quantization.approach == 'post_training_static_quant', \
               "TPE strategy is only for post training static quantization!"
        self.hpopt_search_space = None
        self.warm_start = False
        self.hpopt_trials = Trials()
        self.max_trials = conf.usr_cfg.tuning.get('max_trials', 200)
        self.loss_function_config = {
            'acc_th': conf.usr_cfg.tuning.accuracy_criterion.relative if \
                      conf.usr_cfg.tuning.accuracy_criterion and \
                      conf.usr_cfg.tuning.accuracy_criterion.relative else 0.01,
            'acc_weight': conf.usr_cfg.tuning.strategy.get('accuracy_weight', 1.0),
            'lat_weight': conf.usr_cfg.tuning.strategy.get('latency_weight', 1.0)
        }
        self.tpe_params = {
            'n_initial_point': 10,
            'gamma': 0.3,
            'n_EI_candidates': 100,
            'prior_weight': 1.0
        }
        self.best_result = {
            'best_loss': float('inf'),
            'best_acc_loss': float('inf'),
            'best_lat_diff': 0.0
        }
        self._algo = None

        super(
            TpeTuneStrategy,
            self).__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts)

    def __getstate__(self):
        """Magic method for pickle saving.

        Returns:
            dict: Saved dict for resuming
        """
        for history in self.tuning_history:
            if self._same_yaml(history['cfg'], self.cfg):
                history['warm_start'] = True
                history['hpopt_trials'] = self.hpopt_trials
                history['loss_function_config'] = self.loss_function_config
                history['tpe_params'] = self.tpe_params
        save_dict = super(TpeTuneStrategy, self).__getstate__()
        return save_dict

    def _configure_hpopt_search_space_and_params(self, search_space):
        self.hpopt_search_space = {}
        for param, configs in search_space.items():
            self.hpopt_search_space[(param)] = hp.choice((param[0]), configs)
        # Find minimum number of choices for params with more than one choice
        multichoice_params = [len(configs) for param, configs in search_space.items()
                              if len(configs) > 1]
        min_param_size = min(multichoice_params) if len(multichoice_params) > 0 else 1
        self.tpe_params['n_EI_candidates'] = min_param_size
        self.tpe_params['prior_weight'] = 1 / min_param_size
        self._algo = partial(hpo.tpe.suggest,
                            n_startup_jobs=self.tpe_params['n_initial_point'],
                            gamma=self.tpe_params['gamma'],
                            n_EI_candidates=self.tpe_params['n_EI_candidates'],
                            prior_weight=self.tpe_params['prior_weight'])
    def traverse(self):
        """Tpe traverse logic.

        """
        logger.info('Start tpe strategy')
        # prepare log file
        tmp_path = self.cfg.tuning.snapshot.get('path', './')
        Path(os.path.expanduser(tmp_path)).mkdir(parents=True, exist_ok=True)
        trials_file = os.path.join(os.path.expanduser(tmp_path), 'tpe_trials.csv')
        best_result_file = os.path.join(os.path.expanduser(tmp_path), 'tpe_best_result.csv')
        logger.info('trials_file: {} '.format(trials_file) + \
                    'best_result_file:{}'.format(best_result_file))
        if Path(trials_file).exists():
            os.remove(trials_file)
        self._configure_hpopt_search_space_and_params(self.opwise_tune_cfgs)
        with Timeout(self.cfg.tuning.timeout) as t:
            trials_count = len(self.hpopt_trials.trials) + 1
            # get fp32 model baseline
            if self.baseline is None:
                logger.info('Getting FP32 model baseline...')
                self.baseline = self._evaluate(self.model)
                self._add_tuning_history()
            logger.info('FP32 baseline is: ' + ('[{:.4f}, {:.4f}]'.format(*self.baseline)
                                                if self.baseline else 'None'))
            if not self.objective.relative:
                self.loss_function_config['acc_th'] =\
                    (self.baseline[0] - self.objective.acc_goal) / self.baseline[0]
            # prepare loss function scaling
            self._calculate_loss_function_scaling_components(0.01, 2, self.loss_function_config)
            # start trials
            exit = False
            while not exit:
                logger.info('Trial iteration start: {} / {}'.format(trials_count, self.max_trials))
                fmin(partial(self.object_evaluation, model=self.model),
                    space=self.hpopt_search_space,
                    algo=self._algo,
                    max_evals=trials_count,
                    trials=self.hpopt_trials,
                    show_progressbar=False)
                trials_count += 1
                if pd is not None:
                    self._save_trials(trials_file)
                    self._update_best_result(best_result_file)
                self._save(tmp_path)
                if self.stop(t, trials_count):
                    exit = True

    def object_evaluation(self, tune_cfg, model):
        # check if config was alredy evaluated
        op_cfgs = {}
        op_cfgs['calib_iteration'] = int(self.calib_iter[0])
        op_cfgs['op'] = {}
        for param, configs in tune_cfg.items():
            op_cfgs['op'][(param)] = configs
        tuning_history = self._find_tuning_history(op_cfgs)
        if tuning_history:
            self.last_tune_result = tuning_history['last_tune_result']
            self.best_tune_result = tuning_history['best_tune_result']
            self.last_qmodel = None
            logger.debug('This tuning config was evaluated!')
            return tuning_history['result']

        self.last_qmodel = self.adaptor.quantize(op_cfgs, self.model, self.calib_dataloader)
        self.last_tune_result = self._evaluate(self.last_qmodel)
        logger.info('last_tune_result: {}'.format(self.last_tune_result))

        saved_tune_cfg = copy.deepcopy(op_cfgs)
        saved_last_tune_result = copy.deepcopy(self.last_tune_result)

        # prepare result
        result = self._compute_metrics(tune_cfg)
        result['status'] = STATUS_OK
        self._add_tuning_history(saved_tune_cfg, saved_last_tune_result, result=result)
        logger.info('Current iteration loss: {} acc_loss: {} lat_diff: {} quantization_ratio: {}'
                    .format(result['loss'],
                            result['acc_loss'],
                            result['lat_diff'],
                            result['quantization_ratio']))
        return result

    def _compute_metrics(self, tune_cfg):
        quantization_ratio = 100 - len([param for param in tune_cfg.values()
                                        if param['activation']['dtype'] =='fp32']) / len(tune_cfg)
        int8_acc = self.last_tune_result[0]
        int8_lat = self.last_tune_result[1]
        fp32_acc = self.baseline[0]
        fp32_lat = self.baseline[1]
        acc_diff = (fp32_acc - int8_acc) / fp32_acc
        lat_diff = fp32_lat / int8_lat
        return {
            'loss': self.calculate_loss(acc_diff, lat_diff, self.loss_function_config),
            'acc' : int8_acc,
            'lat' : int8_lat,
            'acc_loss': acc_diff,
            'lat_diff': lat_diff,
            'quantization_ratio': quantization_ratio}

    def calculate_loss(self, acc_diff, lat_diff, config):
        gamma_penalty = 40  # penalty term
        acc_loss_component = self._calculate_acc_loss_component(acc_diff)
        lat_loss_component = self._calculate_lat_diff_component(lat_diff)
        acc_weight = config['acc_weight'] if acc_diff > config['acc_th'] else 0.0
        if acc_weight == 0 and config['lat_weight'] == 0:
            acc_weight = 1.0
        loss = acc_weight * (config['acc_scale'] * (acc_loss_component - config['acc_min'])) \
               + config['lat_weight']\
               * (config['lat_scale'] * (lat_loss_component - config['lat_min']))
        if acc_diff > config['acc_th']:
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
            acc_max = self._calculate_acc_loss_component(config['acc_th'])
        config['acc_min'] = acc_min
        config['acc_scale'] = 10 / np.abs(acc_max - acc_min)

        lat_min = self._calculate_lat_diff_component(lat_diff)
        lat_max = self._calculate_lat_diff_component(1)
        if lat_min == lat_max:
            lat_min = self._calculate_lat_diff_component(2)
        config['lat_min'] = lat_min
        config['lat_scale'] = 10 / np.abs(lat_max - lat_min)

    def _save_trials(self, trials_log):
        """ save trial result to log file"""
        tpe_trials_results = pd.DataFrame(self.hpopt_trials.results)
        csv_file = trials_log
        tpe_trials_results.to_csv(csv_file)

    def _update_best_result(self, best_result_file):
        if not self.hpopt_trials:
            raise Exception(
                'No trials loaded to get best result')
        trials_results = pd.DataFrame(self.hpopt_trials.results)

        if not trials_results[trials_results.acc_loss <=
                              self.loss_function_config['acc_th']].empty:
            # If accuracy threshold reached, choose best latency
            best_result = trials_results[trials_results.acc_loss <=
                                         self.loss_function_config['acc_th']] \
                .reset_index(drop=True).sort_values(by=['lat_diff', 'acc_loss'],
                                                    ascending=[False, True]) \
                .reset_index(drop=True).loc[0]
        else:
            # If accuracy threshold is not reached, choose based on loss function
            best_result = \
                trials_results.sort_values('loss', ascending=True).reset_index(drop=True).loc[0]

        update_best_result = False
        if not self.best_result['best_loss']:
            update_best_result = True
        elif self.best_result['best_acc_loss'] <= self.loss_function_config['acc_th']:
            if best_result['acc_loss'] <= self.loss_function_config['acc_th'] \
                    and best_result['lat_diff'] > self.best_result['best_lat_diff']:
                update_best_result = True
        else:
            if best_result['acc_loss'] <= self.loss_function_config['acc_th'] or \
                    best_result['loss'] < self.best_result['best_loss']:
                update_best_result = True

        if update_best_result:
            best_result.to_csv(best_result_file, header=False)
            self.best_result['best_loss'] = best_result['loss']
            self.best_result['best_acc_loss'] = best_result['acc_loss']
            self.best_result['best_lat_diff'] = best_result['lat_diff']
            self.best_result['quantization_ratio'] = best_result['quantization_ratio']

        logger.info('Trial iteration end: {} / {} best loss: {} acc_loss: {} lat_diff: {} '
                    'quantization_ratio: {}'.format(len(self.hpopt_trials.trials), self.max_trials,
                                                        self.best_result['best_loss'],
                                                        self.best_result['best_acc_loss'],
                                                        self.best_result['best_lat_diff'],
                                                        self.best_result['quantization_ratio']))

    def stop(self, timeout, trials_count):
        """Check if need to stop traversing the tuning space, either accuracy goal is met
           or timeout is reach.

        Args:
            timeout (Timeout): The timeout object instantiated in utils.py

        Returns:
            bool: True if need stop, otherwise False
        """
        need_stop = False

        if self.objective.compare(self.best_tune_result, self.baseline):
            del self.best_tune_result
            del self.best_qmodel
            self.best_tune_result = self.last_tune_result
            self.best_qmodel = self.last_qmodel
        else:
            del self.last_qmodel

        logger.info(
            'Tune result is: ' +
            ('[{:.4f}, {:.4f}]'.format(
                *self.last_tune_result) if self.last_tune_result else 'None') +
            ' Best tune result is: ' +
            ('[{:.4f}, {:.4f}]'.format(
                *self.best_tune_result) if self.best_tune_result else 'None'))

        if timeout.seconds != 0 and timeout.timed_out:
            need_stop = True
        elif timeout.seconds == 0 and self.best_tune_result:
            need_stop = True
        elif trials_count > self.max_trials:
            need_stop = True
        else:
            need_stop = False

        return need_stop
