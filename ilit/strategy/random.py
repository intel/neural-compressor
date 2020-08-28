from .strategy import strategy_registry, TuneStrategy
import numpy as np


@strategy_registry
class RandomTuneStrategy(TuneStrategy):
    """The tuning strategy using random search in tuning space.

    Args:
        model (object):                        The FP32 model specified for low precision tuning.
        conf (Conf):                           The Conf class instance initialized from user yaml config file.
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

    def __init__(self, model, conf, q_dataloader, q_func=None,
                 eval_dataloader=None, eval_func=None, dicts=None):
        super(
            RandomTuneStrategy,
            self).__init__(
            model,
            conf,
            q_dataloader,
            q_func,
            eval_dataloader,
            eval_func,
            dicts)

    def next_tune_cfg(self):
        """The generator of yielding next tuning config to traverse by concrete strategies
           according to last tuning result.

        """
        # generate tuning space according to user chosen tuning strategy

        np.random.seed(self.cfg.tuning.random_seed)
        while True:
            op_cfgs = {}
            op_cfgs['calib_iteration'] = int(np.random.choice(self.calib_iter))
            op_cfgs['op'] = {}
            for op, configs in self.opwise_quant_cfgs.items():
                op_cfgs['op'][op] = np.random.choice(configs)

            yield op_cfgs
