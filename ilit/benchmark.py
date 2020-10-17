import os
from .adaptor import FRAMEWORKS
from .objective import OBJECTIVES
from .conf.config import Conf
from .utils import logger
from .utils.create_obj_from_config import create_eval_func, create_dataset, create_dataloader

from .data import DataLoader as DATALOADER

class Benchmark(object):
    """Benchmark class can be used to evaluate the model performance, with the objective
       setting, user can get the data of what they configured in yaml

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        tuning objective and preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)

    def __call__(self, model, b_dataloader=None):
        cfg = self.conf.usr_cfg
        framework_specific_info = {'device': cfg.device, \
                                   'approach': cfg.quantization.approach, \
                                   'random_seed': cfg.tuning.random_seed}
        if cfg.framework.name.lower() == 'tensorflow':
            framework_specific_info.update({"inputs": cfg.framework.inputs, \
                                            "outputs": cfg.framework.outputs})
        if cfg.framework.name.lower() == 'mxnet':
            framework_specific_info.update({"b_dataloader": b_dataloader})

        framework = cfg.framework.name.lower()
        adaptor = FRAMEWORKS[framework](framework_specific_info)

        if cfg.evaluation and cfg.evaluation.performance and cfg.evaluation.performance.iteration:
            iteration = cfg.evaluation.performance.iteration
        else:
            iteration = -1 

        if cfg.evaluation and cfg.evaluation.accuracy and cfg.evaluation.accuracy.metric:
            metric = cfg.evaluation.accuracy.metric
        else:
            metric = None 

        if b_dataloader is None:
            assert cfg.evaluation is not None and cfg.evaluation.performance is not None \
                   and cfg.evaluation.performance.dataloader is not None, \
                   'dataloader field of yaml file is missing'

            b_dataloader_cfg = cfg.evaluation.performance.dataloader
            b_dataloader = create_dataloader(framework, b_dataloader_cfg)
            b_postprocess_cfg = cfg.evaluation.performance.postprocess
            b_func = create_eval_func(cfg.framework.name, \
                                      b_dataloader, \
                                      adaptor, \
                                      metric, \
                                      b_postprocess_cfg,
                                      iteration=iteration)
        else:
            b_func = create_eval_func(cfg.framework.name, \
                                      b_dataloader, \
                                      adaptor, \
                                      metric, \
                                      iteration=iteration)

        objective = cfg.tuning.objective.lower()
        self.objective = OBJECTIVES[objective](cfg.tuning.accuracy_criterion, \
                                               is_measure=True)

        val = self.objective.evaluate(b_func, model)
        # measurer contain info not only performance(eg, memory, model_size)
        # also measurer have result list among steps
        acc, _ = val
        batch_size = b_dataloader.batch_size
        return acc, batch_size, self.objective.measurer

