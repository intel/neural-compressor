import os
from pathlib import Path
from datetime import datetime
import pickle
from .conf.config import Conf
from .strategy import STRATEGIES
from .utils import logger
from .utils.create_obj_from_config import create_dataset, create_dataloader
from .utils.create_obj_from_config import update_config
from .data import DataLoader as DATALOADER
from .data import DATASETS, TRANSFORMS
from collections import OrderedDict


class Tuner(object):
    """Tuner class automatically searches for optimal quantization recipes for low precision
       model inference, achieving best tuning objectives like inference performance within
       accuracy loss constraints.

       Tuner abstracts out the differences of quantization APIs across various DL frameworks
       and brings a unified API for automatic quantization that works on frameworks including
       tensorflow, pytorch and mxnet.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria
       (<1% or <0.1% etc.) and tuning objectives (performance, memory footprint etc.).
       Tuner class provides a flexible configuration interface via YAML for users to specify
       these parameters.

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal,
        tuning objective and preferred calibration & quantization tuning space etc.

    """

    def __init__(self, conf_fname):
        self.conf = Conf(conf_fname)

    def tune(self, model, q_dataloader=None, q_func=None,
             eval_dataloader=None, eval_func=None, resume_file=None):
        """The main entry point of automatic quantization tuning.

           This interface works on all the DL frameworks that ilit supports
           and provides three usages:
           a) Fully yaml configuration: User specifies all the info through yaml,
              including dataloaders used in calibration and evaluation phases
              and quantization tuning settings.

              For this usage, only model parameter is mandotory.

           b) Partial yaml configuration: User specifies dataloaders used in calibration
              and evaluation phase by code.
              The tool provides built-in dataloaders and evaluators, user just need provide
              a dataset implemented __iter__ or __getitem__ methods and invoke dataloader()
              with dataset as input parameter before calling tune().

              After that, User specifies fp32 "model", calibration dataset "q_dataloader"
              and evaluation dataset "eval_dataloader".
              The calibrated and quantized model is evaluated with "eval_dataloader"
              with evaluation metrics specified in the configuration file. The evaluation tells
              the tuner whether the quantized model meets the accuracy criteria. If not,
              the tuner starts a new calibration and tuning flow.

              For this usage, model, q_dataloader and eval_dataloader parameters are mandotory.

           c) Partial yaml configuration: User specifies dataloaders used in calibration phase
              by code.
              This usage is quite similar with b), just user specifies a custom "eval_func"
              which encapsulates the evaluation dataset by itself.
              The calibrated and quantized model is evaluated with "eval_func".
              The "eval_func" tells the tuner whether the quantized model meets
              the accuracy criteria. If not, the Tuner starts a new calibration and tuning flow.

              For this usage, model, q_dataloader and eval_func parameters are mandotory.

        Args:
            model (object):                        For Tensorflow model, it could be a path
                                                   to frozen pb,loaded graph_def object or
                                                   a path to ckpt/savedmodel folder.
                                                   For PyTorch model, it's torch.nn.model
                                                   instance.
                                                   For MXNet model, it's mxnet.symbol.Symbol
                                                   or gluon.HybirdBlock instance.
            q_dataloader (generator):              Data loader for calibration, mandatory for
                                                   post-training quantization. It is iterable
                                                   and should yield a tuple (input, label) for
                                                   calibration dataset containing label,
                                                   or yield (input, _) for label-free calibration
                                                   dataset. The input could be a object, list,
                                                   tuple or dict, depending on user implementation,
                                                   as well as it can be taken as model input.
            q_func (function, optional):           Training function for Quantization-Aware
                                                   Training. It is optional and only takes effect
                                                   when user choose "quantization_aware_training"
                                                   approach in yaml.
                                                   This function takes "model" as input parameter
                                                   and executes entire training process with self
                                                   contained training hyper-parameters. If this
                                                   parameter specified, eval_dataloader parameter
                                                   plus metric defined in yaml, or eval_func
                                                   parameter should also be specified at same time.
            eval_dataloader (generator, optional): Data loader for evaluation. It is iterable
                                                   and should yield a tuple of (input, label).
                                                   The input could be a object, list, tuple or
                                                   dict, depending on user implementation,
                                                   as well as it can be taken as model input.
                                                   The label should be able to take as input of
                                                   supported metrics. If this parameter is
                                                   not None, user needs to specify pre-defined
                                                   evaluation metrics through configuration file
                                                   and should set "eval_func" paramter as None.
                                                   Tuner will combine model, eval_dataloader
                                                   and pre-defined metrics to run evaluation
                                                   process.
            eval_func (function, optional):        The evaluation function provided by user.
                                                   This function takes model as parameter,
                                                   and evaluation dataset and metrics should be
                                                   encapsulated in this function implementation
                                                   and outputs a higher-is-better accuracy scalar
                                                   value.

                                                   The pseudo code should be something like:

                                                   def eval_func(model):
                                                        input, label = dataloader()
                                                        output = model(input)
                                                        accuracy = metric(output, label)
                                                        return accuracy
            resume_file (string, optional):        The path to the resume snapshot file.
                                                   The resume snapshot file is saved when user
                                                   press ctrl+c to interrupt tuning process.

        Returns:
            quantized model: best qanitized model found, otherwise return None

        """
        cfg = self.conf.usr_cfg
        self.snapshot_path = os.path.abspath(os.path.expanduser(cfg.snapshot.path))
         
        # when eval_func is set, will be directly used and eval_dataloader can be None
        if eval_func is None:
            if eval_dataloader is None:
                eval_dataloader_cfg = cfg.dataloader if cfg.evaluation is None \
                    else update_config(cfg.evaluation.dataloader, cfg.dataloader)

                if eval_dataloader_cfg is None:
                    self.eval_func = self._fake_eval_func
                    self.eval_dataloader = None
                else:
                    self.eval_dataloader = create_dataloader(cfg.framework.name, \
                                                             eval_dataloader_cfg)
                    self.eval_func = None
            else:
                self.eval_dataloader = eval_dataloader
                self.eval_func = None
        else:
            self.eval_dataloader =None
            self.eval_func = eval_func

        if q_func is None:
            if q_dataloader is None:

                calib_dataloader_cfg = cfg.dataloader if cfg.calibration is None \
                    else update_config(cfg.calibration.dataloader, cfg.dataloader)
                assert calib_dataloader_cfg is not None, \
                    'dataloader field of yaml file is missing'
                self.calib_dataloader = create_dataloader(cfg.framework.name, calib_dataloader_cfg)
                self.q_func = None
            else:
                self.calib_dataloader = q_dataloader
                self.q_func = None
        else:
            self.calib_dataloader =None
            self.q_func = q_func

        strategy = cfg.tuning.strategy.name.lower()
        assert strategy in STRATEGIES, "Tuning strategy {} is NOT supported".format(strategy)

        dicts = None
        # check if interrupted tuning procedure exists. if yes, it will resume the
        # whole auto tune process.
        if resume_file:
            resume_file = os.path.abspath(resume_file)
            assert os.path.exists(
                resume_file), "The specified resume file {} doesn't exist!".format(resume_file)
            with open(resume_file, 'rb') as f:
                resume_strategy = pickle.load(f)
                dicts = resume_strategy.__dict__

        self.strategy = STRATEGIES[strategy](
            model,
            self.conf,
            self.calib_dataloader,
            self.q_func,
            self.eval_dataloader,
            self.eval_func,
            dicts)
        try:
            self.strategy.traverse()
        except KeyboardInterrupt:
            self._save()

        if self.strategy.best_qmodel:
            logger.info(
                "Specified timeout is reached! Found a quantized model which meet accuracy goal. \
                    Exit...")
        else:
            logger.info(
                "Specified timeout is reached! Not found any quantized model which meet accuracy \
                    goal. Exit...")

        return self.strategy.best_qmodel

    def dataset(self, dataset_type, *args, **kwargs):
        return DATASETS(self.conf.usr_cfg.framework.name)[dataset_type](*args, **kwargs)

    def dataloader(self, dataset, batch_size=1, collate_fn=None, last_batch='rollover',
                   sampler=None, batch_sampler=None, num_workers=0, pin_memory=False):

        return DATALOADER(framework=self.conf.usr_cfg.framework.name, dataset=dataset,
                          batch_size=batch_size, collate_fn=collate_fn, last_batch=last_batch,
                          sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers,
                          pin_memory=pin_memory)

    # if user don't config evaluation dataloader give eval_func, create fake eval func
    # only do quantizetion without tuning
    def _fake_eval_func(self, model):
        return 1.

    def _save(self):
        """save current tuning state to snapshot for resuming.
        """
        path = Path(self.snapshot_path)
        path.mkdir(exist_ok=True, parents=True)

        fname = self.snapshot_path + '/ilit-' + datetime.today().strftime(
            '%Y-%m-%d-%H-%M-%S') + '.snapshot'
        with open(fname, 'wb') as f:
            pickle.dump(self.strategy, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("\nSave snapshot to {}".format(fname))
