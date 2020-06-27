import os
from pathlib import Path
from datetime import datetime
import pickle
from .conf.config import Conf
from .strategy import STRATEGIES

class Tuner(object):
    """Tuner class automatically searches for optimal quantization recipes for low precision model inference,
       achieving best tuning objectives like inference performance within accuracy loss constraints.

       Tuner abstracts out the differences of quantization APIs across various DL frameworks and brings a
       unified API for automatic quantization that works on frameworks including tensorflow, pytorch and mxnet.

       Since DL use cases vary in the accuracy metrics (Top-1, MAP, ROC etc.), loss criteria (<1% or <0.1% etc.)
       and tuning objectives (performance, memory footprint etc.). Tuner class provides a flexible configuration
       interface via YAML for users to specify these parameters.

    Args:
        conf_fname (string): The path to the YAML configuration file containing accuracy goal, tuning objective
                             and preferred calibration & quantization tuning space etc.

    """    
    def __init__(self, conf_fname):
        self.cfg  = Conf(conf_fname).cfg

    def tune(self, model, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, resume_file=None):
        """The main entry point of automatic quantization tuning.

           This interface works on all the DL frameworks that iLiT supports and provides two usages:
           a) Calibration and tuning with pre-defined evaluation metrics: User specifies fp32 "model",
              calibration dataset "q_dataloader" and evaluation dataset "eval_dataloader". The calibrated
              and quantized model is evaluated with "eval_dataloader" with evaluation metrics specified
              in the configuration file. The evaluation tells the tuner whether the quantized model meets
              the accuracy criteria. If not, the tuner starts a new calibration and tuning flow.

           b) Calibration and tuning with custom evaluation: User specifies fp32 "model", calibration dataset
              "q_dataloader" and a custom "eval_func" which encapsulates the evaluation dataset by itself.
              The calibrated and quantized model is evaluated with "eval_func". The "eval_func" tells the
              tuner whether the quantized model meets the accuracy criteria. If not, the Tuner starts a new
              calibration and tuning flow.

        Args:
            model (object):                        For Tensorflow model, it's a path to frozen pb or loaded graph_def object.
                                                   For PyTorch model, it's torch.nn.model instance.
                                                   For MXNet model, it's mxnet.symbol.Symbol or gluon.HybirdBlock instance.
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
            resume_file (string, optional):        The path to the resume snapshot file. The resume snapshot file is
                                                   saved when user press ctrl+c to interrupt tuning process.

        Returns:
            quantized model: best qanitized model found, otherwise return None

        """
        self.snapshot_path = self.cfg.snapshot.path if self.cfg.snapshot else './'

        strategy = 'basic'
        if self.cfg.tuning.strategy:
            strategy = self.cfg.tuning.strategy.lower()
            assert strategy.lower() in STRATEGIES, "The tuning strategy {} specified is NOT supported".format(strategy)

        dicts = None
        # check if interrupted tuning procedure exists. if yes, it will resume the whole auto tune process.
        if resume_file:
            resume_file = os.path.abspath(resume_file)
            assert os.path.exists(resume_file), "The specified resume file {} doesn't exist!".format(resume_file)
            with open(resume_file, 'rb') as f:
                resume_strategy = pickle.load(f)
                dicts = resume_strategy.__dict__

        self.strategy = STRATEGIES[strategy](model, self.cfg, q_dataloader, q_func, eval_dataloader, eval_func, dicts)

        try:
            self.strategy.traverse()
        except KeyboardInterrupt:
            self._save()

        if self.strategy.best_qmodel:
            print("Specified timeout is reached! Found a quantized model which meet accuracy goal. Exit...")
        else:
            print("Specified timeout is reached! Not found any quantized model which meet accuracy goal. Exit...")

        return self.strategy.best_qmodel

    def _save(self):
        """save current tuning state to snapshot for resuming.
        """        
        path = Path(self.snapshot_path)
        path.mkdir(exist_ok=True, parents=True)
        
        fname = self.snapshot_path + '/ilit-' + datetime.today().strftime(
            '%Y-%m-%d-%H-%M-%S') + '.snapshot'
        with open(fname, 'wb') as f:
            pickle.dump(self.strategy, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("\nSave snapshot to {}".format(os.path.abspath(fname)))
