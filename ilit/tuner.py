import os
import sys
from datetime import datetime
import pickle
from .conf import Conf
from .strategy import STRATEGIES

class Tuner(object):
    '''The auto tuner class provides unified low precision quantization interface cross frameworks, including
       tensorflow, pytorch and mxnet.

       As low precision model usually has precision loss comparing with fp32 model, it requires additional manual
       tune effort. This auto tuner class implements different tuning strategies to traverse all possible tuning
       space and figure out best low precision model which can meet goal.

       Args:
           conf_fname (string): The name of configuration file.
    '''
    def __init__(self, conf_fname):
        self.cfg  = Conf(conf_fname).cfg
        self._customized_ops = None
        self._inputs = None
        self._outputs = None

    def tune(self, model, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, model_specific_cfg=dict()):
        '''The main entry of the auto tuning interface.

           This interface provides a unified low precision quantization tuning interface cross supported framework backends.

           There are two usages for this interface:
           a) simple
              User provides the fp32 model and calibration data loader and evaluation data loader through model and q_dataloader
              and eval_dataloader paramters of this API, and specifies ilit supported evaluation metric through yaml
              configuration file.

           b) advance
              User provides the fp32 model, calibration data loader through model and q_dataloader parameters of this API
              and provides evaluation function by eval_func parameter.

           Args:
               model (object): If user chooses tensorflow backend, it's frozen pb or graph_def object in tensorflow.
                               If user chooses pytorch backend, it's torch.nn.model instantiate object.
                               If user chooses mxnet backend, it's mxnet.symbol.Symbol or gluon.HybirdBlock
                               instantiate object.

               q_dataloader (object): The calibration data feeder and it is iterable. It should yield input and label tensor tuple
                                      for each iteration.
                                      The input tensor should be taken as the input of model, and the label tensor should be able
                                      to take as input of supported metrics.

               q_func (optional): Reserved for furture use.

               eval_dataloader (optional): The evaluation data feeder and it is iterable.It should yield input and label tensor tuple
                                           for each iteration.
                                           The input tensor should be taken as the input of model, and the label tensor should be able
                                           to take as input of supported metrics.
                                           If this parameter is not None, user need specify evaluation metric through yaml tuning
                                           configuration file and should set eval_func paramter as None. Auto tuner will combine model
                                           and eval_dataloader and supported metrics to run evaluation process.

               eval_func (optional): The evaluation function provided by user. This function takes model as parameter, and evaluation
                                     dataset should be encapsuled in this function implementation and outputs a higher-is-better
                                     accuracy scalar value.

                                     Its pyseudo code should be something like:
                                     def eval_func(model):
                                         ...
                                         output = model(input)
                                         accuracy = metric(output, label)
                                         ...
                                         return accuracy

               model_specific_cfg (optional): The dict of model specific configuration.
                                              It includes two parts:
                                              1. inputs/outputs related info required by tensorflow if needed.
                                              2. customized_ops defined by user to set constraint on specified ops.
                                              {
                                                'resume_file': '/path/to/resume/file'
                                                'customized_ops':
                                                   {
                                                     'op1': {
                                                       'activation':  {'data_type': ['uint8', 'fp32'], 'algo': ['minmax', 'kl'], 'mode':['sym']},
                                                       'weight': {'data_type': ['int8', 'fp32'], 'algo': ['kl']}
                                                     },
                                                     'op2': {
                                                       'activation': {'data_type': ['int8'], 'mode': ['sym'], 'granularity': ['per_tensor'], 'algo': ['minmax', 'kl']},
                                                     },
                                                     'op3': {
                                                       'activation':  {'data_type': ['fp32']},
                                                       'weight': {'data_type': ['fp32']}
                                                     },
                                                     ...
                                                   },
                                                'inputs': ['input'],
                                                'outputs': ['resnet_v1_101/predictions/Reshape_1']
                                              }

        '''
        if self.cfg.snapshot:
            self.snapshot = os.path.dirname(str(self.cfg.snapshot))
        else:
            self.snapshot = './'

        strategy = 'basic'
        if self.cfg.tuning.strategy:
            strategy = self.cfg.tuning.strategy.lower()
            assert strategy.lower() in STRATEGIES, "The tuning strategy {} specified is NOT supported".format(strategy)

        self.cfg.customized_ops = model_specific_cfg.get('customized_ops', None)
        self.cfg.inputs = model_specific_cfg.get('inputs', None)
        self.cfg.outputs = model_specific_cfg.get('outputs', None)
        self.cfg.resume_file = model_specific_cfg.get('resume_file', None)

        dicts = None
        # check if interrupted tuning procedure exists. if yes, it will resume the whole auto tune process.
        if self.cfg.resume_file:
            assert os.path.exists(self.cfg.resume_file), "The specified resume file {} doesn't exist!".format(self.cfg.resume_file)
            with open(self.cfg.resume_file, 'rb') as f:
                resume_strategy = pickle.load(f)
                dicts = resume_strategy.__dict__

        self.strategy = STRATEGIES[strategy](model, self.cfg, q_dataloader, q_func, eval_dataloader, eval_func, dicts)

        try:
            self.strategy.traverse()
        except KeyboardInterrupt:
            self._save()

        if self.best_qmodel:
            print("Specified timeout is reached! Found a quantized model which meet accuracy goal. Exit...")
        else:
            print("Specified timeout is reached! Not found any quantized model which meet accuracy goal. Exit...")

        return self.best_qmodel

    def _save(self):
        '''restore the tuning process if interrupted

           Return: dict to contain all info needed by resume

        '''
        path = self.snapshot
        if not os.path.exists(path):
            os.makedirs(path)

        fname = path + '/ilit-' + datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.snapshot'
        with open(fname, 'wb') as f:
            pickle.dump(self.strategy, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("\nSave snapshot to {}".format(fname))

