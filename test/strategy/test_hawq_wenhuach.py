import torch
import unittest
import os
import sys
import copy
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from neural_compressor.data import DATASETS
from neural_compressor.experimental.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.adaptor.pytorch import TemplateAdaptor
from neural_compressor.adaptor import FRAMEWORKS
import shutil
from neural_compressor.strategy.st_utils.hawq_wenhuach import fix_seed
from torch.quantization.quantize_fx import fuse_fx
# fix_seed(1)

def build_ptq_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch_fx
        quantization: 
          calibration:
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
          strategy:
            name: hawq
          accuracy_criterion:
            relative: -0.1
          random_seed: 9527
          exit_policy:
            max_trials: 3
          workspace:
            path: saved
        '''
    with open('ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

class TestPytorchAdaptor(unittest.TestCase):
    framework_specific_info = {"device": "gpu",
                               "approach": "post_training_static_quant",
                               "random_seed": 1234,
                               "q_dataloader": None,
                               "workspace_path": None}
    framework = "pytorch"
    adaptor = FRAMEWORKS[framework](framework_specific_info)
    model = torchvision.models.resnet18()


    # from collections import OrderedDict
    # model = torch.nn.Sequential(OrderedDict([
    #     ('conv1', torch.nn.Conv2d(3, 2, 1, 1)),
    #     ('conv2', torch.nn.Conv2d(2, 1, 1, 1)),
    #     ('flat', torch.nn.Flatten()),
    # ]))
    # model = torch.quantization.QuantWrapper(model)

    @classmethod
    def setUpClass(self):
        self.i = 0
        build_ptq_yaml()


    @classmethod
    def tearDownClass(self):
        os.remove('ptq_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)



    def test_run_hawq_one_trial(self):
        def eval_func(model):
            self.i -= 1
            return self.i
        from neural_compressor.experimental import Quantization, common
        model = copy.deepcopy(self.model)
        model.eval()
        model = fuse_fx(model)
        quantizer = Quantization('ptq_yaml.yaml')
        quantizer.eval_func = eval_func
        dataset = quantizer.dataset('dummy', (32, 3, 224, 224), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = model
        quantizer()

if __name__ == "__main__":

    unittest.main()

# def build_hessian_trace():
#     hessian_trace_config_yaml = '''
#     loss:
#         CrossEntropyLoss
#     random_seed:
#         1
#     max_Iteration:
#         100
#     tolerance:
#         1e-3
#     enable_op_fuse:
#         True
#     max_cal_smaple:
#         100
#     quantize_mode:
#         ptq
#     '''
#     with open('./hessian_trace_config_yaml', 'w+', encoding="utf-8") as f:
#         f.write(hessian_trace_config_yaml)
#
#
# class Test_hessian_trace(unittest.TestCase):
#     # boot up test
#     @classmethod
#     def setUpClass(cls) -> None:
#         build_hessian_trace()
#         cls.model = torchvision.models.resnet18()
#
#     # shotdown test
#     @classmethod
#     def tearDownClass(cls) -> None:
#         os.remove('./hessian_trace_config_yaml')
#
#     # one test case
#     def test_run_hessian_trace(cls):
#         """
#         hessian_trace_top
#         Inputs:
#             model:                      FP32 model
#             dataloader:                 imagenet
#         """
#
#         model = cls.model
#         datasets = DATASETS('pytorch')
#         dummy_dataset = datasets['dummy'](shape=(200, 3, 224, 224), low=0., high=1., label=True)
#         dummy_dataloader = PyTorchDataLoader(dummy_dataset)
#         # yaml_cpu='/home/bfang1/Projects/HAWQ_INC/frameworks.ai.lpot.intel-lpot/neural_compressor/adaptor/pytorch_cpu.yaml'
#         # hessian_cmp=hawq_metric.Hawq_top(model,'./hessian_trace_config_yaml',yaml_cpu,dummy_dataloader)
#         hessian_cmp = Hawq_top(model, yaml_cpu=None, yaml_trace=None, dataloader=dummy_dataloader)
#         tuning_init_config = hessian_cmp.get_init_config()
#         # print tuning init_config
#         for i in tuning_init_config:
#             print(i)


# if __name__ == "__main__":
#     unittest.main()
