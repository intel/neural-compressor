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


def build_ptq_yaml():
    fake_yaml = '''
    model:
        name: resnet18
        framework: pytorch_fx
    tuning:
        strategy:
            name: basic
        accuracy_criterion:
            absolute:  -1
        exit_policy:
            timeout: 0
    '''
    with open('ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)

class TestPytorchAdaptor(unittest.TestCase):
    framework_specific_info = {"device": "cpu",
                               "approach": "post_training_static_quant",
                               "random_seed": 1234,
                               "q_dataloader": None,
                               "workspace_path": None}
    framework = "pytorch"
    adaptor = FRAMEWORKS[framework](framework_specific_info)
    model = torchvision.models.resnet18()

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

    def test_basic_fallback(self):
        def eval_func(model):
          self.i -= 1
          return self.i
          
        from neural_compressor.experimental import Quantization, common
        model = copy.deepcopy(self.model)
        quantizer = Quantization('ptq_yaml.yaml')
        quantizer.eval_func = eval_func
        dataset = quantizer.dataset('dummy', (1, 3, 224, 224), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = model
        q_model = quantizer()
        self.assertTrue(q_model is None)
        
if __name__ == "__main__":
    unittest.main()
