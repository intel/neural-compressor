import os
import shutil
import unittest

import torch
import torchvision
import yaml
import onnx

from lpot.adaptor import FRAMEWORKS
from lpot.data import DATASETS, DATALOADERS

def build_static_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:                                        
          approach: post_training_static_quant  
          calibration:
            sampling_size: 50
          op_wise: {
            'Gather_*': {
            'activation':  {'dtype': ['fp32'], 'scheme':['sym']},
            'weight': {'dtype': ['fp32'], 'scheme':['sym']}
            }
          }                                       

        evaluation:
          accuracy:
            metric:
              topk: 1

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("static_yaml.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)

def build_dynamic_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_integerops

        quantization:                                        
          approach: post_training_dynamic_quant 
          calibration:
              sampling_size: 50

        evaluation:
          accuracy:
            metric:
              topk: 1

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("dynamic_yaml.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)

def build_non_MSE_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:
          approach: post_training_static_quant
          calibration:
              sampling_size: 50
          op_wise: {
            'Gather_*': {
            'activation':  {'dtype': ['fp32'], 'scheme':['sym']},
            'weight': {'dtype': ['fp32'], 'scheme':['sym']}
            }
          }

        evaluation:
          accuracy:
            metric:
              MSE: 
               compare_label: False
          performance:
            warmup: 5
            iteration: 10

        tuning:
          accuracy_criterion:
            relative:  0.1
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("non_MSE_yaml.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)

def eval_func(model):
    return 1.0

def export_onnx_model(model, path):
    x = torch.randn(100, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    path,                      # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to, please ensure at least 11.
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ["input"],   # the model"s input names
                    output_names = ["output"], # the model"s output names
                    dynamic_axes={"input" : {0 : "batch_size"},    # variable lenght axes
                                  "output" : {0 : "batch_size"}})

class TestAdaptorONNXRT(unittest.TestCase):

    mb_v2_export_path = "mb_v2.onnx"
    mb_v2_model = torchvision.models.mobilenet_v2()
    rn50_export_path = "rn50.onnx"
    rn50_model = torchvision.models.resnet50()
 
    datasets = DATASETS('onnxrt_qlinearops')
    cv_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
    cv_dataloader = DATALOADERS['onnxrt_qlinearops'](cv_dataset)

    @classmethod
    def setUpClass(self):
        build_static_yaml()
        build_dynamic_yaml()
        build_non_MSE_yaml()
        export_onnx_model(self.mb_v2_model, self.mb_v2_export_path)
        self.mb_v2_model = onnx.load(self.mb_v2_export_path)
        export_onnx_model(self.rn50_model, self.rn50_export_path)
        self.rn50_model = onnx.load(self.rn50_export_path)

    @classmethod
    def tearDownClass(self):
        os.remove("static_yaml.yaml")
        os.remove("dynamic_yaml.yaml")
        os.remove("non_MSE_yaml.yaml")
        os.remove(self.mb_v2_export_path)
        os.remove(self.rn50_export_path)
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_adaptor(self):
        framework_specific_info = {"device": "cpu",
                               "approach": "post_training_static_quant",
                               "random_seed": 1234,
                               "q_dataloader": None,
                               "backend": "qlinearops",
                               "workspace_path": './lpot_workspace/{}/{}/'.format(
                                                       'onnxrt',
                                                       'imagenet')}
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, ["Conv"])

    def test_quantizate(self):
        from lpot.experimental import Quantization, common
        for fake_yaml in ["static_yaml.yaml", "dynamic_yaml.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.cv_dataloader
            quantizer.eval_dataloader = self.cv_dataloader
            quantizer.model = common.Model(self.rn50_model)
            q_model = quantizer()
            eval_func(q_model)
        for fake_yaml in ["non_MSE_yaml.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.cv_dataloader
            quantizer.eval_dataloader = self.cv_dataloader
            quantizer.model = common.Model(self.mb_v2_model)
            q_model = quantizer()
            eval_func(q_model)


if __name__ == "__main__":
    unittest.main()
