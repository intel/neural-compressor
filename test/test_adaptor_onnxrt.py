import os
import shutil
import unittest

import torch
import torchvision
import yaml
import onnx

from lpot.adaptor import FRAMEWORKS


def build_static_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:                                        
          approach: post_training_static_quant  
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
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("static_yaml.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

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
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("dynamic_yaml.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

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
                    opset_version=11,          # the ONNX version to export the model to, please ensure at least 11.
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ["input"],   # the model"s input names
                    output_names = ["output"], # the model"s output names
                    dynamic_axes={"input" : {0 : "batch_size"},    # variable lenght axes
                                  "output" : {0 : "batch_size"}})

class TestAdaptorONNXRT(unittest.TestCase):

    cnn_export_path = "cnn.onnx"
    cnn_model = torchvision.models.quantization.resnet18()

    @classmethod
    def setUpClass(self):
        build_static_yaml()
        build_dynamic_yaml()
        export_onnx_model(self.cnn_model, self.cnn_export_path)
        self.cnn_model = onnx.load(self.cnn_export_path)

    @classmethod
    def tearDownClass(self):
        os.remove("static_yaml.yaml")
        os.remove("dynamic_yaml.yaml")
        os.remove(self.cnn_export_path)
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_adaptor(self):
        framework_specific_info = {"device": "cpu",
                               "approach": "post_training_static_quant",
                               "random_seed": 1234,
                               "q_dataloader": None,
                               "backend": "qlinearops",
                               "workspace_path": None}
        framework = "onnxrt_qlinearops"
        _ = FRAMEWORKS[framework](framework_specific_info)

    def test_quantizate(self):
        from lpot import Quantization
        for fake_yaml in ["static_yaml.yaml", "dynamic_yaml.yaml"]:
            quantizer = Quantization(fake_yaml)
            dataset = quantizer.dataset("dummy", (100, 3, 224, 224), label=True)
            dataloader = quantizer.dataloader(dataset)
            q_model = quantizer(
                self.cnn_model,
                q_dataloader=dataloader,
                eval_dataloader=dataloader
            )
            eval_func(q_model)


if __name__ == "__main__":
    unittest.main()
