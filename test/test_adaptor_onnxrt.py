import os
import shutil
import unittest

import torch
import torchvision
import yaml
import onnx
import numpy as np

from onnx import onnx_pb as onnx_proto
from onnx import helper, TensorProto, numpy_helper
from lpot.adaptor import FRAMEWORKS
from lpot.data import DATASETS, DATALOADERS
from lpot.experimental import Quantization, common
from lpot.experimental import Benchmark, common

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
    with open("static.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)

def build_benchmark_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        evaluation:
          performance:
            warmup: 1
            iteration: 10
            configs:
              num_of_instance: 1
            dataloader:
              batch_size: 1
              dataset:
                ImageFolder:
                  root: /path/to/evaluation/dataset/
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
    with open("benchmark.yaml", "w", encoding="utf-8") as f:
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
    with open("dynamic.yaml", "w", encoding="utf-8") as f:
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
    with open("non_MSE.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)

def eval_func(model):
    return 1.0

def get_torch_version():
    try:
        torch_version = torch.__version__.split('+')[0]
    except ValueError as e:
        assert False, 'Got an unknow version of torch: {}'.format(e)
    return torch_version

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
                    dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                                  "output" : {0 : "batch_size"}})

def build_ir3_model():
    def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
        '''
        Helper function to generate initializers for test inputs
        '''
        tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
        init = numpy_helper.from_array(tensor, input_name)
        return init  

    input0 = helper.make_tensor_value_info('input0', TensorProto.FLOAT, [1, 2048])
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])
    weight = helper.make_tensor_value_info('X1_weight', TensorProto.FLOAT, [1000, 2048])

    X1_weight = generate_input_initializer([1000, 2048], np.float32, 'X1_weight')
    X1_bias = generate_input_initializer([1000], np.float32, 'X1_bias')
    kwargs = {'alpha':1.0, 'beta':1.0, 'transA':0, 'transB':1}
    gemm = helper.make_node('Gemm', ['input0', 'X1_weight'], ['output'], name='gemm', **kwargs)

    graph = helper.make_graph([gemm], 'test_graph_6', [input0], [output])
    graph.initializer.add().CopyFrom(X1_weight)
    graph.initializer.add().CopyFrom(X1_bias)  
    graph.input.extend([weight])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    model.ir_version = 3
    return model

class TestAdaptorONNXRT(unittest.TestCase):

    mb_v2_export_path = "mb_v2.onnx"
    mb_v2_model = torchvision.models.mobilenet_v2()
    rn50_export_path = "rn50.onnx"
    rn50_model = torchvision.models.resnet50()
 
    datasets = DATASETS('onnxrt_qlinearops')
    cv_dataset = datasets['dummy'](shape=(100, 3, 224, 224), low=0., high=1., label=True)
    cv_dataloader = DATALOADERS['onnxrt_qlinearops'](cv_dataset)
    
    ir3_dataset = datasets['dummy'](shape=(10, 2048), low=0., high=1., label=True)
    ir3_dataloader = DATALOADERS['onnxrt_qlinearops'](ir3_dataset)

    @classmethod
    def setUpClass(self):
        build_static_yaml()
        build_dynamic_yaml()
        build_non_MSE_yaml()
        build_benchmark_yaml()
        export_onnx_model(self.mb_v2_model, self.mb_v2_export_path)
        self.mb_v2_model = onnx.load(self.mb_v2_export_path)
        export_onnx_model(self.rn50_model, self.rn50_export_path)
        self.rn50_model = onnx.load(self.rn50_export_path)
        self.ir3_model = build_ir3_model()

    @classmethod
    def tearDownClass(self):
        os.remove("static.yaml")
        os.remove("dynamic.yaml")
        os.remove("non_MSE.yaml")
        os.remove("benchmark.yaml")
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
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type='activation')
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type='activation', save_to_disk=True)
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type='weight')
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type='all')
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, ["Conv_0"], inspect_type='activation')

    def test_set_tensor(self):
        quantizer = Quantization("static.yaml")
        quantizer.calib_dataloader = self.cv_dataloader
        quantizer.eval_dataloader = self.cv_dataloader
        quantizer.model = common.Model(self.mb_v2_model)
        q_model = quantizer()
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
        q_config = {'fused Conv_0': {'weight': {'granularity': 'per_channel', 'dtype': onnx_proto.TensorProto.INT8}}}
        adaptor.q_config = q_config
        version = get_torch_version()
        if version >= '1.7':
            adaptor.set_tensor(q_model, {'545': np.random.random([32, 3, 3, 3])})
            adaptor.set_tensor(q_model, {'546': np.random.random([32])})
        else:
            adaptor.set_tensor(q_model, {'ConvBnFusion_W_features.0.0.weight': np.random.random([32, 3, 3, 3])})
            adaptor.set_tensor(q_model, {'ConvBnFusion_BN_B_features.0.1.bias': np.random.random([32])})

    def test_adaptor(self):
        for fake_yaml in ["static.yaml", "dynamic.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.cv_dataloader
            quantizer.eval_dataloader = self.cv_dataloader
            quantizer.model = common.Model(self.rn50_model)
            q_model = quantizer()
            eval_func(q_model)
        for fake_yaml in ["non_MSE.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.cv_dataloader
            quantizer.eval_dataloader = self.cv_dataloader
            quantizer.model = common.Model(self.mb_v2_model)
            q_model = quantizer()
            eval_func(q_model)

        for fake_yaml in ["static.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.ir3_dataloader
            quantizer.eval_dataloader = self.ir3_dataloader
            quantizer.model = common.Model(self.ir3_model)
            q_model = quantizer()

        for mode in ["performance", "accuracy"]:
            fake_yaml = "benchmark.yaml"
            evaluator = Benchmark(fake_yaml)
            evaluator.b_dataloader = self.cv_dataloader
            evaluator.model = common.Model(self.rn50_model)
            evaluator(mode)


if __name__ == "__main__":
    unittest.main()
