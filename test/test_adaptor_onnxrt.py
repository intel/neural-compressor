import os
import shutil
import unittest

import torch
import torchvision
import yaml
import onnx
import numpy as np
from collections import OrderedDict
from onnx import onnx_pb as onnx_proto
from onnx import helper, TensorProto, numpy_helper
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.data import DATASETS, DATALOADERS
from neural_compressor.experimental import Quantization, common
from neural_compressor.experimental import Benchmark, common

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
          workspace: 
            path: ./nc_workspace/recover/
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

def build_gather_yaml():
    fake_yaml = """
        model:
          name: imagenet
          framework: onnxrt_qlinearops

        quantization:                                        
          approach: post_training_static_quant 
          calibration:
              sampling_size: 1
        evaluation:
          accuracy:
            metric:
              Accuracy: {}

        tuning:
          accuracy_criterion:
            relative:  -0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("gather.yaml", "w", encoding="utf-8") as f:
        f.write(fake_yaml)

def build_rename_yaml():
    fake_yaml = """
        model:
          name: test
          framework: onnxrt_integerops

        quantization:                                        
          approach: post_training_dynamic_quant 
          calibration:
              sampling_size: 1

        evaluation:
          accuracy:
            metric:
              Accuracy: {}

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 0
          random_seed: 9527
        """
    with open("rename.yaml", "w", encoding="utf-8") as f:
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

def export_onnx_model(model, path, opset=12):
    x = torch.randn(100, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    path,                      # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=opset,          # the ONNX version to export the model to, please ensure at least 11.
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ["input"],   # the model"s input names
                    output_names = ["output"], # the model"s output names
                    dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                                  "output" : {0 : "batch_size"}})

def generate_input_initializer(tensor_shape, tensor_dtype, input_name):
    '''
    Helper function to generate initializers for test inputs
    '''
    tensor = np.random.ranf(tensor_shape).astype(tensor_dtype)
    init = numpy_helper.from_array(tensor, input_name)
    return init  

def build_ir3_model():
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

def build_matmul_model():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 5, 1])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 1])
    matmul_node = onnx.helper.make_node('MatMul', ['A', 'B'], ['C'], name='Matmul')
    graph = helper.make_graph([matmul_node], 'test_graph_1', [A, B], [C])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    return model

def build_model_with_gather():
    b_value = np.random.randint(2, size=(10)).astype(np.int32)
    B_init = helper.make_tensor('B', TensorProto.INT32, [10], b_value.reshape(10).tolist())
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 100, 4])
    D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [100, 4])
    squeeze = onnx.helper.make_node('Squeeze', ['A'], ['D'], name='squeeze')
    B = helper.make_tensor_value_info('B', TensorProto.INT32, [10])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [10, 4])
    node = onnx.helper.make_node('Gather', ['D', 'B'], ['C'], name='gather')
    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    E_init = helper.make_tensor('E', TensorProto.FLOAT, [10, 1], e_value.reshape(10).tolist())
    F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [10, 4])
    add = onnx.helper.make_node('Add', ['C', 'E'], ['F'], name='add')
    graph = helper.make_graph([squeeze, node, add], 'test_graph_1', [A], [F], [B_init, E_init])
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    return model

def build_rename_model():
    b_value = np.random.randint(2, size=(10)).astype(np.int32)
    B_init = helper.make_tensor('B', TensorProto.INT32, [10], b_value.reshape(10).tolist())
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 100, 4])
    D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [100, 4])
    squeeze = onnx.helper.make_node('Squeeze', ['A'], ['D'], name='')
    B = helper.make_tensor_value_info('B', TensorProto.INT32, [10])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [10, 4])
    node = onnx.helper.make_node('Gather', ['D', 'B'], ['C'], name='')
    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    E_init = helper.make_tensor('E', TensorProto.FLOAT, [10, 1], e_value.reshape(10).tolist())
    F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [10, 4])
    add = onnx.helper.make_node('Add', ['C', 'E'], ['F'], name='')
    graph = helper.make_graph([squeeze, node, add], 'test_graph_1', [A], [F], [B_init, E_init])
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    return model

class MatmulDataset:
    def __init__(self):
        self.data = []
        self.label = []
        for i in range(3):
            self.data.append([np.random.randn(1,5,5).astype('float32'), 
                              np.random.randn(1,5,1).astype('float32')])
            self.label.append(np.random.randn(1,5,1).astype('float32'))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

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

    gather_dataset = DATASETS('onnxrt_qlinearops')['dummy'](shape=(5, 100, 4), label=True)
    gather_dataloader = DATALOADERS['onnxrt_qlinearops'](gather_dataset)

    rename_dataloader = gather_dataloader

    matmul_dataset = MatmulDataset()
    matmul_dataloader = DATALOADERS['onnxrt_qlinearops'](matmul_dataset)

    @classmethod
    def setUpClass(self):
        build_rename_yaml()
        build_static_yaml()
        build_dynamic_yaml()
        build_gather_yaml()
        build_non_MSE_yaml()
        build_benchmark_yaml()
        export_onnx_model(self.mb_v2_model, self.mb_v2_export_path)
        self.mb_v2_model = onnx.load(self.mb_v2_export_path)
        export_onnx_model(self.rn50_model, self.rn50_export_path)
        export_onnx_model(self.rn50_model, 'rn50_9.onnx', 9)
        self.rn50_model = onnx.load(self.rn50_export_path)
        self.ir3_model = build_ir3_model()
        self.gather_model = build_model_with_gather()
        self.matmul_model = build_matmul_model()
        self.rename_model = build_rename_model()

    @classmethod
    def tearDownClass(self):
        os.remove("static.yaml")
        os.remove("dynamic.yaml")
        os.remove("non_MSE.yaml")
        os.remove("benchmark.yaml")
        os.remove("gather.yaml")
        os.remove("rename.yaml")
        os.remove("rn50_9.onnx")
        os.remove(self.mb_v2_export_path)
        os.remove(self.rn50_export_path)
        os.remove("best_model.onnx")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_inspect_tensor(self):
        framework_specific_info = {"device": "cpu",
                               "approach": "post_training_static_quant",
                               "random_seed": 1234,
                               "q_dataloader": None,
                               "backend": "qlinearops",
                               "workspace_path": './nc_workspace/{}/{}/'.format(
                                                       'onnxrt',
                                                       'imagenet')}
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info)
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type='activation')
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type='activation', save_to_disk=True)
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type='weight')
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, inspect_type='all')
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, ["Conv_0"], inspect_type='activation')
        op_list = OrderedDict()
        op_list[("Conv_0", "Conv")] = None
        adaptor.inspect_tensor(self.rn50_model, self.cv_dataloader, op_list.keys(), inspect_type='activation')

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
                     "workspace_path": './nc_workspace/{}/{}/'.format(
                                             'onnxrt',
                                             'imagenet')}
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info) 
        q_config = {'fused Conv_0': {'weight': {'granularity': 'per_channel', 'dtype': onnx_proto.TensorProto.INT8, 'scheme': 'sym'}}}
        adaptor.quantize_config = q_config
        version = get_torch_version()
        q_model.save('./best_model.onnx')
        if version >= '1.7':
            adaptor.set_tensor(onnx.load("best_model.onnx"), 
                {self.mb_v2_model.graph.node[0].input[1]: np.random.random([32, 3, 3, 3])})
            adaptor.set_tensor(q_model,
                {self.mb_v2_model.graph.node[0].input[2]: np.random.random([32])})
        else:
            adaptor.set_tensor(onnx.load("best_model.onnx"), 
                {'ConvBnFusion_W_features.0.0.weight': np.random.random([32, 3, 3, 3])})
            adaptor.set_tensor(q_model, {'ConvBnFusion_BN_B_features.0.1.bias': np.random.random([32])})


    def test_adaptor(self):
        for fake_yaml in ["rename.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.rename_dataloader
            quantizer.eval_dataloader = self.rename_dataloader
            quantizer.model = common.Model(self.rename_model)
            q_model = quantizer()

        for fake_yaml in ["static.yaml", "dynamic.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.cv_dataloader
            quantizer.eval_dataloader = self.cv_dataloader
            quantizer.model = common.Model(self.rn50_model)
            q_model = quantizer()
            eval_func(q_model)

        import copy
        tmp_model = copy.deepcopy(self.rn50_model)
        tmp_model.opset_import[0].version = 10
        quantizer.model = common.Model(tmp_model)
        q_model = quantizer()
        tmp_model.opset_import.extend([onnx.helper.make_opsetid("", 11)]) 
        quantizer.model = common.Model(tmp_model)
        q_model = quantizer()
        model = onnx.load('rn50_9.onnx')
        quantizer.model = common.Model(model)
        q_model = quantizer()

        framework_specific_info = {"device": "cpu",
                     "approach": "post_training_static_quant",
                     "random_seed": 1234,
                     "q_dataloader": None,
                     "backend": "qlinearops",
                     "workspace_path": './nc_workspace/{}/{}/'.format(
                                             'onnxrt',
                                             'imagenet')}
        framework = "onnxrt_qlinearops"
        adaptor = FRAMEWORKS[framework](framework_specific_info) 
        tune_cfg = {'calib_iteration': 1,
                    'op': {('gather', 'Gather'): {'activation':  {'dtype': ['uint8']},
                                                 'weight': {'dtype': ['uint8']}},
                           ('add', 'Add'): {'activation':  {'dtype': ['uint8']},
                                           'weight': {'dtype': ['int8']}}}}
        adaptor.quantize(tune_cfg, common.Model(self.gather_model), self.gather_dataloader)
        self.assertTrue(len(adaptor.quantizable_ops), 2)
 
        for fake_yaml in ["gather.yaml"]:
            quantizer = Quantization(fake_yaml)
            quantizer.calib_dataloader = self.gather_dataloader
            quantizer.eval_dataloader = self.gather_dataloader
            quantizer.model = common.Model(self.gather_model)
            q_model = quantizer()

            quantizer.model = common.Model(self.matmul_model)
            q_model = quantizer()

            quantizer.eval_dataloader = self.matmul_dataloader
            q_model = quantizer()

            quantizer.calib_dataloader = self.matmul_dataloader
            quantizer.eval_dataloader = self.matmul_dataloader
            quantizer.model = common.Model(self.matmul_model)
            q_model = quantizer()

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

            from neural_compressor.utils.utility import recover
            model = recover(self.ir3_model, './nc_workspace/recover/history.snapshot', 0)
            self.assertTrue(model.model == q_model.model)

        for mode in ["performance", "accuracy"]:
            fake_yaml = "benchmark.yaml"
            evaluator = Benchmark(fake_yaml)
            evaluator.b_dataloader = self.cv_dataloader
            evaluator.model = common.Model(self.rn50_model)
            evaluator(mode)


if __name__ == "__main__":
    unittest.main()
