import copy
import unittest
import onnx
import numpy as np
import shutil
import torch
from onnx import onnx_pb as onnx_proto
from onnx import helper, TensorProto, numpy_helper
from neural_compressor.data import Datasets, DATALOADERS
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.adaptor.torch_utils.smooth_quant import TorchSmoothQuant
from neural_compressor.adaptor.ox_utils.smooth_quant import ORTSmoothQuant

try:
    import intel_extension_for_pytorch as ipex
    TEST_IPEX = True
except:
    TEST_IPEX = False

def build_onnx_model():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 5, 2])
    H = helper.make_tensor_value_info('H', TensorProto.FLOAT, [1, 5, 2])

    g_value = np.random.uniform(low=0.001, high=0.5, size=(25)).astype(np.float32)
    G_init = helper.make_tensor('G', TensorProto.FLOAT, [5, 5], g_value.reshape(25).tolist())
    matmul_node = onnx.helper.make_node('MatMul', ['A', 'G'], ['C'], name='Matmul')

    b_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    B_init = helper.make_tensor('B', TensorProto.FLOAT, [5, 2], b_value.reshape(10).tolist())
    matmul_node2 = onnx.helper.make_node('MatMul', ['C', 'B'], ['I'], name='Matmul2')

    e_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    E_init = helper.make_tensor('E', TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    matmul_node3 = onnx.helper.make_node('MatMul', ['C', 'E'], ['K'], name='Matmul3')

    add = onnx.helper.make_node('Add', ['I', 'E'], ['D'], name='add')

    f_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    F_init = helper.make_tensor('F', TensorProto.FLOAT, [5, 2], f_value.reshape(10).tolist())
    add2 = onnx.helper.make_node('Add', ['D', 'F'], ['H'], name='add2')

    graph = helper.make_graph([matmul_node, matmul_node2, matmul_node3, add, add2], 'test_graph_1', [A], [H], [B_init, E_init, F_init, G_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    return model

class TestORTSq(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = build_onnx_model()
        dataset = Datasets("onnxrt_qdq")["dummy_v2"]((5,5), (5,1))
        self.dataloader = DATALOADERS['onnxrt_qlinearops'](dataset)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_sq(self):
        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 1)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, folding=False)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 2)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, folding=False, scales_per_op=True)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, scales_per_op=True)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, scales_per_op=True, alpha='auto')
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)


        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, alpha='auto')
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 1)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

class TestSqDepthwiseConv(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                pass

            def __iter__(self):
                yield torch.rand((1, 3, 1, 1))

        self.conv_dl = RandDataloader()

    @classmethod
    def test_sq_dw_conv_relu6_auto(self):
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 1, 1), low=0., high=1.0)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1, 1, groups=3)
                self.act = torch.nn.ReLU6()
                self.conv2 = torch.nn.Conv2d(3, 3, 1, 1, groups=3)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out)
                return out

        model = Model()

        data = torch.rand((1, 3, 1, 1))
        output = model(data)

        sq = TorchSmoothQuant(model, dummy_dataloader)
        sq.transform(alpha='auto', calib_iter=1, folding=True)
        output_sq = model(data)
        assert torch.sum(torch.abs(output - output_sq)) < 1e-5
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_dw_conv_relu6(self):
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 1, 1), low=0., high=1.0)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1, 1)
                self.act = torch.nn.ReLU6()
                self.conv2 = torch.nn.Conv2d(3, 3, 1, 1, groups=3)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out)
                return out

        model = Model()

        data = torch.rand((1, 3, 1, 1))
        output = model(data)

        sq = TorchSmoothQuant(model, dummy_dataloader)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        output_sq = model(data)
        assert torch.sum(torch.abs(output - output_sq)) < 1e-5
        assert len(sq.absorb_to_layer) == 1


class TestSqConvOpFuseAuto(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                pass

            def __iter__(self):
                yield torch.rand((1, 3, 1, 1))

        self.conv_dl = RandDataloader()

    @classmethod
    def test_sq_conv_relu6(self):
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 2, 2), low=0., high=1.0)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 1, 1)
                self.act = torch.nn.ReLU6()
                self.conv2 = torch.nn.Conv2d(4, 3, 1, 1)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, dummy_dataloader)
        sq.transform(alpha='auto', calib_iter=3, folding=True)
        assert len(sq.absorb_to_layer) == 1


class TestSqConvOpFuse(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                pass

            def __iter__(self):
                yield torch.rand((1, 3, 1, 1))

        self.conv_dl = RandDataloader()

    @classmethod
    def test_sq_conv_relu6(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 1, 1)
                self.act = torch.nn.ReLU6()
                self.conv2 = torch.nn.Conv2d(4, 3, 1, 1)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.conv_dl)
        sq.transform(alpha=0.5, folding=True)
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_conv_relu(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 1, 1)
                self.act = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(4, 3, 1, 1)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.conv_dl)
        sq.transform(alpha=0.5, calib_iter=2, folding=True)
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_conv_gelu(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 1, 1)
                self.act = torch.nn.GELU()
                self.conv2 = torch.nn.Conv2d(4, 3, 1, 1)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = self.conv2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.conv_dl)
        sq.transform(alpha=0.5, calib_iter=2, folding=True)
        assert len(sq.absorb_to_layer) == 0

    @classmethod
    def test_sq_conv_bn(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 1, 1)
                self.norm = torch.nn.BatchNorm2d(4)
                self.act = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(4, 3, 1, 1)

            def forward(self, x):
                out = self.conv1(x)
                out = self.norm(out)
                out = self.act(out)
                out = self.conv2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.conv_dl)
        sq.transform(alpha=0.5, calib_iter=2, folding=True)
        assert len(sq.absorb_to_layer) == 1

    def test_sq_conv_gn(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 1, 1)
                self.norm = torch.nn.GroupNorm(num_channels=4, num_groups=2)
                self.act = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(4, 3, 1, 1)

            def forward(self, x):
                out = self.conv1(x)
                out = self.norm(out)
                out = self.act(out)
                out = self.conv2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.conv_dl)
        sq.transform(alpha=0.6, calib_iter=2, folding=True)
        assert len(sq.absorb_to_layer) == 1

    def test_sq_add(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1, 1)
                self.norm = torch.nn.InstanceNorm2d(3)
                self.act = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(3, 3, 1, 1)

            def forward(self, x):
                out = self.conv1(x)
                out = self.act(out)
                out = out + x
                out = self.conv2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.conv_dl)
        sq.transform(alpha=0.6, calib_iter=2, folding=True)
        assert len(sq.absorb_to_layer) == 0


import torch.nn as nn


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class TestSqListInput(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class ListDataloader:
            def __init__(self):
                pass

            def __iter__(self):
                yield [torch.rand((1, 3))]

        class TupleDataloader:
            def __init__(self):
                pass

            def __iter__(self):
                yield (torch.rand((1, 3)))

        self.list_dl = ListDataloader()
        self.tuple_dl = TupleDataloader()

    @classmethod
    def test_sq_linear_LlamaRMSNorm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.norm = LlamaRMSNorm(4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.norm(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.list_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_linear_LlamaRMSNorm_tuple(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.norm = LlamaRMSNorm(4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.norm(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.tuple_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 1


class TestAlphaAutoLinear(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                pass

            def __iter__(self):
                yield torch.rand((1, 3))

        self.linear_dl = RandDataloader()

    @classmethod
    def test_sq_linear_LlamaRMSNorm_auto(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.norm = LlamaRMSNorm(4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.norm(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha='auto', calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 1


class TestSqLinearOpFuse(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                pass

            def __iter__(self):
                yield torch.rand((1, 3))

        self.linear_dl = RandDataloader()

    @classmethod
    def test_sq_linear_LlamaRMSNorm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.norm = LlamaRMSNorm(4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.norm(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_linear_T5Norm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.norm = T5LayerNorm(4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.norm(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_linear_relu6(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.act = torch.nn.ReLU6()
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.act(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_linear_norm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.norm = torch.nn.LayerNorm(4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.norm(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_linear_norm_linear(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.norm_1 = torch.nn.LayerNorm(3)
                self.fc1 = torch.nn.Linear(3, 4)
                self.norm_2 = torch.nn.LayerNorm(4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.norm_1(x)
                out = self.fc1(out)
                out = self.norm_2(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 2

    @classmethod
    def test_sq_linear_gelu_norm(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.norm = torch.nn.LayerNorm(4)
                self.act = torch.nn.GELU()
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.norm(out)
                out = self.act(out)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 0

    def test_sq_linear(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha=0.5, calib_iter=1) # By default, folding=False
        from neural_compressor.adaptor.torch_utils.model_wrapper import SQLinearWrapper
        assert isinstance(sq.model.fc1, SQLinearWrapper)

    def test_sq_quant(self):
        from neural_compressor import PostTrainingQuantConfig, quantization
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                return out

        input_ids = torch.randn([2, 3])
        fp32_model = Model()
        conf = PostTrainingQuantConfig(
            calibration_sampling_size=8,
            recipes={"smooth_quant": True, 
                     "smooth_quant_args": {'alpha': 'auto', 'folding': False}}
        )#  By default, folding args: {IPEX: False, ONNX RT: False, Stock PT: True}
        class CalibDataloader:
            def __init__(self):
                self.batch_size = 1
            def __iter__(self):
                yield input_ids
        def calib_func(model):
            for i in range(10):
                model(input_ids)

        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=CalibDataloader(),
            eval_func=lambda x: 0.1,
        )
        from neural_compressor.adaptor.torch_utils.model_wrapper import SQLinearWrapper
        assert isinstance(q_model.model.fc1, SQLinearWrapper)
        assert isinstance(fp32_model.fc1.weight, torch.Tensor)
        assert isinstance(fp32_model.fc1, SQLinearWrapper) # for smoothquant, inplace=True.

        q_model.save('saved_result')
        from neural_compressor.utils.pytorch import load
        model_origin = Model()
        qdq_model = load("./saved_result", model_origin)

        fp32_model = Model()
        origin_bias = float(fp32_model.fc1.bias[0])
        conf = PostTrainingQuantConfig(
            calibration_sampling_size=8,
            recipes={"smooth_quant": True, 
                     "smooth_quant_args": {'alpha': 'auto'}}
        )#  By default, folding args: {IPEX: False, ONNX RT: False, Stock PT: True}
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=CalibDataloader(),
            eval_func=lambda x: 0.1,
        )
        self.assertTrue(float(q_model.model.fc1.bias()[0]) != origin_bias)

        # with calib_func
        conf = PostTrainingQuantConfig(
                        recipes={"smooth_quant": True,
                                "smooth_quant_args": {'alpha': 'auto', 'folding': False}}
                        )
        fp32_model = Model()
        q_model = quantization.fit(
                        fp32_model,
                        conf,
                        calib_func=calib_func,
                        eval_func=lambda x: 0.1,
                        )
        self.assertTrue(isinstance(q_model.model.fc1, SQLinearWrapper))

    @unittest.skipIf(not TEST_IPEX, "Please install Intel extension for Pytorch")
    def test_sq_quant_ipex(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                return out

        input_ids = torch.randn([1, 3])
        fp32_model = Model()
        output1 = fp32_model(input_ids)

        from neural_compressor import PostTrainingQuantConfig, quantization
        conf = PostTrainingQuantConfig(
            backend="ipex",
            calibration_sampling_size=8,
            excluded_precisions=['bf16'],
            example_inputs=(input_ids,),
            recipes={"smooth_quant": True, "smooth_quant_args": {'alpha': 'auto'}}
        )
        def calib_func(model):
            model(input_ids)

        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_func=calib_func,
        )

        conf = PostTrainingQuantConfig(
            backend="ipex",
            calibration_sampling_size=8,
            excluded_precisions=['bf16'],
            recipes={"smooth_quant": True, "smooth_quant_args": {'alpha': 0.5, 'folding': True}}
        )
        class CalibDataloader:
            def __init__(self):
                self.batch_size = 1
            def __iter__(self):
                yield input_ids
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=CalibDataloader(),
        )
        output2 = q_model.model(input_ids)


if __name__ == '__main__':
    unittest.main()
