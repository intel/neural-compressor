import copy
import unittest
import numpy as np
import shutil
import torch
import sys
import math
import transformers

sys.path.append('./')

from neural_compressor.data import Datasets, DATALOADERS
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.adaptor.torch_utils.smooth_quant import TorchSmoothQuant
from neural_compressor.adaptor.torch_utils.model_wrapper import SQLinearWrapper
import logging
logger = logging.getLogger("neural_compressor")

try:
    import intel_extension_for_pytorch as ipex
    TEST_IPEX = True
except:
    TEST_IPEX = False


class DemoModel(torch.nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

class DemoCalibDataloader:
    def __init__(self):
        self.batch_size = 1
    def __iter__(self):
        yield torch.randn([1, 3])


class LLMCalibDataloader:
    def __init__(self):
        self.batch_size = 1
    def __iter__(self):
        yield torch.ones([1, 3], dtype=torch.long)


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
            device = torch.device('cpu')
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
        assert torch.sum(torch.abs(output - output_sq)) < 1e-3
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_dw_conv_relu6(self):
        datasets = Datasets('pytorch')
        dummy_dataset = datasets['dummy'](shape=(10, 3, 1, 1), low=0., high=1.0)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        class Model(torch.nn.Module):
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
            device = torch.device('cpu')
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
        assert isinstance(sq.model.fc1, SQLinearWrapper)

    def test_sq_qkv(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
                        'facebook/opt-125m', torchscript=True,)
        sq = TorchSmoothQuant(model, LLMCalibDataloader())
        sq.transform(alpha=0.5, calib_iter=-1, folding=False)
        assert isinstance(
            sq.model.model.decoder.layers[0].self_attn.k_proj, SQLinearWrapper
        )

    def test_sq_quant(self):
        from neural_compressor import PostTrainingQuantConfig, quantization
        class Model(torch.nn.Module):
            device = torch.device('cpu')
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
        assert isinstance(q_model.model.fc1, SQLinearWrapper)

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
                        example_inputs=input_ids,
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
            device = torch.device('cpu')
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

        fp32_model = Model()
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


class TestSqSkipOp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                pass
            def __iter__(self):
                yield torch.rand((1, 4))

        self.linear_dl = RandDataloader()

    @classmethod 
    def test_sq_skip_op_auto(self):
        class Model(torch.nn.Module):
            device = torch.device('cpu')
            def __init__(self):
                super(Model, self).__init__()
                self.linear0 = nn.Linear(4, 4, bias=False)
                self.layernorm1 = nn.LayerNorm(4)
                self.linear1 = nn.Linear(4, 4, bias=False)
                self.ac1 = nn.ReLU()
                self.ac2 = nn.LeakyReLU()
                self.linear2 = nn.Linear(4, 4, bias=True)
                self.linear3 = nn.Linear(4, 2, bias=True)
                self.ac3 = nn.Sigmoid()

            def forward(self, x):
                x = self.linear0(x)
                x1 = self.layernorm1(x)
                x_l1 = self.linear1(x1)
                x_ac1 = self.ac1(x1)
                x_ac2 = self.ac2(x_ac1)
                x_l2 = self.linear2(x1)
                x = x_l1 * x_l2 + x_ac2
                x = self.linear3(x)
                x = self.ac3(x)
                return x
                
        model = Model()
        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha='auto', calib_iter=1, folding=True)
        #the layernorm could not used for sq-absorb because it outputs to an add op.
        assert len(sq.absorb_to_layer) == 0

    def test_sq_no_skip_op_auto(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            'facebook/opt-125m', torchscript=True,
        )
        sq = TorchSmoothQuant(model, LLMCalibDataloader())
        sq.transform(alpha='auto', calib_iter=0, folding=False)
        # folding=False will absorb all Linears with mul, kqv will use same input.
        assert len(sq.absorb_to_layer['model.decoder.layers.2.self_attn.q_proj']) == 3


class TestSqSkipOp_attn(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                pass
            def __iter__(self):
                yield torch.rand((1, 4))
        self.linear_dl = RandDataloader()

    @classmethod 
    def test_sq_skip_op_attn_auto(self):
        class Model(torch.nn.Module):
            device = torch.device('cpu')
            def __init__(self):
                super(Model, self).__init__()
                self.hidden_size = 4
                self.linear0 = nn.Linear(self.hidden_size, self.hidden_size,bias=False)
                self.layernorm1 = nn.LayerNorm(self.hidden_size)
                self.dim_k, self.dim_v = 8, 4
                self.linear_q = nn.Linear(self.hidden_size, self.dim_k, bias=False)
                self.linear_k = nn.Linear(self.hidden_size, self.dim_k, bias=False)
                self.linear_v = nn.Linear(self.hidden_size, self.dim_v, bias=False)   
                self.ac1 = nn.ReLU()
                self.ac2 = nn.LeakyReLU()
                self.linear3 = nn.Linear(self.hidden_size, 3, bias=True)
                self.ac3 = nn.Sigmoid()

            def forward(self, x):
                x = self.linear0(x)
                x = self.layernorm1(x)
                q = self.linear_q(x)
                k = self.linear_k(x)
                v = self.linear_v(x)
                score = torch.matmul(q, k.transpose(1, 0)) / math.sqrt(self.dim_k)
                score = torch.softmax(score, dim=-1)
                attn = torch.matmul(score, v)
                x_ac1 = self.ac1(x)
                x_ac2 = self.ac2(x_ac1)
                x = attn + x_ac2
                x = self.linear3(x)
                x = self.ac3(x)
                return x

                
        model = Model()
        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha='auto', calib_iter=1, folding=True)
        #the layernorm could not used for sq-absorb because it outputs to an add op.
        assert len(sq.absorb_to_layer) == 0 


class TestTuneSqAlpha(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.ns_workspace = "./nc_workspace"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.ns_workspace, ignore_errors=True)

    @unittest.skipIf(not TEST_IPEX, "Please install Intel extension for Pytorch")
    def test_sq_tune_alpha_ipex(self):
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        tuning_criterion = TuningCriterion(max_trials=5)

        for folding in [True, False]:
            for fp32_model, dataloader in [
                (DemoModel(), DemoCalibDataloader()), 
                (
                    transformers.AutoModelForCausalLM.from_pretrained(
                        'facebook/opt-125m', torchscript=True,),
                    LLMCalibDataloader()
                )
            ]:
                conf = PostTrainingQuantConfig(
                    backend='ipex',
                    quant_level=1,
                    tuning_criterion=tuning_criterion,
                    calibration_sampling_size=8,
                    recipes={"smooth_quant": True,
                            "smooth_quant_args": {'folding': folding,
                                                "alpha": np.arange(0.1, 0.4, 0.05).tolist()}
                            }
                )
                eval_result_lst = [0.98, 0.9, 0.8, 0.7, 1.1]
                def fake_eval(model):
                    acc = eval_result_lst.pop(0)
                    return acc

                q_model = quantization.fit(
                    fp32_model,
                    conf,
                    calib_dataloader=dataloader,
                    eval_func=fake_eval
                )

    def test_sq_tune_alpha(self):
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        tuning_criterion = TuningCriterion(max_trials=5)

        for folding in [False]:
            for fp32_model, dataloader in [
                (DemoModel(), DemoCalibDataloader()), 
                (
                    transformers.AutoModelForCausalLM.from_pretrained(
                        'facebook/opt-125m', torchscript=True,),
                    LLMCalibDataloader()
                )
            ]:
                conf = PostTrainingQuantConfig(
                    quant_level=1,
                    tuning_criterion=tuning_criterion,
                    calibration_sampling_size=8,
                    recipes={"smooth_quant": True,
                            "smooth_quant_args": {'folding': folding,
                                                "alpha": np.arange(0.1, 0.4, 0.05).tolist()}
                            }
                )
                eval_result_lst = [0.98, 0.9, 0.8, 0.7, 1.1]
                def fake_eval(model):
                    acc = eval_result_lst.pop(0)
                    return acc
                    
                q_model = quantization.fit(
                    fp32_model,
                    conf,
                    calib_dataloader=dataloader,
                    eval_func=fake_eval
                )
                q_model.save(self.ns_workspace + "saved_result")

    def _test_sq_tune_alpha_common(self, eval_func, alpha=np.arange(0.1, 0.2, 0.05).tolist(), quant_level=1):
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        tuning_criterion = TuningCriterion(max_trials=8)

        fp32_model = DemoModel()
        conf = PostTrainingQuantConfig(
            quant_level=quant_level,
            tuning_criterion=tuning_criterion,
            calibration_sampling_size=8,
            recipes={"smooth_quant": True, 
                     "smooth_quant_args": {'folding': False,
                                           "alpha": alpha,
                                           }
                     }
        )
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=DemoCalibDataloader(),
            eval_func=eval_func,
        )
        q_model.save(self.ns_workspace + "saved_result")

    def test_tune_sq_alpha(self):
        from functools import partial
        def fake_eval(model, eval_result_lst):
            acc = eval_result_lst.pop(0)
            return acc
        
        # test for alpha is a list
        for eval_result_lst, note in [
                ([1, 0.8, 1.1, 0.7, 1.1], "Expect tuning ends at 2nd trial with alpha is 0.15"),
                ([1, 0.8, 0.9, 0.7, 1.1], "Expect tuning ends at 4th trial with alpha is 0.15"),
                ([1, 0.9, 0.8, 0.7, 1.1], "Expect tuning ends at 4th trial with alpha is 0.10")
                ]:
            logger.info(f"test_sq_tune_alpha_common with eval_result_lst: {eval_result_lst}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst = eval_result_lst )
            self._test_sq_tune_alpha_common(partial_fake_eval)
        
        # test for various alphas
        for eval_result_lst, alpha, note in [
                ([1, 0.8, 1.1, 0.7, 1.1], 0.5 ,"Expect tuning ends at 2nd trial with alpha is 0.5 and not tune sq's alpha."),
                ([1, 0.8, 0.9, 0.7, 1.1], [0.5], "Expect tuning ends at 4th trial with alpha is  0.5 and not tune sq's alpha."),
                ([1, 0.9, 0.8, 0.7, 1.1], [0.5, 0.7, 0.9] ,"Expect tuning ends at 4th trial with alpha is 0.5")
                ]:
            logger.info(f"test_sq_tune_alpha_common with eval_result_lst: {eval_result_lst}, alpha: {alpha}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst=eval_result_lst)
            self._test_sq_tune_alpha_common(partial_fake_eval, alpha=alpha)

        # test for quant_level is auto or 0
        for eval_result_lst, alpha, quant_level, note in [
                (
                    [1, 0.8, 1.1, 0.7, 1.1], 
                    np.arange(0.1, 0.2, 0.05).tolist(), 
                    "auto", 
                    "Expect tuning ends at 2nd trial with alpha is 0.15."
                    ),
                (
                    [1, 0.8, 0.9, 0.7, 1.1],
                    np.arange(0.1, 0.2, 0.05).tolist(),
                    "auto",
                    "Expect tuning ends at 4th trial with alpha is  0.15 at basic strategy."
                    ),
                (
                    [1, 1.1, 0.8, 0.7, 1.1], 
                    np.arange(0.1, 0.2, 0.05).tolist(),
                    0,
                    "Expect tuning ends at 1th trial with alpha is 0.1")
                ]:
            logger.info(f"test_sq_tune_alpha_common with ")
            logger.info(f"eval_result_lst: {eval_result_lst}, alpha: {alpha}, quant_level: {quant_level}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst=eval_result_lst)
            self._test_sq_tune_alpha_common(partial_fake_eval, alpha=alpha, quant_level=quant_level)


if __name__ == '__main__':
    unittest.main()
