import copy
import math
import shutil
import sys
import unittest

import numpy as np
import torch
import transformers
from packaging.version import Version

sys.path.append("./")

import logging

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.adaptor.torch_utils.model_wrapper import SQLinearWrapper
from neural_compressor.adaptor.torch_utils.smooth_quant import TorchSmoothQuant
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader

logger = logging.getLogger("neural_compressor")

try:
    import intel_extension_for_pytorch as ipex

    TEST_IPEX = True
    IPEX_VERSION = Version(ipex.__version__)
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
        self.batch_size = 3

    def __iter__(self):
        for i in range(4):
            yield torch.ones([3, 3], dtype=torch.long)


class TestSqDepthwiseConv(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield torch.rand((1, 3, 1, 1))

        self.conv_dl = RandDataloader()

    @classmethod
    def test_sq_dw_conv_relu6_auto(self):
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(10, 3, 1, 1), low=0.0, high=1.0)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
        sq.transform(alpha="auto", calib_iter=1, folding=True)
        output_sq = model(data)
        assert torch.sum(torch.abs(output - output_sq)) < 1e-3
        assert len(sq.absorb_to_layer) == 1

    @classmethod
    def test_sq_dw_conv_relu6(self):
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(10, 3, 1, 1), low=0.0, high=1.0)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
                self.batch_size = 1

            def __iter__(self):
                yield torch.rand((1, 3, 1, 1))

        self.conv_dl = RandDataloader()

    @classmethod
    def test_sq_conv_relu6(self):
        datasets = Datasets("pytorch")
        dummy_dataset = datasets["dummy"](shape=(10, 3, 2, 2), low=0.0, high=1.0)
        dummy_dataloader = PyTorchDataLoader(dummy_dataset)

        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
        sq.transform(alpha="auto", calib_iter=3, folding=True)
        assert len(sq.absorb_to_layer) == 1


class TestSqConvOpFuse(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield torch.rand((1, 3, 1, 1))

        self.conv_dl = RandDataloader()

    @classmethod
    def test_sq_conv_relu6(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
            device = torch.device("cpu")

            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 1, 1)
                self.norm = torch.nn.GroupNorm(num_channels=4, num_groups=2)
                self.act = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(4, 3, 1, 1)
                self.conv3 = torch.nn.Conv2d(4, 3, 1, 1)

            def forward(self, x):
                out = self.conv1(x)
                out = self.norm(out)
                out = self.act(out)
                tmp1 = self.conv2(out)
                tmp2 = self.conv3(out)
                out = tmp1 + tmp2
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.conv_dl)
        sq.transform(alpha=0.6, calib_iter=2, folding=True)
        assert len(sq.absorb_to_layer["norm"]) == 2

    def test_sq_add(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
        """LlamaRMSNorm is equivalent to T5LayerNorm."""
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
        """Construct a layernorm module in the T5 style.

        No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus variance is calculated
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
                self.batch_size = 1

            def __iter__(self):
                yield [torch.rand((1, 3))]

        class TupleDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield (torch.rand((1, 3)))

        class ListTupleDataLoader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                input1 = torch.rand((1, 3))
                input2 = torch.rand((1, 3))
                yield [input1, ((input2, input1)), input2]

        self.list_dl = ListDataloader()
        self.tuple_dl = TupleDataloader()
        self.list_tuple_dl = ListTupleDataLoader()

    @classmethod
    def test_sq_linear_LlamaRMSNorm(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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

    @classmethod
    def test_sq_linear_LlamaRMSNorm_list_tuple(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.fc1_1 = torch.nn.Linear(3, 4)
                self.fc1_2 = torch.nn.Linear(3, 4)
                self.fc1_3 = torch.nn.Linear(4, 4)
                self.fc2 = torch.nn.Linear(4, 4)
                self.norm = LlamaRMSNorm(4)
                self.fc3 = torch.nn.Linear(4, 3)

            def forward(self, x1, x_tuple, x4):
                x2, x3 = x_tuple
                out1 = self.fc1_1(x1 + x4)
                out2 = self.fc1_2(x2 + x3)
                out = out1 + out2
                out = self.fc1_3(out)
                out = self.fc2(out)
                out = self.norm(out)
                out = self.fc3(out)
                return out

        model = Model()
        sq = TorchSmoothQuant(model, self.list_tuple_dl)
        sq.transform(alpha=0.5, calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 2

    def test_device(self):
        input1 = torch.rand((1, 3))
        input2 = torch.rand((1, 3))
        example_input = {"k": [input1, ((input2, input1)), input2]}
        from neural_compressor.adaptor.torch_utils.smooth_quant import move_input_to_device

        move_input_to_device(example_input)


class TestAlphaAutoLinear(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield torch.rand((1, 3))

        self.linear_dl = RandDataloader()

    @classmethod
    def test_sq_linear_LlamaRMSNorm_auto(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
        sq.transform(alpha="auto", calib_iter=1, folding=True)
        assert len(sq.absorb_to_layer) == 1


class TestSqLinearOpFuse(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield torch.rand((1, 3))

        self.linear_dl = RandDataloader()

    @classmethod
    def test_sq_linear_LlamaRMSNorm(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
            device = torch.device("cpu")

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
        sq.transform(alpha=0.5, calib_iter=1)  # By default, folding=False
        assert isinstance(sq.model.fc1, SQLinearWrapper)

    def test_sq_trace_failure(self):
        class Output:
            out = None

        class Model(torch.nn.Module):
            device = torch.device("cpu")

            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                Output.out = out
                return Output

        model = Model()
        sq = TorchSmoothQuant(model, self.linear_dl)
        sq.transform(alpha=0.5, calib_iter=1)  # By default, folding=False
        assert isinstance(sq.model.fc1, SQLinearWrapper)

    def test_sq_qkv(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
        )
        sq = TorchSmoothQuant(model, LLMCalibDataloader())
        sq.transform(alpha=0.5, calib_iter=-1, folding=False)
        assert isinstance(sq.model.model.decoder.layers[0].self_attn.k_proj, SQLinearWrapper)


class TestExample(unittest.TestCase):
    def test_sq_quant(self):
        from neural_compressor import PostTrainingQuantConfig, quantization

        class Model(torch.nn.Module):
            device = torch.device("cpu")

            def __init__(self):
                super(Model, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                return out

        input_ids = torch.randn([3, 3])
        fp32_model = Model()
        output1 = fp32_model(input_ids)

        conf = PostTrainingQuantConfig(
            calibration_sampling_size=8,
            recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": "auto", "folding": False}},
        )  #  By default, folding args: {IPEX: False, ONNX RT: False, Stock PT: True}

        class CalibDataloader:
            def __init__(self):
                self.batch_size = 3

            def __iter__(self):
                for i in range(4):
                    yield input_ids

        def calib_func(model):
            for i in range(10):
                model(input_ids)

        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=CalibDataloader(),
        )
        output2 = q_model.model(input_ids)
        assert isinstance(q_model.model.fc1, SQLinearWrapper)
        # set a big atol to avoid random issue
        self.assertTrue(torch.allclose(output1, output2, atol=2e-02))

        q_model.save("saved_result")
        from neural_compressor.utils.pytorch import load

        model_origin = Model()
        qdq_model = load("./saved_result", model_origin)

        fp32_model = Model()
        origin_bias = float(fp32_model.fc1.bias[0])
        conf = PostTrainingQuantConfig(
            calibration_sampling_size=8,
            recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": "auto", "folding": True}},
        )  #  By default, folding args: {IPEX: False, ONNX RT: False, Stock PT: True}
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
            recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": "auto", "folding": False}},
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
            device = torch.device("cpu")

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

        def calib_func(model):
            model(input_ids)

        ipex_version = Version(ipex.__version__.split("+")[0])
        # pure ipex quantization
        if ipex_version >= Version("2.1.0"):
            qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5)
            from intel_extension_for_pytorch.quantization import convert, prepare

            user_model = copy.deepcopy(fp32_model)
            user_model = prepare(user_model.eval(), qconfig, example_inputs=input_ids, inplace=True)
            calib_func(user_model)
            user_model.save_qconf_summary(qconf_summary="ipex.json")
            import json

            with open("ipex.json", "r") as f:
                ipex_config_json = json.load(f)
            with torch.no_grad():
                user_model = convert(user_model.eval(), inplace=True).eval()
                user_model(input_ids)
                user_model = torch.jit.trace(user_model.eval(), input_ids, strict=False)
                user_model = torch.jit.freeze(user_model.eval())
                user_model(input_ids)
                user_model(input_ids)
            ipex_out = user_model(input_ids)
            # inc quantization
            conf = PostTrainingQuantConfig(
                backend="ipex",
                calibration_sampling_size=8,
                excluded_precisions=["bf16"],
                example_inputs=(input_ids,),
                recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}},
            )
            tmp_model = copy.deepcopy(fp32_model)
            q_model = quantization.fit(
                tmp_model,
                conf,
                calib_func=calib_func,
            )
            q_model.save("saved")
            # test recover_model_from_json
            from neural_compressor.utils.pytorch import recover_model_from_json

            tmp_model = copy.deepcopy(fp32_model)

            ipex_model = recover_model_from_json(tmp_model, "./saved/best_configure.json", example_inputs=input_ids)
            inc_output = q_model.model(input_ids)
            ipex_output = ipex_model(input_ids)
            self.assertTrue(torch.allclose(inc_output, ipex_output, atol=1e-05))

            example_tuple = (input_ids,)
            ipex_model = recover_model_from_json(tmp_model, "./saved/best_configure.json", example_inputs=example_tuple)
            ipex_output = ipex_model(input_ids)
            self.assertTrue(torch.allclose(inc_output, ipex_output, atol=1e-05))

            example_dict = {"x": input_ids}
            ipex_model = recover_model_from_json(tmp_model, "./saved/best_configure.json", example_inputs=example_dict)
            ipex_output = ipex_model(input_ids)
            self.assertTrue(torch.allclose(inc_output, ipex_output, atol=1e-05))

            # compare ipex and inc quantization
            with open("saved/best_configure.json", "r") as f:
                inc_config_json = json.load(f)
            inc_out = q_model.model(input_ids)
            ipex_sq_weight_scale = torch.tensor(
                ipex_config_json[" "]["q_op_infos"]["0"]["weight_tensor_infos"][0]["smooth_quant_scaling_factor"]
            )
            inc_sq_weight_scale = torch.tensor(
                inc_config_json[" "]["q_op_infos"]["0"]["weight_tensor_infos"][0]["smooth_quant_scaling_factor"]
            )
            self.assertTrue(torch.allclose(inc_sq_weight_scale, ipex_sq_weight_scale))
            # set a big atol to avoid random issue
            self.assertTrue(torch.allclose(ipex_out, inc_out, atol=2e-02))
            self.assertTrue(torch.allclose(output1, inc_out, atol=2e-02))

        class CalibDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield input_ids

        conf = PostTrainingQuantConfig(
            backend="ipex",
            calibration_sampling_size=8,
            excluded_precisions=["bf16"],
            recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": "auto"}},
        )
        tmp_model = copy.deepcopy(fp32_model)
        q_model = quantization.fit(
            tmp_model,
            conf,
            calib_dataloader=CalibDataloader(),
        )
        output2 = q_model.model(input_ids)
        # set a big atol to avoid random issue
        self.assertTrue(torch.allclose(output1, output2, atol=2e-02))

        conf = PostTrainingQuantConfig(
            backend="ipex",
            calibration_sampling_size=8,
            excluded_precisions=["bf16"],
            recipes={"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5, "folding": True}},
        )
        tmp_model = copy.deepcopy(fp32_model)
        q_model = quantization.fit(
            tmp_model,
            conf,
            calib_dataloader=CalibDataloader(),
        )
        output2 = q_model.model(input_ids)
        # set a big atol to avoid random issue
        self.assertTrue(torch.allclose(output1, output2, atol=2e-02))


class TestSqSkipOp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield torch.rand((1, 4))

        self.linear_dl = RandDataloader()

    @classmethod
    def test_sq_skip_op_auto(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
        sq.transform(alpha="auto", calib_iter=1, folding=True)
        # the layernorm could not used for sq-absorb because it outputs to an add op.
        assert len(sq.absorb_to_layer) == 0

    def test_sq_no_skip_op_auto(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
        )
        sq = TorchSmoothQuant(model, LLMCalibDataloader())
        sq.transform(alpha="auto", calib_iter=0, folding=False)
        # folding=False will absorb all Linears with mul, kqv will use same input.
        assert len(sq.absorb_to_layer["model.decoder.layers.2.self_attn.q_proj"]) == 3


class TestSqSkipOp_attn(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield torch.rand((1, 4))

        self.linear_dl = RandDataloader()

    @classmethod
    def test_sq_skip_op_attn_auto(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

            def __init__(self):
                super(Model, self).__init__()
                self.hidden_size = 4
                self.linear0 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
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
        sq.transform(alpha="auto", calib_iter=1, folding=True)
        # the layernorm could not used for sq-absorb because it outputs to an add op.
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
                        "facebook/opt-125m",
                        torchscript=True,
                    ),
                    LLMCalibDataloader(),
                ),
            ]:
                conf = PostTrainingQuantConfig(
                    backend="ipex",
                    quant_level=1,
                    tuning_criterion=tuning_criterion,
                    calibration_sampling_size=8,
                    recipes={
                        "smooth_quant": True,
                        "smooth_quant_args": {"folding": folding, "alpha": np.arange(0.1, 0.4, 0.05).tolist()},
                    },
                )
                eval_result_lst = [0.98, 0.9, 0.8, 0.7, 1.1]

                def fake_eval(model):
                    acc = eval_result_lst.pop(0)
                    return acc

                q_model = quantization.fit(fp32_model, conf, calib_dataloader=dataloader, eval_func=fake_eval)

    def test_sq_tune_alpha(self):
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion

        tuning_criterion = TuningCriterion(max_trials=5)

        for folding in [False, True]:
            for fp32_model, dataloader in [
                (DemoModel(), DemoCalibDataloader()),
                (
                    transformers.AutoModelForCausalLM.from_pretrained(
                        "facebook/opt-125m",
                        torchscript=True,
                    ),
                    LLMCalibDataloader(),
                ),
            ]:
                conf = PostTrainingQuantConfig(
                    quant_level=1,
                    tuning_criterion=tuning_criterion,
                    calibration_sampling_size=8,
                    recipes={
                        "smooth_quant": True,
                        "smooth_quant_args": {"folding": folding, "alpha": np.arange(0.1, 0.4, 0.05).tolist()},
                    },
                )
                eval_result_lst = [0.98, 0.9, 0.8, 0.7, 1.1]

                def fake_eval(model):
                    acc = eval_result_lst.pop(0)
                    return acc

                q_model = quantization.fit(fp32_model, conf, calib_dataloader=dataloader, eval_func=fake_eval)
                q_model.save(self.ns_workspace + "saved_result")

    def _test_sq_tune_alpha_common(self, eval_func, alpha=np.arange(0.1, 0.2, 0.05).tolist(), quant_level=1):
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion

        logger.info(f"alpha is: {alpha}")

        tuning_criterion = TuningCriterion(max_trials=8)

        fp32_model = DemoModel()
        conf = PostTrainingQuantConfig(
            quant_level=quant_level,
            tuning_criterion=tuning_criterion,
            calibration_sampling_size=8,
            recipes={
                "smooth_quant": True,
                "smooth_quant_args": {
                    "folding": False,
                    "alpha": alpha,
                },
            },
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
            ([1, 0.8, 0.9, 0.7, 1.1], "Expect tuning ends at 2nd trial with alpha is 0.15"),
            ([1, 0.9, 0.8, 0.7, 1.1], "Expect tuning ends at 1st trial with alpha is 0.10"),
        ]:
            logger.info(f"test_sq_tune_alpha_common with eval_result_lst: {eval_result_lst}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst=eval_result_lst)
            self._test_sq_tune_alpha_common(partial_fake_eval)

        # test for various alphas
        for eval_result_lst, alpha, note in [
            (
                [1, 0.8, 1.1, 0.7, 1.1],
                0.5,
                "Expect tuning ends at 2nd trial with alpha is 0.5 and not tune sq's alpha.",
            ),
            (
                [1, 0.8, 0.9, 0.7, 1.1],
                [0.5],
                "Expect tuning ends at 4th trial with alpha is  0.5 and not tune sq's alpha.",
            ),
            ([1, 0.9, 0.8, 0.7, 1.1], [0.5, 0.7, 0.9], "Expect tuning ends at 4th trial with alpha is 0.5"),
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
                "Expect tuning ends at 2nd trial with alpha is 0.15.",
            ),
            (
                [1, 0.8, 0.9, 0.7, 1.1],
                np.arange(0.1, 0.2, 0.05).tolist(),
                "auto",
                "Expect tuning ends at 2th trial with alpha is  0.15 at basic strategy.",
            ),
        ]:
            logger.info("test_sq_tune_alpha_common with ")
            logger.info(f"eval_result_lst: {eval_result_lst}, alpha: {alpha}, quant_level: {quant_level}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst=eval_result_lst)
            self._test_sq_tune_alpha_common(partial_fake_eval, alpha=alpha, quant_level=quant_level)


class TestTextGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        from modeling_gptj import GPTJForCausalLM

        self.clm_model = GPTJForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gptj", torchscript=True)

    def test_text_generation(self):
        input_ids = torch.tensor([[531, 574, 658, 492, 156], [309, 296, 471, 817, 435], [182, 176, 756, 944, 768]])
        input_bs, input_len = input_ids.shape
        new_shape = [input_bs, 4, 1, 8]
        dummy_tensor = torch.ones(size=new_shape)
        pkv = tuple(dummy_tensor for _ in range(2))
        past_key_values = tuple(tuple(pkv) for _ in range(28))

        attention_mask = torch.ones(input_bs, input_len + 1)
        attention_mask[:, 0] = 0
        example_inputs = (
            input_ids,
            tuple(past_key_values),
            attention_mask,
        )

        def calib_func(prepared_model):
            for i in range(10):
                prepared_model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                )

        from neural_compressor import PostTrainingQuantConfig, quantization

        recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}}
        conf = PostTrainingQuantConfig(
            backend="ipex",
            excluded_precisions=["bf16"],
            recipes=recipes,
            example_inputs=example_inputs,
        )
        q_model = quantization.fit(
            self.clm_model,
            conf,
            calib_func=calib_func,
        )
        out = q_model(*example_inputs)
        values, indices = out[0][:, -1, :].log_softmax(dim=-1).topk(1)
        self.assertEqual(indices[0], torch.tensor([887]))
        self.assertEqual(indices[1], torch.tensor([362]))
        self.assertEqual(indices[2], torch.tensor([504]))


class TestMemoryUsage(unittest.TestCase):
    def test_sq_auto_mem_usage(self):
        import psutil

        data = psutil.virtual_memory()
        cpu_process = psutil.Process()
        p = psutil.Process(cpu_process.pid)
        mem_use0 = p.memory_info().rss / (1024**3)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
        )
        sq = TorchSmoothQuant(model, LLMCalibDataloader())
        sq.transform(alpha="auto", calib_iter=0, folding=False)
        mem_use1 = p.memory_info().rss / (1024**3)
        logger.info(f"The memory usage of this ut is {mem_use1 - mem_use0} GBs.")
        assert (mem_use1 - mem_use0) <= 2.0


class TestPeftModel(unittest.TestCase):
    def test_peft_model_fixed_alpha(self):
        import peft

        model_id = "peft-internal-testing/tiny_OPTForSequenceClassification-lora"
        model = peft.AutoPeftModelForSequenceClassification.from_pretrained(model_id)
        example_input = torch.ones(1, 12, dtype=torch.long)
        out1 = model(example_input)

        def calib_func(model):
            model(example_input)

        sq = TorchSmoothQuant(model, example_inputs=example_input, q_func=calib_func)
        sq.transform(alpha=0.5, folding=False)
        decoder = model.base_model.model.model.decoder
        if Version(peft.__version__) < Version("0.7.0"):
            self.assertTrue(isinstance(decoder.layers[0].self_attn.v_proj, SQLinearWrapper))
            self.assertTrue(
                isinstance(
                    decoder.layers[0].self_attn.v_proj.sq_linear.lora_A.default,
                    SQLinearWrapper,
                )
            )  # Linear in Linear
        else:
            self.assertTrue(
                isinstance(
                    decoder.layers[0].self_attn.v_proj.lora_A.default,
                    SQLinearWrapper,
                )
            )  # Linear in Linear
        self.assertTrue(
            isinstance(model.base_model.model.score.original_module, torch.nn.Linear)
        )  # Linear that is not called in calibration

    def test_peft_model_auto_alpha(self):
        import peft

        model_id = "peft-internal-testing/tiny_OPTForSequenceClassification-lora"
        model = peft.AutoPeftModelForSequenceClassification.from_pretrained(model_id, torchscript=True)
        example_input = torch.ones(1, 12, dtype=torch.long)
        out1 = model(example_input)

        def calib_func(model):
            model(example_input)

        # folding=False
        sq = TorchSmoothQuant(model, example_inputs=example_input, q_func=calib_func)
        sq.transform(alpha="auto", folding=False)
        decoder = model.base_model.model.model.decoder
        if Version(peft.__version__) < Version("0.7.0"):
            self.assertTrue(isinstance(decoder.layers[0].self_attn.v_proj, SQLinearWrapper))
            self.assertTrue(
                isinstance(
                    decoder.layers[0].self_attn.v_proj.sq_linear.lora_A.default,
                    SQLinearWrapper,
                )
            )  # Linear in Linear
        else:
            self.assertTrue(
                isinstance(
                    decoder.layers[0].self_attn.v_proj.lora_A.default,
                    SQLinearWrapper,
                )
            )  # Linear in Linear
        self.assertTrue(
            isinstance(model.base_model.model.score.original_module, torch.nn.Linear)
        )  # Linear that is not called in calibration

        # folding=True
        model = peft.AutoPeftModelForSequenceClassification.from_pretrained(model_id, torchscript=True)
        example_input = torch.ones(1, 12, dtype=torch.long)
        out1 = model(example_input)

        def calib_func(model):
            model(example_input)

        sq = TorchSmoothQuant(model, example_inputs=example_input, q_func=calib_func)
        sq.transform(alpha="auto", folding=True)
        if Version(peft.__version__) < Version("0.7.0"):
            self.assertTrue(
                isinstance(model.base_model.model.model.decoder.layers[0].self_attn.v_proj, torch.nn.Linear)
            )
        else:
            self.assertTrue(
                isinstance(
                    model.base_model.model.model.decoder.layers[0].self_attn.v_proj, peft.tuners.lora.layer.Linear
                )
            )
        self.assertTrue(
            isinstance(model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_A.default, torch.nn.Linear)
        )  # Linear in Linear

    def test_peft_model_quantization(self):
        import peft

        model_id = "peft-internal-testing/tiny_OPTForSequenceClassification-lora"
        model = peft.AutoPeftModelForSequenceClassification.from_pretrained(model_id)
        # model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_B.default.weight is Zero
        # peft model is needed to be trained first.
        example_input = torch.ones(1, 12, dtype=torch.long)
        out1 = model(example_input)

        def calib_func(model):
            model(example_input)

        from neural_compressor import PostTrainingQuantConfig, quantization

        recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}}
        conf = PostTrainingQuantConfig(
            excluded_precisions=["bf16"],
            recipes=recipes,
            example_inputs=example_input,
        )
        q_model = quantization.fit(
            model,
            conf,
            calib_func=calib_func,
        )
        decoder = q_model.model.base_model.model.model.decoder
        if Version(peft.__version__) < Version("0.7.0"):
            self.assertTrue(isinstance(decoder.layers[0].self_attn.v_proj, SQLinearWrapper))
            self.assertTrue(
                isinstance(
                    decoder.layers[0].self_attn.v_proj.sq_linear.lora_A.default,
                    SQLinearWrapper,
                )
            )  # Linear in Linear
        else:
            self.assertTrue(
                isinstance(
                    decoder.layers[0].self_attn.v_proj.lora_A.default,
                    SQLinearWrapper,
                )
            )  # Linear in Linear
        self.assertTrue(
            isinstance(q_model.model.base_model.model.score.original_module, torch.nn.Linear)
        )  # Linear that is not called in calibration

    @unittest.skipIf(
        IPEX_VERSION.release <= Version("2.1.0").release and ipex.__version__ != "2.1.0+cpu",
        "Please use Intel extension for Pytorch version higher or equal to 2.1.0",
    )
    def test_peft_model_quantization_ipex(self):
        import peft

        model_id = "peft-internal-testing/tiny_OPTForSequenceClassification-lora"
        model = peft.AutoPeftModelForSequenceClassification.from_pretrained(model_id, torchscript=True)
        # model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_B.default.weight is Zero
        # peft model is needed to be trained first.
        example_input = torch.ones(1, 12, dtype=torch.long)
        out1 = model(example_input)[0]

        def calib_func(model):
            model(example_input)

        from neural_compressor import PostTrainingQuantConfig, quantization

        recipes = {"smooth_quant": True, "smooth_quant_args": {"alpha": 0.5}}
        conf = PostTrainingQuantConfig(
            backend="ipex",  # IPEX will got error now, will enhance it.
            excluded_precisions=["bf16"],
            op_name_dict={".*": {"activation": {"algorithm": "minmax"}}},
            recipes=recipes,
            example_inputs=example_input,
        )
        q_model = quantization.fit(
            model,
            conf,
            calib_func=calib_func,
        )
        out2 = q_model.model(example_input)[0]


class TestInputConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        class RandDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                yield torch.rand((1, 3))

        self.linear_dl = RandDataloader()

    @classmethod
    def test_sq_weight_clipping(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
        sq.transform(alpha="auto", calib_iter=1, folding=True, weight_clip=False)
        assert sq.weight_clip is False

    @classmethod
    def test_sq_auto_alpha_arg(self):
        class Model(torch.nn.Module):
            device = torch.device("cpu")

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
        sq.transform(
            alpha="auto",
            calib_iter=1,
            folding=False,
            auto_alpha_args={
                "init_alpha": 0.7,
                "alpha_min": 0.5,
                "alpha_max": 0.9,
                "alpha_step": 0.1,
                "shared_criterion": "mean",
            },
        )
        assert sq.init_alpha == 0.7
        assert sq.auto_alpha_args["alpha_min"] == 0.5


class TestAlphaAutoLinearBlockwise(unittest.TestCase):
    @classmethod
    def test_sq_linear_Blockwise_auto(self):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
        )
        sq = TorchSmoothQuant(model, LLMCalibDataloader())
        sq.transform(
            alpha="auto",
            calib_iter=1,
            folding=False,
            auto_alpha_args={
                "alpha_min": 0.45,
                "alpha_max": 0.55,
                "alpha_step": 0.01,
                "shared_criterion": "mean",
                "enable_blockwise_loss": True,
            },
        )
        for i in range(12):
            op_name1 = "model.decoder.layers." + str(i) + ".self_attn.out_proj"
            op_name2 = "model.decoder.layers." + str(i) + ".fc1"
            assert sq.alpha_per_layer[op_name1] == sq.alpha_per_layer[op_name2]
        assert len(sq.block_names) == 13


if __name__ == "__main__":
    unittest.main()
