import unittest
import torch
from neural_compressor.data import Datasets
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.adaptor.torch_utils.smooth_quant import TorchSmoothQuant
import torchvision


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
        sq.transform(alpha='auto', calib_iter=1)
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
        sq.transform(alpha=0.5, calib_iter=1)
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
        sq.transform(alpha='auto', calib_iter=3)
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
        sq.transform(alpha=0.5)
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
        sq.transform(alpha=0.5, calib_iter=2)
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
        sq.transform(alpha=0.5, calib_iter=2)
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
        sq.transform(alpha=0.5, calib_iter=2)
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
        sq.transform(alpha=0.6, calib_iter=2)
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
        sq.transform(alpha=0.6, calib_iter=2)
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
        sq.transform(alpha=0.5, calib_iter=1)
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
        sq.transform(alpha=0.5, calib_iter=1)
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
        sq.transform(alpha='auto', calib_iter=1)
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
        sq.transform(alpha=0.5, calib_iter=1)
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
        sq.transform(alpha=0.5, calib_iter=1)
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
        sq.transform(alpha=0.5, calib_iter=1)
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
        sq.transform(alpha=0.5, calib_iter=1)
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
        sq.transform(alpha=0.5, calib_iter=1)
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
        sq.transform(alpha=0.5, calib_iter=1)
        assert len(sq.absorb_to_layer) == 0


if __name__ == '__main__':
    unittest.main()
