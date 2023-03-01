import unittest
import torch
from neural_compressor.adaptor.torch_utils.smooth_quant import TorchSmoothQuant


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
        sq.transform(alpha=0.5,calib_iter=2)
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
                out = out+x
                out = self.conv2(out)
                return out

        model = Model()

        sq = TorchSmoothQuant(model, self.conv_dl)
        sq.transform(alpha=0.6,calib_iter=2)
        assert len(sq.absorb_to_layer) == 0



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
    def test_sq_linear_norm(self):
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
    def test_sq_linear_norm(self):
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
