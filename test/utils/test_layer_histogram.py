"""Tests for collecting layer histogram."""
from neural_compressor.utils.collect_layer_histogram import LayerHistogramCollector
from collections import OrderedDict
from neural_compressor.utils import logger
import numpy as np
import torch
import torch.nn as nn
import unittest

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class BuildFakeModel(nn.Module):
    def conv_1x1_bn(self, inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    def conv_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    def make_divisible(self, x, divisor=8):
        return int(np.ceil(x * 1. / divisor) * divisor)

    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        self.last_channel = self.make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [self.conv_bn(3, input_channel, 2)]
        for t, c, n, s in interverted_residual:
            output_channel = self.make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        self.features.append(self.conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(self.last_channel, n_class)

    def forward(self, x):
        x = self.features(x)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
        x = self.classifier(x)
        return x

class CollectLayerHistogram(unittest.TestCase):
    def setUp(self):
        model = BuildFakeModel(width_mult=1)
        layer_tensor, include_layer = OrderedDict(), OrderedDict()
        i = 0
        for key, value in model.state_dict().items():
            if not value.ndim:
                value = np.expand_dims(value, axis=0)
            if i>200:
                pass
            else:
                include_layer[key] = np.array(value, dtype=np.float32)
            layer_tensor[key] = np.array(value, dtype=np.float32)
            i += 1
        self.layer_histogram_collector = LayerHistogramCollector \
            (num_bins=8001, layer_tensor=layer_tensor, include_layer=include_layer, logger=logger)
        
    def test_layer_histogram(self):
        self.layer_histogram_collector.collect()
        self.assertEqual(self.layer_histogram_collector.layer_tensor.keys() \
            & self.layer_histogram_collector.include_layer.keys(), \
                self.layer_histogram_collector.hist_dict.keys())

if __name__ == '__main__':
    unittest.main()
