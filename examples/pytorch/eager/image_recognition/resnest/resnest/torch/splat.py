"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from torch.quantization import fuse_modules

__all__ = ['SplAtConv2d']

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu0 = ReLU(inplace=True)
        self.skip_add = nn.quantized.FloatFunctional()
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.relu1 = ReLU(inplace=True)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)
        self.skip_mul = nn.quantized.FloatFunctional()
        self.quant1 = torch.quantization.QuantStub()
        self.quant2 = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu0(x)
        if self.radix > 2:
            x = self.dequant(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            if self.radix == 2:
                gap = self.skip_add.add(splited[0], splited[1])
            else:
                gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu1(gap)

        atten = self.fc2(gap)
        if self.radix <= 2:
            atten = self.dequant(atten)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
        if self.radix <= 2:
            atten = self.quant1(atten)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            if self.radix == 2:
                tmp = [self.skip_mul.mul(att, split) for (att, split) in zip(attens, splited)]
                out = self.skip_add.add(tmp[0], tmp[1])
            else:
                out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        if self.radix > 2:
            out = self.quant2(out)
        return out.contiguous()

    def fuse_model(self):
        if self.dropblock_prob <= 0.0:
            if self.use_bn:
                fuse_modules(self, ['conv', 'bn0', 'relu0'], inplace=True)
                fuse_modules(self, ['fc1', 'bn1', 'relu1'], inplace=True)
            else:
                fuse_modules(self, ['conv', 'relu0'], inplace=True)
                fuse_modules(self, ['fc1', 'relu1'], inplace=True)
        else:
            if self.use_bn:
                fuse_modules(self, ['conv', 'bn0'], inplace=True)
                fuse_modules(self, ['fc1', 'bn1', 'relu1'], inplace=True)
            else:
                fuse_modules(self, ['fc1', 'relu1'], inplace=True)

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

