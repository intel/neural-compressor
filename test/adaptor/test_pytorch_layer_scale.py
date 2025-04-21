import copy
import logging
import os
import sys
import unittest

import torch
import torch.nn as nn

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import neural_compressor.adaptor.pytorch as pytorch_util
from neural_compressor import quantization
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.adaptor.pytorch import PyTorchAdaptor
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.model.torch_model import PyTorchModel
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(42)


class ConvEncoderWithLayerScale(nn.Module):
    """Test model with layer_scale parameter that caused the original issue."""

    def __init__(self, dim=64, hidden_dim=128, kernel_size=3, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.layer_scale * x
        x = input + self.drop_path(x)
        return x


class ConvEncoderWithLayerGamma(nn.Module):
    """Test model with renamed layer_gamma parameter (the fix)."""

    def __init__(self, dim=64, hidden_dim=128, kernel_size=3, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_gamma = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.layer_gamma * x
        x = input + self.drop_path(x)
        return x


class CalibDataloader:
    """Simple calibration dataloader for testing."""
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.batch_size = 1  # Since we're yielding single samples

    def __iter__(self):
        yield self.data, self.label


class TestPyTorchLayerScale(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.constant_data = torch.randn(1, 64, 32, 32)
        self.constant_label = torch.randint(0, 10, (1,))

    def test_layer_scale_error(self):
        """Test that the original layer_scale parameter causes an error."""
        model = ConvEncoderWithLayerScale()
        model.eval()

        calib_dataloader = CalibDataloader(self.constant_data, self.constant_label)

        # Configure quantization
        conf = PostTrainingQuantConfig()

        # Try to quantize and verify it fails
        q_model = quantization.fit(model, conf, calib_dataloader=calib_dataloader)
        # The quantization should fail and return None
        self.assertIsNone(q_model, "Quantization should fail with layer_scale parameter")

    def test_layer_gamma_success(self):
        """Test that the renamed layer_gamma parameter works correctly."""
        model = ConvEncoderWithLayerGamma()
        model.eval()

        calib_dataloader = CalibDataloader(self.constant_data, self.constant_label)

        # Configure quantization
        conf = PostTrainingQuantConfig()

        # This should succeed with layer_gamma parameter
        try:
            q_model = quantization.fit(model, conf, calib_dataloader=calib_dataloader)
            self.assertIsNotNone(q_model)
        except ValueError as e:
            self.fail(f"Quantization failed with layer_gamma: {str(e)}")


if __name__ == "__main__":
    unittest.main()
