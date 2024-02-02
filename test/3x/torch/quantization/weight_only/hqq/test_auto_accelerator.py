import os

import pytest
import torch

from neural_compressor.torch.algorithms.weight_only.hqq.auto_accelerator import auto_detect_accelerator


class Test_CPU_Accelerator:
    @pytest.fixture
    def force_use_cpu(self, monkeypatch):
        # Force use CPU
        monkeypatch.setenv("FORCE_DEVICE", "cpu")

    def test_cpu_accelerator(self, force_use_cpu):
        print(f"FORCE_DEVICE: {os.environ.get('FORCE_DEVICE', None)}")
        accelerator = auto_detect_accelerator()
        assert accelerator.current_device() == "cpu", f"{accelerator.current_device()}"
        assert accelerator.current_device_name() == "cpu"
        assert accelerator.device() is None
        assert accelerator.empty_cache() is None
        assert accelerator.synchronize() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class Test_CUDA_Accelerator:

    @pytest.fixture
    def force_use_cuda(self, monkeypatch):
        # Force use CUDA
        monkeypatch.setenv("FORCE_DEVICE", "cuda")

    def test_cuda_accelerator(self, force_use_cuda):
        print(f"FORCE_DEVICE: {os.environ.get('FORCE_DEVICE', None)}")
        accelerator = auto_detect_accelerator()
        assert accelerator.current_device() == 0, f"{accelerator.current_device()}"
        assert accelerator.current_device_name() == "cuda:0"
        assert accelerator.device() is not None
        assert accelerator.empty_cache() is None
        assert accelerator.synchronize() is None
        assert accelerator.set_device(0) is None
        assert accelerator.device_name(0) == "cuda:0"
        assert accelerator.is_available() is True
        assert accelerator.name() == "cuda"
        assert accelerator.device_name(1) == "cuda:1"
        assert accelerator.set_device(1) is None
        assert accelerator.device_name(1) == "cuda:1"
        assert accelerator.current_device() == 1
        assert accelerator.current_device_name() == "cuda:1"
        assert accelerator.synchronize() is None
        assert accelerator.empty_cache() is None
