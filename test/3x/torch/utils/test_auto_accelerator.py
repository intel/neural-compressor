import os

import pytest
import torch

from neural_compressor.torch.utils import get_accelerator
from neural_compressor.torch.utils.auto_accelerator import (
    CPU_Accelerator,
    CUDA_Accelerator,
    HPU_Accelerator,
    XPU_Accelerator,
    accelerator_registry,
    auto_detect_accelerator,
)


@pytest.mark.skipif(not HPU_Accelerator.is_available(), reason="HPEX is not available")
class TestHPUAccelerator:
    def test_cuda_accelerator(self):
        assert (
            os.environ.get("INC_TARGET_DEVICE", None) is None
        ), "INC_TARGET_DEVICE shouldn't be set. HPU is the first priority."
        accelerator = auto_detect_accelerator()
        assert accelerator.current_device() == 0, f"{accelerator.current_device()}"
        assert accelerator.current_device_name() == "hpu:0"
        assert accelerator.device() is not None
        assert accelerator.device_name(0) == "hpu:0"
        assert accelerator.is_available() is True
        assert accelerator.name() == "hpu"
        assert accelerator.device_name(1) == "hpu:1"
        assert accelerator.synchronize() is None
        assert accelerator.empty_cache() is None

    def test_get_device(self):
        if torch.hpu.device_count() < 2:
            return
        accelerator = auto_detect_accelerator()
        assert accelerator.set_device(1) is None
        assert accelerator.current_device_name() == "hpu:1"
        cur_device = get_accelerator().current_device_name()
        assert cur_device == "hpu:1"
        tmp_tensor = torch.tensor([1, 2], device=cur_device)
        assert "hpu:1" == str(tmp_tensor.device)


@pytest.mark.skipif(not XPU_Accelerator.is_available(), reason="XPU is not available")
class TestXPUAccelerator:

    @pytest.fixture
    def force_use_xpu(self, monkeypatch):
        # Force use xpu
        monkeypatch.setenv("INC_TARGET_DEVICE", "xpu")

    def test_xpu_accelerator(self, force_use_xpu):
        print(f"INC_TARGET_DEVICE: {os.environ.get('INC_TARGET_DEVICE', None)}")
        accelerator = auto_detect_accelerator()
        assert accelerator.current_device() == 0, f"{accelerator.current_device()}"
        assert accelerator.current_device_name() == "xpu:0"
        assert accelerator.device() is not None
        assert accelerator.set_device(0) is None
        assert accelerator.device_name(0) == "xpu:0"
        assert accelerator.is_available() is True
        assert accelerator.name() == "xpu"
        assert accelerator.device_name(1) == "xpu:1"
        assert accelerator.synchronize() is None
        assert accelerator.empty_cache() is None

    def test_get_device(self):
        if torch.xpu.device_count() < 2:
            return
        accelerator = auto_detect_accelerator()
        assert accelerator.set_device(1) is None
        assert accelerator.current_device_name() == "xpu:1"
        cur_device = get_accelerator().current_device_name()
        assert cur_device == "xpu:1"
        tmp_tensor = torch.tensor([1, 2], device=cur_device)
        assert "xpu:1" == str(tmp_tensor.device)


class TestCPUAccelerator:
    @pytest.fixture
    def force_use_cpu(self, monkeypatch):
        # Force use CPU
        monkeypatch.setenv("INC_TARGET_DEVICE", "cpu")

    def test_cpu_accelerator(self, force_use_cpu):
        print(f"INC_TARGET_DEVICE: {os.environ.get('INC_TARGET_DEVICE', None)}")
        accelerator = auto_detect_accelerator()
        assert accelerator.current_device() == "cpu", f"{accelerator.current_device()}"
        assert accelerator.current_device_name() == "cpu"
        assert accelerator.is_available()
        assert accelerator.set_device(1) is None
        assert accelerator.device() is None
        assert accelerator.empty_cache() is None
        assert accelerator.synchronize() is None


@pytest.mark.skipif(not CUDA_Accelerator.is_available(), reason="CUDA is not available")
class TestCUDAAccelerator:

    @pytest.fixture
    def force_use_cuda(self, monkeypatch):
        # Force use CUDA
        monkeypatch.setenv("INC_TARGET_DEVICE", "cuda")

    def test_cuda_accelerator(self, force_use_cuda):
        print(f"INC_TARGET_DEVICE: {os.environ.get('INC_TARGET_DEVICE', None)}")
        accelerator = auto_detect_accelerator()
        assert accelerator.current_device() == 0, f"{accelerator.current_device()}"
        assert accelerator.current_device_name() == "cuda:0"
        assert accelerator.device() is not None
        assert accelerator.set_device(0) is None
        assert accelerator.device_name(0) == "cuda:0"
        assert accelerator.is_available() is True
        assert accelerator.name() == "cuda"
        assert accelerator.device_name(1) == "cuda:1"
        assert accelerator.synchronize() is None
        assert accelerator.empty_cache() is None

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Only one GPU is available")
    def test_get_device(self):
        accelerator = auto_detect_accelerator()
        assert accelerator.set_device(1) is None
        assert accelerator.current_device_name() == "cuda:1"
        cur_device = get_accelerator().current_device_name()
        assert cur_device == "cuda:1"
        tmp_tensor = torch.tensor([1, 2], device=cur_device)
        assert "cuda:1" == str(tmp_tensor.device)


class TestAutoAccelerator:

    @pytest.fixture
    def set_cuda_available(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def test_auto_accelerator(self, set_cuda_available):
        accelerator = auto_detect_accelerator()
        all_accelerators = accelerator_registry.get_sorted_accelerators()
        assert accelerator.name() == all_accelerators[0]().name()
