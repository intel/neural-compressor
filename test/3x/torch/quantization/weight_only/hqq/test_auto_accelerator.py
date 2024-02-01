import os

import pytest
import torch

from neural_compressor.torch.algorithms.weight_only.hqq.auto_accelerator import auto_detect_accelerator


class Test_CPU_Accelerator:
    ORIG_FORCE_DEVICE = os.environ.get("FORCE_DEVICE", None)

    @classmethod
    def setup_class(cls):
        os.environ["FORCE_DEVICE"] = "cpu"

    @classmethod
    def teardown_class(cls):
        if cls.ORIG_FORCE_DEVICE:
            os.environ["FORCE_DEVICE"] = cls.ORIG_FORCE_DEVICE

    def test_cpu_accelerator(self):
        print(f"FORCE_DEVICE: {os.environ.get('FORCE_DEVICE', None)}")
        accelerator = auto_detect_accelerator()
        assert accelerator.current_device() == "cpu", f"{accelerator.current_device()}"
        assert accelerator.current_device_name() == "cpu"
        assert accelerator.device() is None
        assert accelerator.empty_cache() is None
        assert accelerator.synchronize() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
class Test_CUDA_Accelerator:
    ORIG_FORCE_DEVICE = os.environ.get("FORCE_DEVICE", None)

    @classmethod
    def setup_class(cls):
        os.environ["FORCE_DEVICE"] = "cuda"

    @classmethod
    def teardown_class(cls):
        if cls.ORIG_FORCE_DEVICE:
            os.environ["FORCE_DEVICE"] = cls.ORIG_FORCE_DEVICE

    def test_cuda_accelerator(self):
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
