import pytest
import torch

from neural_compressor.torch.algorithms.weight_only.hqq.bitpack import Packer
from neural_compressor.torch.algorithms.weight_only.hqq.config import (
    HQQModuleConfig,
    QTensorConfig,
    default_hqq_module_config,
    default_scale_quant_config,
    default_weight_quant_config,
    default_zero_quant_config,
)
from neural_compressor.torch.algorithms.weight_only.hqq.qtensor import QTensor, QTensorMetaInfo


def test_default_hqq_module_config():
    config = default_hqq_module_config
    print(config)
    assert isinstance(config, HQQModuleConfig)
    assert config.weight == default_weight_quant_config
    assert config.zero == default_zero_quant_config
    assert config.scale == default_scale_quant_config


def test_default_weight_quant_config():
    config = default_weight_quant_config
    assert isinstance(config, QTensorConfig)
    assert config.nbits == 4
    assert config.channel_wise is True


def test_default_zero_quant_config():
    config = default_zero_quant_config
    assert isinstance(config, QTensorConfig)
    assert config.nbits == 8
    assert config.channel_wise is False


def test_default_scale_quant_config():
    config = default_scale_quant_config
    assert isinstance(config, QTensorConfig)
    assert config.nbits == 8
    assert config.channel_wise is True


def test_qtensor_meta_info():
    meta_info = QTensorMetaInfo
    print(meta_info)


@pytest.mark.parametrize("nbits", [2, 3, 4, 8])
def test_packer(nbits):
    # TODO:　add test for 3 bits
    range_max = 2**nbits
    dims = 16 if nbits != 3 else 10
    W = torch.randint(0, range_max, (dims, dims)).to(torch.uint8)
    W_pack = Packer.get_pack_fn(nbits)(W)
    W_pack_unpack = Packer.get_unpack_fn(nbits)(W_pack)
    assert torch.allclose(W, W_pack_unpack)
    print("Packer test passed!")


class TestQTensor:
    def test_q_tensor(self):
        in_feats = 3
        out_feats = 4

        val = torch.randn(out_feats, in_feats)
        scale = torch.randn(out_feats)
        zero = torch.randint(1, 10, (out_feats,))
        q_tensor_meta = QTensorMetaInfo(nbits=4, group_size=64, shape=(out_feats, in_feats), axis=0, packing=False)
        q_tensor = QTensor(val, scale, zero, q_tensor_meta)
        print(q_tensor)
        q_tensor_half = q_tensor.half()
        print(q_tensor_half)

    def test_q_tensor2(self):
        in_feats = 64
        out_feats = 64

        val = torch.randn(out_feats, in_feats)
        scale = torch.randn(out_feats)
        zero = torch.randint(1, 10, (out_feats,))
        q_tensor_meta = QTensorMetaInfo(nbits=4, group_size=64, shape=(out_feats, in_feats), axis=0, packing=False)
        q_tensor = QTensor(val, scale, zero, q_tensor_meta)
        q_scale_meta = QTensorMetaInfo(nbits=8, group_size=64, shape=(out_feats,), axis=0, packing=False)
        q_scale_scale = torch.randn(out_feats)
        q_scale_zero = torch.randint(1, 10, (1,))
        q_scale = QTensor(scale, q_scale_scale, q_scale_zero, q_tensor_meta)
        q_tensor.scale = q_scale
        print(q_tensor)
        print(q_tensor.half())

    def test_qtensor_meta_info(self):
        in_feats = 64
        out_feats = 64
        meta_config = QTensorMetaInfo(nbits=4, group_size=64, shape=(out_feats, in_feats), axis=0, packing=False)
        print(meta_config)
        print(meta_config.to_dict)
        assert meta_config.to_dict() == {
            "nbits": 4,
            "group_size": 64,
            "shape": (out_feats, in_feats),
            "axis": 0,
            "packing": False,
        }
