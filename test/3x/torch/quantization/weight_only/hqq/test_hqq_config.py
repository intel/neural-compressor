from neural_compressor.torch.algorithms.weight_only.hqq.config import (
    HQQModuleConfig,
    QTensorConfig,
    default_hqq_module_config,
    default_scale_quant_config,
    default_weight_quant_config,
    default_zero_quant_config,
)
from neural_compressor.torch.algorithms.weight_only.hqq.qtensor import QTensorMetaInfo


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
