"""Tests for common components.

!!! Please do not import any framework-specific modules in this file. !!!
* Note, we may need to add some auto check mechanisms to ensure this.

These tests aim to assess the fundamental functionalities of common components and enhance code coverage.
All tests will be included for each framework CI.

* Note
The folder structure:
.
├── 3x
│   ├── common
│   ├── onnxrt
│   ├── tensorflow
│   └── torch

For each fwk CI:

onnxrt_included_folder:
    ├── 3x
    │   ├── common
    │   ├── onnxrt

tensorflow_included_folder:
    ├── 3x
    │   ├── common
    │   ├── tensorflow


torch_included_folder:
    ├── 3x
    │   ├── common
    │   ├── torch
"""

import unittest

from neural_compressor.common import Logger

logger = Logger().get_logger()

from typing import Any, Callable, List, Optional, Tuple, Union

from neural_compressor.common.base_config import (
    BaseConfig,
    ComposableConfig,
    get_all_config_set_from_config_registry,
    register_config,
)
from neural_compressor.common.base_tuning import BaseConfigSet, ConfigLoader, Sampler
from neural_compressor.common.tuning_param import TuningParam
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, OP_NAME_OR_MODULE_TYPE

PRIORITY_FAKE_ALGO = 100
FAKE_CONFIG_NAME = "fake"
DEFAULT_WEIGHT_BITS = [4, 6]

FAKE_FRAMEWORK_NAME = "FAKE_FWK"


@register_config(framework_name=FAKE_FRAMEWORK_NAME, algo_name=FAKE_CONFIG_NAME, priority=PRIORITY_FAKE_ALGO)
class FakeAlgoConfig(BaseConfig):
    """Config class for fake algo."""

    supported_configs: List = []
    params_list = [
        "weight_dtype",
        "weight_bits",
        TuningParam("target_op_type_list", tunable_type=List[str]),
    ]
    name = FAKE_CONFIG_NAME

    def __init__(
        self,
        weight_dtype: str = "int",
        weight_bits: int = 4,
        target_op_type_list: List[str] = ["Conv", "Gemm"],
        white_list: Optional[List[OP_NAME_OR_MODULE_TYPE]] = DEFAULT_WHITE_LIST,
    ):
        """Init fake config.

        Args:
            weight_dtype (str): Data type for weights, default is "int".
            weight_bits (int): Number of bits used to represent weights, default is 4.
        """
        super().__init__(white_list=white_list)
        self.weight_bits = weight_bits
        self.weight_dtype = weight_dtype
        self.target_op_type_list = target_op_type_list
        self._post_init()

    def to_dict(self):
        return super().to_dict()

    @classmethod
    def from_dict(cls, config_dict):
        return super(FakeAlgoConfig, cls).from_dict(config_dict=config_dict)

    @classmethod
    def register_supported_configs(cls) -> List:
        pass

    @staticmethod
    def get_model_info(model: Any) -> List[Tuple[str, Callable]]:
        pass

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FakeAlgoConfig", List["FakeAlgoConfig"]]:
        return FakeAlgoConfig(weight_bits=DEFAULT_WEIGHT_BITS)


FakeAlgoConfig.register_supported_configs()


def get_default_fake_config() -> FakeAlgoConfig:
    """Generate the default fake config.

    Returns:
        the default fake config.
    """
    return FakeAlgoConfig()


def get_all_config_set() -> Union[BaseConfig, List[BaseConfig]]:
    return get_all_config_set_from_config_registry(fwk_name=FAKE_FRAMEWORK_NAME)


class TestBaseConfig(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestBaseConfig test: {self.id()}")

    def test_api(self):
        fake_default_config = get_default_fake_config()
        self.assertEqual(fake_default_config.weight_dtype, "int")
        config_set = get_all_config_set()
        self.assertEqual(len(config_set), 1)
        self.assertEqual(config_set[0].weight_bits, DEFAULT_WEIGHT_BITS)


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.config_set = [get_default_fake_config(), get_default_fake_config()]
        self.loader = ConfigLoader(self.config_set)

    def test_config_loader(self) -> None:
        quant_config = get_default_fake_config()
        result = ConfigLoader(quant_config)
        self.assertEqual(len(list(result)), len(quant_config.expand()))


if __name__ == "__main__":
    unittest.main()
