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

from typing import Any, List, Optional, Tuple, Union

from neural_compressor.common.base_config import (
    BaseConfig,
    config_registry,
    get_all_config_set_from_config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from neural_compressor.common.base_tuning import ConfigLoader, ConfigSet, Evaluator, SequentialSampler
from neural_compressor.common.tuning_param import TuningParam
from neural_compressor.common.utils import DEFAULT_WHITE_LIST, OP_NAME_OR_MODULE_TYPE

PRIORITY_FAKE_ALGO = 100
FAKE_CONFIG_NAME = "fake"
PRIORITY_FAKE_ALGO_1 = 90
FAKE_CONFIG_NAME_1 = "fake_one"
DEFAULT_WEIGHT_BITS = [4, 6]

FAKE_FRAMEWORK_NAME = "FAKE_FWK"

FAKE_MODEL_INFO = [("OP1_NAME", "OP_TYPE1"), ("OP2_NAME", "OP_TYPE1"), ("OP3_NAME", "OP_TYPE2")]


class FakeModel:
    def __init__(self) -> None:
        self.name = "fake_model"

    def __call__(self, x) -> Any:
        return x

    def __repr__(self) -> str:
        return "FakeModel"


@register_config(framework_name=FAKE_FRAMEWORK_NAME, algo_name=FAKE_CONFIG_NAME, priority=PRIORITY_FAKE_ALGO)
class FakeAlgoConfig(BaseConfig):
    """Config class for fake algo."""

    supported_configs: List = []
    params_list = [
        "weight_dtype",
        "weight_bits",
        TuningParam("target_op_type_list", tunable_type=List[List[str]]),
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
    def get_model_info(model: Any) -> List[Tuple[str, Any]]:
        return FAKE_MODEL_INFO

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FakeAlgoConfig", List["FakeAlgoConfig"]]:
        return FakeAlgoConfig(weight_bits=DEFAULT_WEIGHT_BITS)


def get_default_fake_config() -> FakeAlgoConfig:
    """Generate the default fake config.

    Returns:
        the default fake config.
    """
    return FakeAlgoConfig()


@register_config(framework_name=FAKE_FRAMEWORK_NAME, algo_name=FAKE_CONFIG_NAME_1, priority=PRIORITY_FAKE_ALGO_1)
class FakeAlgoOneConfig(BaseConfig):
    """Config class for fake algo."""

    supported_configs: List = []
    params_list = [
        "weight_dtype",
        "weight_bits",
        TuningParam("target_op_type_list", tunable_type=List[List[str]]),
    ]
    name = FAKE_CONFIG_NAME_1

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
        return super(FakeAlgoOneConfig, cls).from_dict(config_dict=config_dict)

    @classmethod
    def register_supported_configs(cls) -> List:
        pass

    @staticmethod
    def get_model_info(model: Any) -> List[Tuple[str, Any]]:
        return FAKE_MODEL_INFO

    @classmethod
    def get_config_set_for_tuning(cls) -> Union[None, "FakeAlgoOneConfig", List["FakeAlgoOneConfig"]]:
        return FakeAlgoOneConfig(weight_bits=DEFAULT_WEIGHT_BITS)


def get_all_config_set() -> Union[BaseConfig, List[BaseConfig]]:
    return get_all_config_set_from_config_registry(fwk_name=FAKE_FRAMEWORK_NAME)


register_supported_configs_for_fwk(fwk_name=FAKE_FRAMEWORK_NAME)


class TestEvaluator(unittest.TestCase):
    def test_single_eval_fn(self):
        def fake_eval_fn(model):
            return 1.0

        evaluator = Evaluator()
        evaluator.set_eval_fn_registry(fake_eval_fn)
        evaluator.self_check()
        self.assertEqual(evaluator.get_number_of_eval_functions(), 1)

    def test_single_eval_fn_dict(self):
        acc_data = iter([1.0, 0.8, 0.99, 1.0, 0.99, 0.99])

        def eval_acc_fn(model) -> float:
            return next(acc_data)

        eval_fns = {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"}

        evaluator = Evaluator()
        evaluator.set_eval_fn_registry(eval_fns)
        evaluator.self_check()
        self.assertEqual(evaluator.get_number_of_eval_functions(), 1)


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
        self.assertEqual(len(config_set), len(config_registry.get_all_config_cls_by_fwk_name(FAKE_FRAMEWORK_NAME)))
        self.assertEqual(config_set[0].weight_bits, DEFAULT_WEIGHT_BITS)

    def test_config_expand_complex_tunable_type(self):
        target_op_type_list_options = [["Conv", "Gemm"], ["Conv", "Matmul"]]
        configs = FakeAlgoConfig(target_op_type_list=target_op_type_list_options)
        configs_list = configs.expand()
        self.assertEqual(len(configs_list), len(target_op_type_list_options))
        for i in range(len(configs_list)):
            self.assertEqual(configs_list[i].target_op_type_list, target_op_type_list_options[i])

    def test_mixed_two_algos(self):
        model = FakeModel()
        OP1_NAME = "OP1_NAME"
        OP2_NAME = "OP2_NAME"
        fake_config = FakeAlgoConfig(weight_bits=4, white_list=[OP1_NAME])
        fake1_config = FakeAlgoOneConfig(weight_bits=2, white_list=[OP2_NAME])
        mixed_config = fake_config + fake1_config
        model_info = mixed_config.get_model_info(model)
        config_mapping = mixed_config.to_config_mapping(model_info=model_info)
        self.assertIn(OP1_NAME, [op_info[0] for op_info in config_mapping])
        self.assertIn(OP2_NAME, [op_info[0] for op_info in config_mapping])


class TestConfigSet(unittest.TestCase):
    def setUp(self):
        self.config_set = [get_default_fake_config(), get_default_fake_config()]
        self.config_set_obj = ConfigSet.from_fwk_configs(self.config_set)

    def test_config_set(self) -> None:
        self.assertEqual(len(self.config_set_obj), len(self.config_set))
        self.assertEqual(self.config_set_obj[0].weight_bits, self.config_set[0].weight_bits)


class TestConfigSampler(unittest.TestCase):
    def setUp(self):
        self.config_set = [get_default_fake_config(), get_default_fake_config()]
        self.seq_sampler = SequentialSampler(self.config_set)

    def test_config_sampler(self) -> None:
        self.assertEqual(list(self.seq_sampler), list(range(len(self.config_set))))


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.config_set = [FakeAlgoConfig(weight_bits=4), FakeAlgoConfig(weight_bits=8)]
        self.loader = ConfigLoader(self.config_set)

    def test_config_loader(self) -> None:
        self.assertEqual(len(list(self.loader)), len(self.config_set))
        for i, config in enumerate(self.loader):
            self.assertEqual(config, self.config_set[i])


if __name__ == "__main__":
    unittest.main()
