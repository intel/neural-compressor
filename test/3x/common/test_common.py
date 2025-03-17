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

import copy
import unittest

from neural_compressor.common import Logger

logger = Logger().get_logger()

from typing import Any, Callable, List, Optional, Tuple, Union

from neural_compressor.common.base_config import (
    BaseConfig,
    config_registry,
    get_all_config_set_from_config_registry,
    register_config,
    register_supported_configs_for_fwk,
)
from neural_compressor.common.base_tuning import (
    ConfigLoader,
    ConfigSet,
    EvaluationFuncWrapper,
    Evaluator,
    SequentialSampler,
    TuningConfig,
    init_tuning,
)
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


class FakeOpType:
    def __init__(self) -> None:
        self.name = "fake_module"

    def __call__(self, x) -> Any:
        return x

    def __repr__(self) -> str:
        return "FakeModule"


class OP_TYPE1(FakeOpType):
    pass


class OP_TYPE2(FakeOpType):
    pass


def build_simple_fake_model():
    return FakeModel()


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

    def test_config_expand_with_empty_options(self):
        configs = FakeAlgoConfig(weight_dtype=["int", "float32"], weight_bits=[])
        configs_list = configs.expand()
        self.assertEqual(len(configs_list), 2)

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

    def test_set_local_op_name(self):
        quant_config = FakeAlgoConfig(weight_bits=4)
        # set `OP1_NAME`
        fc1_config = FakeAlgoConfig(weight_bits=6)
        quant_config.set_local("OP1_NAME", fc1_config)
        model_info = FAKE_MODEL_INFO
        logger.info(quant_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("OP1_NAME", "OP_TYPE1")].weight_bits == 6)
        self.assertTrue(configs_mapping[("OP2_NAME", "OP_TYPE1")].weight_bits == 4)
        self.assertTrue(configs_mapping[("OP3_NAME", "OP_TYPE2")].weight_bits == 4)

    def test_set_local_op_type(self):
        quant_config = FakeAlgoConfig(weight_bits=4)
        # set all `OP_TYPE1`
        fc1_config = FakeAlgoConfig(weight_bits=6)
        quant_config.set_local(OP_TYPE1, fc1_config)
        model_info = FAKE_MODEL_INFO
        logger.info(quant_config)
        configs_mapping = quant_config.to_config_mapping(model_info=model_info)
        logger.info(configs_mapping)
        self.assertTrue(configs_mapping[("OP1_NAME", "OP_TYPE1")].weight_bits == 6)
        self.assertTrue(configs_mapping[("OP2_NAME", "OP_TYPE1")].weight_bits == 6)
        self.assertTrue(configs_mapping[("OP3_NAME", "OP_TYPE2")].weight_bits == 4)


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

    def test_config_loader_skip_verified_config(self) -> None:
        config_set = [FakeAlgoConfig(weight_bits=[4, 8]), FakeAlgoConfig(weight_bits=8)]
        config_loader = ConfigLoader(config_set)
        config_count = 0
        for i, config in enumerate(config_loader):
            config_count += 1
        self.assertEqual(config_count, 2)


class TestEvaluationFuncWrapper(unittest.TestCase):
    def test_evaluate(self):
        # Define a sample evaluation function
        def eval_fn(model):
            return model * 2

        # Create an instance of EvaluationFuncWrapper
        wrapper = EvaluationFuncWrapper(eval_fn)

        # Test the evaluate method
        result = wrapper.evaluate(5)
        self.assertEqual(result, 10)


class TestAutoTune(unittest.TestCase):
    def test_autotune(self):
        class AutoTuner:

            @staticmethod
            def _quantize(model, quant_config, *args, **kwargs):
                return model

            def run(
                self, model: FakeModel, tune_config: TuningConfig, eval_fn: Callable, eval_args=None, *args, **kwargs
            ) -> Optional[FakeModel]:
                """The main entry of auto-tune."""
                best_quant_model = None
                eval_func_wrapper = EvaluationFuncWrapper(eval_fn, eval_args)
                config_loader, tuning_logger, tuning_monitor = init_tuning(tuning_config=tune_config)
                baseline: float = eval_func_wrapper.evaluate(model)
                tuning_monitor.set_baseline(baseline)
                tuning_logger.tuning_start()
                for trial_index, quant_config in enumerate(config_loader):
                    tuning_logger.trial_start(trial_index=trial_index)
                    tuning_logger.execution_start()
                    logger.info(quant_config.to_dict())
                    q_model = self._quantize(copy.deepcopy(model), quant_config, *args, **kwargs)
                    tuning_logger.execution_end()
                    tuning_logger.evaluation_start()
                    eval_result: float = eval_func_wrapper.evaluate(q_model)
                    tuning_logger.evaluation_end()
                    tuning_monitor.add_trial_result(trial_index, eval_result, quant_config)
                    tuning_logger.trial_end(trial_index)
                    if tuning_monitor.need_stop():
                        logger.info("Stopped tuning.")
                        del q_model  # maybe gc.collect() is needed for memory release
                        best_quant_config: BaseConfig = tuning_monitor.get_best_quant_config()
                        q_model = self._quantize(copy.deepcopy(model), best_quant_config, *args, **kwargs)
                        best_quant_model = q_model  # quantize model inplace
                        break
                tuning_logger.tuning_end()
                return best_quant_model

        config_set = [FakeAlgoConfig(weight_bits=4), FakeAlgoConfig(weight_bits=8)]
        tuning_config = TuningConfig(config_set=config_set)
        tunner = AutoTuner()
        model = FakeModel()

        def fake_eval_fn(model):
            return 1.0

        q_model = tunner.run(model=model, tune_config=tuning_config, eval_fn=fake_eval_fn)
        self.assertIsNotNone(q_model)


if __name__ == "__main__":
    unittest.main()
