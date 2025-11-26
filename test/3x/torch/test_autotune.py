import unittest
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import patch

import torch
import transformers

import neural_compressor.common.utils.utility as inc_utils
from neural_compressor.common import logger
from neural_compressor.torch.quantization import (
    MixedPrecisionConfig,
    RTNConfig,
    TuningConfig,
    autotune,
    get_all_config_set,
)
from neural_compressor.torch.utils import constants

FAKE_DOUBLE_QUANT_CONFIGS = {
    "BNB_NF4": {
        "dtype": "nf4",
        "bits": 4,
        "group_size": 32,
        "use_double_quant": True,
        "double_quant_bits": 8,
        "double_quant_dtype": "int",
        "double_quant_use_sym": False,
        "double_quant_group_size": 256,
    },
    "GGML_TYPE_Q4_K": {
        "dtype": "int",
        "bits": 4,
        "use_sym": False,
        "group_size": 32,
        "use_double_quant": True,
        "double_quant_bits": 6,
        "double_quant_dtype": "int",
        "double_quant_use_sym": True,
        "double_quant_group_size": 8,
    },
}

from neural_compressor.common.base_tuning import Evaluator


def _create_evaluator_for_eval_fns(eval_fns: Optional[Union[Callable, Dict, List[Dict]]] = None) -> Evaluator:
    evaluator = Evaluator()
    evaluator.set_eval_fn_registry(eval_fns)
    return evaluator


def reset_tuning_target(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Reset tuning targets before running the test
        from neural_compressor.common.base_tuning import evaluator

        evaluator.eval_fn_registry = []
        return test_func(*args, **kwargs)

    return wrapper


def build_simple_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = torch.nn.Linear(30, 50)
            self.fc2 = torch.nn.Linear(50, 30)
            self.fc3 = torch.nn.Linear(30, 5)

        def forward(self, x):
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)
            return out

    model = Model()
    return model


def get_gpt_j():
    import transformers

    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


class GPTQLLMDataLoader:
    def __init__(self, length=512):
        self.batch_size = 1
        self.length = length

    def __iter__(self):
        for i in range(10):
            yield torch.ones([1, self.length], dtype=torch.long)


class GPTQLLMDataLoaderList(GPTQLLMDataLoader):
    def __iter__(self):
        for i in range(10):
            yield (torch.ones([1, self.length], dtype=torch.long), torch.ones([1, self.length], dtype=torch.long))


class GPTQLLMDataLoaderDict(GPTQLLMDataLoader):
    def __iter__(self):
        for i in range(10):
            yield {
                "input_ids": torch.ones([1, self.length], dtype=torch.long),
                "attention_mask": torch.ones([1, self.length], dtype=torch.long),
            }


from tqdm import tqdm

from neural_compressor.torch.algorithms.weight_only.utility import move_input_to_device


def run_fn_for_gptq(model, dataloader_for_calibration, calibration_mode=False):
    logger.info("Collecting calibration inputs...")
    for batch in tqdm(dataloader_for_calibration):
        batch = move_input_to_device(batch, device=None)
        try:
            if isinstance(batch, tuple) or isinstance(batch, list):
                model(batch[0])
            elif isinstance(batch, dict):
                model(**batch)
            else:
                model(batch)
        except ValueError:
            pass
    if not calibration_mode:
        print("Accuracy: 1.0")  # demo the usage


class TestAutoTune(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.fp32_model = build_simple_torch_model()
        self.input = torch.randn(1, 30)
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestAutoTune test: {self.id()}")

    @reset_tuning_target
    def test_autotune_api(self):
        logger.info("test_autotune_api")

        def eval_acc_fn(model) -> float:
            return 1.0

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6])], max_trials=2)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        print(inc_utils.FUNC_CALL_COUNTS)
        self.assertIsNotNone(best_model)

    def test_autotune_return_qmodel_directly(self):
        inc_utils.FUNC_CALL_COUNTS.clear()

        baseline = 1
        eval_result = [0.9, 1.1]
        acc_list = [baseline] + eval_result

        def eval_acc_fn(model) -> float:
            acc = acc_list.pop(0)
            return acc

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6])], max_trials=2)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        assert (
            inc_utils.FUNC_CALL_COUNTS.get("quantize") == 2
        ), f"quantize should be called twice, but got {inc_utils.FUNC_CALL_COUNTS.get('quantize')}"
        self.assertIsNotNone(best_model)

    def test_autotune_return_re_quant_qmodel(self):
        inc_utils.FUNC_CALL_COUNTS.clear()

        baseline = 1
        eval_result = [0.9, 0.8]
        acc_list = [baseline] + eval_result

        def eval_acc_fn(model) -> float:
            acc = acc_list.pop(0)
            return acc

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6])], max_trials=2)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        assert (
            inc_utils.FUNC_CALL_COUNTS.get("quantize") == 3
        ), f"quantize should be called three times, but got {inc_utils.FUNC_CALL_COUNTS.get('quantize')}"
        self.assertIsNotNone(best_model)

    @reset_tuning_target
    def test_autotune_api_2(self):
        logger.info("test_autotune_api")

        def eval_acc_fn(model) -> float:
            return 1.0

        def eval_perf_fn(model) -> float:
            return 1.0

        eval_fns = [
            {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
            {
                "eval_fn": eval_perf_fn,
                "weight": 0.5,
            },
        ]

        evaluator = _create_evaluator_for_eval_fns(eval_fns)

        def eval_fn_wrapper(model):
            result = evaluator.evaluate(model)
            return result

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6])], max_trials=2)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_fn_wrapper)
        self.assertIsNotNone(best_model)
        self.assertEqual(len(evaluator.eval_fn_registry), 2)

    @reset_tuning_target
    def test_autotune_get_config_set_api(self):
        for dataloader in [GPTQLLMDataLoader(), GPTQLLMDataLoaderList(), GPTQLLMDataLoaderDict()]:
            model = get_gpt_j()

            def eval_acc_fn(model) -> float:
                return 1.0

            def eval_perf_fn(model) -> float:
                return 1.0

            eval_fns = [
                {"eval_fn": eval_acc_fn, "weight": 0.5, "name": "accuracy"},
                {
                    "eval_fn": eval_perf_fn,
                    "weight": 0.5,
                },
            ]

            evaluator = _create_evaluator_for_eval_fns(eval_fns)

            def eval_fn_wrapper(model):
                result = evaluator.evaluate(model)
                return result

            custom_tune_config = TuningConfig(config_set=get_all_config_set(), max_trials=4)
            best_model = autotune(
                model=model,
                tune_config=custom_tune_config,
                eval_fn=eval_fn_wrapper,
                run_fn=run_fn_for_gptq,
                run_args=(dataloader, True),  # run_args should be a tuple
            )
            self.assertIsNotNone(best_model)

    def test_autotune_baseline(self):
        logger.info("test_autotune_api")

        baseline = [1.0]

        # case 1
        # Where default tolerable_loss is 0.01, we expect the tuning to end with a "2-trail end" output logged.
        acc_res_lst = baseline + [0.9] * 2 + [0.99]

        def eval_acc_fn(model):
            res = acc_res_lst.pop(0)
            return res

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6, 5, 8])], max_trials=6)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        self.assertIsNotNone(best_model)

        # case 2
        # Where tolerable_loss is 0.1, we expect the tuning to end with a "0-trail end" output logged.
        acc_res_lst = baseline + [0.9] * 2 + [0.99] + [1.01]
        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6, 5, 8])], tolerable_loss=0.1)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        self.assertIsNotNone(best_model)

        # case 3
        # Where tolerable_loss is -0.01, we expect the tuning to end with a "3-trail end" output logged.
        acc_res_lst = baseline + [0.9] * 2 + [0.99] + [1.01]
        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6, 5, 8])], tolerable_loss=-0.01)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        self.assertIsNotNone(best_model)

        # case 4
        # Where tolerable_loss is 0.01 and accuracy doesn't meets the goal, best_model is the best model in trails.
        acc_res_lst = baseline + [0.9] * 2 + [0.9] + [0.9]
        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6, 5, 8])], tolerable_loss=0.01)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        self.assertIsNotNone(best_model)

    @reset_tuning_target
    def test_rtn_double_quant_config_set(self) -> None:
        from neural_compressor.torch.quantization import TuningConfig, autotune, get_rtn_double_quant_config_set
        from neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

        rtn_double_quant_config_set = get_rtn_double_quant_config_set()
        self.assertEqual(len(rtn_double_quant_config_set), len(DOUBLE_QUANT_CONFIGS))

        def eval_acc_fn(model) -> float:
            return 1.0

        custom_tune_config = TuningConfig(config_set=get_rtn_double_quant_config_set(), max_trials=10)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        self.assertIsNotNone(best_model)

    @reset_tuning_target
    def test_rtn_double_quant_config_set2(self) -> None:
        from neural_compressor.torch.quantization import TuningConfig, autotune, get_rtn_double_quant_config_set
        from neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

        rtn_double_quant_config_set = get_rtn_double_quant_config_set()
        self.assertEqual(len(rtn_double_quant_config_set), len(DOUBLE_QUANT_CONFIGS))

        def eval_acc_fn(model) -> float:
            return 1.0

        custom_tune_config = TuningConfig(
            config_set=get_rtn_double_quant_config_set(), max_trials=10, tolerable_loss=-1
        )
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        self.assertIsNotNone(best_model)

    @patch("neural_compressor.torch.utils.constants.DOUBLE_QUANT_CONFIGS", FAKE_DOUBLE_QUANT_CONFIGS)
    def test_rtn_double_quant_config_set3(self) -> None:
        from neural_compressor.torch.quantization import get_rtn_double_quant_config_set

        rtn_double_quant_config_set = get_rtn_double_quant_config_set()
        print(len(rtn_double_quant_config_set))
        self.assertEqual(len(constants.DOUBLE_QUANT_CONFIGS), len(FAKE_DOUBLE_QUANT_CONFIGS))

        def eval_acc_fn(model) -> float:
            return 1.0

        custom_tune_config = TuningConfig(config_set=get_rtn_double_quant_config_set(), tolerable_loss=-1)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)
        self.assertIsNotNone(best_model)

    def test_woq_tuning(self):
        from neural_compressor.torch.quantization import autotune, get_woq_tuning_config

        baseline = [1]
        acc_res_lst = baseline + [0.9, 0.95, 0.95, 0.99, 1.1]

        def eval_acc_fn(model):
            res = acc_res_lst.pop(0)
            return res

        custom_tune_config = TuningConfig(config_set=get_woq_tuning_config(), tolerable_loss=-1)
        example_inputs = torch.ones([1, 32], dtype=torch.long)
        model = get_gpt_j()
        dataloader = GPTQLLMDataLoader()
        best_model = autotune(
            model=model,
            tune_config=custom_tune_config,
            eval_fn=eval_acc_fn,
            run_fn=run_fn_for_gptq,
            run_args=(dataloader, True),  # run_args should be a tuple,
            example_inputs=example_inputs,
        )
        self.assertIsNotNone(best_model)

    @reset_tuning_target
    def test_autotune_mixed_precision_default(self):
        from neural_compressor.torch.algorithms.mixed_precision import HalfPrecisionModuleWrapper

        baseline = [1]
        acc_res_lst = baseline + [0.9, 0.99, 1]

        def eval_acc_fn(model):
            res = acc_res_lst.pop(0)
            return res

        custom_tune_config = TuningConfig(
            config_set=[MixedPrecisionConfig(dtype=["fp16", "bf16", "fp32"])], max_trials=3
        )
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)

        self.assertIsNotNone(best_model)
        self.assertTrue(isinstance(best_model.fc1, HalfPrecisionModuleWrapper))
        self.assertTrue(isinstance(best_model.fc2, HalfPrecisionModuleWrapper))
        self.assertTrue(isinstance(best_model.fc3, HalfPrecisionModuleWrapper))

    @reset_tuning_target
    def test_autotune_mixed_precision_set_op_name(self):
        from neural_compressor.common.base_config import ComposableConfig, config_registry
        from neural_compressor.torch.algorithms.mixed_precision import HalfPrecisionModuleWrapper

        baseline = [1]
        acc_res_lst = baseline + [0.9, 1.1]

        def eval_acc_fn(model):
            res = acc_res_lst.pop(0)
            return res

        config1 = {
            "mixed_precision": {
                "global": {
                    "dtype": "bf16",
                },
                "local": {
                    "fc2": {
                        "dtype": "fp32",
                    }
                },
            }
        }
        config2 = {
            "mixed_precision": {
                "global": {
                    "dtype": "fp16",
                },
                "local": {
                    "fc1": {
                        "dtype": "fp32",
                    }
                },
            }
        }

        registered_configs = config_registry.get_cls_configs()
        config1 = ComposableConfig.from_dict(config1, config_registry=registered_configs["torch"])
        config2 = ComposableConfig.from_dict(config2, config_registry=registered_configs["torch"])

        custom_tune_config = TuningConfig(config_set=[config1, config2], max_trials=2)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fn=eval_acc_fn)

        self.assertIsNotNone(best_model)
        self.assertTrue(isinstance(best_model.fc1, torch.nn.Linear))
        self.assertTrue(isinstance(best_model.fc2, HalfPrecisionModuleWrapper))
        self.assertTrue(isinstance(best_model.fc3, HalfPrecisionModuleWrapper))


if __name__ == "__main__":
    unittest.main()
