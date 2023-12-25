import unittest

import transformers

from neural_compressor.common.logger import Logger

logger = Logger().get_logger()
from functools import wraps

import torch


def reset_tuning_target(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Reset tuning targets before running the test
        from neural_compressor.common.base_tune import target_manager

        target_manager.eval_fn_registry = []
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
        from neural_compressor.common.base_tune import target_manager
        from neural_compressor.torch import RTNWeightQuantConfig, TuningConfig, autotune, register_tuning_target

        @register_tuning_target(weight=0.5, target_name="accuracy")
        def eval_acc_fn(model) -> float:
            return 1.0

        @register_tuning_target(weight=-0.5)
        def eval_perf_fn(model) -> float:
            return 1.0

        custom_tune_config = TuningConfig(tuning_order=[RTNWeightQuantConfig(weight_bits=[4, 6])], max_trials=3)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config)
        self.assertIsNone(best_model)
        self.assertEqual(len(target_manager.eval_fn_registry), 2)

    @reset_tuning_target
    def test_autotune_not_register_eval_func(self):
        logger.info("test_autotune_api")
        from neural_compressor.torch import RTNWeightQuantConfig, TuningConfig, autotune

        custom_tune_config = TuningConfig(tuning_order=[RTNWeightQuantConfig(weight_bits=[4, 6])], max_trials=3)

        # Use assertRaises to check that an AssertionError is raised
        with self.assertRaises(AssertionError) as context:
            best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config)
        self.assertEqual(
            str(context.exception), "Please ensure that you register at least one evaluation metric for auto-tune."
        )


if __name__ == "__main__":
    unittest.main()
