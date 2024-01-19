import unittest

import transformers

from neural_compressor.common import Logger

logger = Logger().get_logger()
from functools import wraps

import torch


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
        from neural_compressor.common.base_tuning import evaluator
        from neural_compressor.torch import RTNWeightQuantConfig, TuningConfig, autotune

        def eval_acc_fn(model) -> float:
            return 1.0

        custom_tune_config = TuningConfig(quant_configs=[RTNWeightQuantConfig(weight_bits=[4, 6])], max_trials=2)
        best_model = autotune(
            model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=[{"eval_fn": eval_acc_fn}]
        )
        self.assertIsNotNone(best_model)
        self.assertEqual(len(evaluator.eval_fn_registry), 1)

    @reset_tuning_target
    def test_autotune_api_2(self):
        logger.info("test_autotune_api")
        from neural_compressor.common.base_tuning import evaluator
        from neural_compressor.torch import RTNWeightQuantConfig, TuningConfig, autotune

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

        custom_tune_config = TuningConfig(quant_configs=[RTNWeightQuantConfig(weight_bits=[4, 6])], max_trials=2)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=eval_fns)
        self.assertIsNotNone(best_model)
        self.assertEqual(len(evaluator.eval_fn_registry), 2)

    @reset_tuning_target
    def test_autotune_not_eval_func(self):
        logger.info("test_autotune_api")
        from neural_compressor.torch import RTNWeightQuantConfig, TuningConfig, autotune

        custom_tune_config = TuningConfig(quant_configs=[RTNWeightQuantConfig(weight_bits=[4, 6])], max_trials=2)

        # Use assertRaises to check that an AssertionError is raised
        with self.assertRaises(AssertionError) as context:
            best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config)
        self.assertEqual(
            str(context.exception), "Please ensure that you register at least one evaluation metric for auto-tune."
        )


if __name__ == "__main__":
    unittest.main()
