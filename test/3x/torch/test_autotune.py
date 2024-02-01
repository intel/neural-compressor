import unittest
from functools import wraps

import torch
import transformers

from neural_compressor.torch.quantization import RTNConfig, TuningConfig, autotune, get_all_config_set
from neural_compressor.torch.utils import logger


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

from neural_compressor.torch.algorithms.weight_only.gptq import move_input_to_device


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
        from neural_compressor.common.base_tuning import evaluator

        def eval_acc_fn(model) -> float:
            return 1.0

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6])], max_trials=2)
        best_model = autotune(
            model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=[{"eval_fn": eval_acc_fn}]
        )
        self.assertIsNotNone(best_model)
        self.assertEqual(len(evaluator.eval_fn_registry), 1)

    @reset_tuning_target
    def test_autotune_api_2(self):
        logger.info("test_autotune_api")
        from neural_compressor.common.base_tuning import evaluator

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

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6])], max_trials=2)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=eval_fns)
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
            custom_tune_config = TuningConfig(config_set=get_all_config_set(), max_trials=4)
            best_model = autotune(
                model=model,
                tune_config=custom_tune_config,
                eval_fns=eval_fns,
                run_fn=run_fn_for_gptq,
                run_args=(dataloader, True),  # run_args should be a tuple
            )
            self.assertIsNotNone(best_model)

    @reset_tuning_target
    def test_autotune_not_eval_func(self):
        logger.info("test_autotune_api")

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6])], max_trials=2)

        # Use assertRaises to check that an AssertionError is raised
        with self.assertRaises(AssertionError) as context:
            best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config)
        self.assertEqual(
            str(context.exception), "Please ensure that you register at least one evaluation metric for auto-tune."
        )


if __name__ == "__main__":
    unittest.main()
