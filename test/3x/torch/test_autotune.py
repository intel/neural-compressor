import unittest
from functools import wraps

import torch
import transformers

from neural_compressor.torch.algorithms.weight_only.gptq import DataloaderPreprocessor
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


def run_fn_for_gptq(model, dataloader_for_calibration, *args):
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
    return


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
        dataloader = GPTQLLMDataLoader()

        model = get_gpt_j()
        input = torch.ones([1, 512], dtype=torch.long)

        dataloaderPreprocessor = DataloaderPreprocessor(
            dataloader_original=dataloader, use_max_length=False, pad_max_length=512, nsamples=128
        )
        dataloader_for_calibration = dataloaderPreprocessor.get_prepared_dataloader()

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
            model=get_gpt_j(),
            tune_config=custom_tune_config,
            eval_fns=eval_fns,
            run_fn=run_fn_for_gptq,
            run_args=dataloader_for_calibration,
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

    def test_autotune_baseline(self):
        logger.info("test_autotune_api")
        from neural_compressor.common.base_tuning import evaluator

        baseline = [1.0]

        # case 1
        # Where default tolerable_loss is 0.01, we expect the tuning to end with a "2-trail end" output logged.
        acc_res_lst = baseline + [0.9] * 2 + [0.99]

        def eval_acc_fn(model):
            res = acc_res_lst.pop(0)
            return res

        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6, 5, 8])], max_trials=6)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=eval_acc_fn)
        self.assertIsNotNone(best_model)

        # case 2
        # Where tolerable_loss is 0.1, we expect the tuning to end with a "0-trail end" output logged.
        acc_res_lst = baseline + [0.9] * 2 + [0.99] + [1.01]
        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6, 5, 8])], tolerable_loss=0.1)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=eval_acc_fn)
        self.assertIsNotNone(best_model)

        # case 3
        # Where tolerable_loss is -0.01, we expect the tuning to end with a "3-trail end" output logged.
        acc_res_lst = baseline + [0.9] * 2 + [0.99] + [1.01]
        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6, 5, 8])], tolerable_loss=-0.01)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=eval_acc_fn)
        self.assertIsNotNone(best_model)

        # case 4
        # Where tolerable_loss is 0.01 and accuracy meets the goal, we expect best model is None.
        acc_res_lst = baseline + [0.9] * 2 + [0.9] + [0.9]
        custom_tune_config = TuningConfig(config_set=[RTNConfig(bits=[4, 6, 5, 8])], tolerable_loss=0.01)
        best_model = autotune(model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=eval_acc_fn)
        self.assertIsNone(best_model)

    @reset_tuning_target
    def test_rtn_double_quant_config_set(self) -> None:
        from neural_compressor.torch.quantization import TuningConfig, autotune, get_rtn_double_quant_config_set
        from neural_compressor.torch.utils.constants import DOUBLE_QUANT_CONFIGS

        rtn_double_quant_config_set = get_rtn_double_quant_config_set()
        self.assertEqual(len(rtn_double_quant_config_set), len(DOUBLE_QUANT_CONFIGS))

        def eval_acc_fn(model) -> float:
            return 1.0

        custom_tune_config = TuningConfig(config_set=get_rtn_double_quant_config_set(), max_trials=10)
        best_model = autotune(
            model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=[{"eval_fn": eval_acc_fn}]
        )
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
        best_model = autotune(
            model=build_simple_torch_model(), tune_config=custom_tune_config, eval_fns=[{"eval_fn": eval_acc_fn}]
        )
        self.assertIsNone(best_model)


if __name__ == "__main__":
    unittest.main()
