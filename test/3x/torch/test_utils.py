import unittest

import torch

from neural_compressor.torch.utils import logger


def get_gpt_j():
    import transformers

    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


def build_simple_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = torch.nn.Linear(8, 30)
            self.fc2 = torch.nn.Linear(30, 60)
            self.fc3 = torch.nn.Linear(60, 30)
            self.fc4 = torch.nn.Linear(30, 50)

        def forward(self, x):
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.fc4(out)
            return out

    model = Model()
    return model


from neural_compressor.torch.utils.utility import fetch_module, set_module


class TestTorchUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = get_gpt_j()

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        # print the test name
        logger.info(f"Running TestTorchUtils test: {self.id()}")

    def test_fetch_module(self):
        result = fetch_module(self.model, "transformer.h.2.mlp.fc_in")
        self.assertIsInstance(result, torch.nn.Linear)

    def test_set_module(self):
        module_name = "transformer.h.2.mlp.fc_in"
        mew_value = torch.nn.Linear(32, 128, bias=False)
        set_module(self.model, module_name, mew_value)
        result = fetch_module(self.model, module_name)
        self.assertFalse(result.bias)

    def test_set_module_nonexistent_attribute(self):
        new_value = torch.nn.Parameter(torch.Tensor([3.0]))
        attr_name = "transformer.nonexistent_attr"
        set_module(self.model, attr_name, new_value)
        result = fetch_module(self.model, attr_name)
        self.assertTrue(torch.equal(result, torch.Tensor([3.0])))

    def test_fetch_module_nonexistent_attribute(self):
        attr_name = "transformer.nonexistent_attr"
        result = fetch_module(self.model, attr_name)
        self.assertIsNone(result)

    def test_get_model_info(self):
        from neural_compressor.torch.utils.utility import get_model_info

        white_module_list = [torch.nn.Linear]
        model_info = get_model_info(build_simple_torch_model(), white_module_list)
        self.assertEqual(len(model_info), 4)


if __name__ == "__main__":
    unittest.main()
