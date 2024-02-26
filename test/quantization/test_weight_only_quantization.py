import copy
import shutil
import unittest

import torch
import transformers

from neural_compressor.adaptor.torch_utils.model_wrapper import WeightOnlyLinear
from neural_compressor.adaptor.torch_utils.smooth_quant import GraphTrace
from neural_compressor.adaptor.torch_utils.weight_only import (
    autoround_quantize,
    awq_quantize,
    gptq_quantize,
    rtn_quantize,
    teq_quantize,
)

try:
    import auto_round

    auto_round_installed = True
except ImportError:
    auto_round_installed = False


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 32)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class SimpleDataLoader:
    def __init__(self):
        self.batch_size = 1
        self.input = torch.randn([1, 32])

    def __iter__(self):
        yield self.input


class TestAWQWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = Model()
        self.dataloader = SimpleDataLoader()
        self.example_inputs = torch.randn([1, 32])
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    def test_trace(self):
        op_types = ["Linear"]
        tg = GraphTrace()
        # absorb_to_layer={'absorb_layer': absorbed_layer}
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(self.model, self.example_inputs, op_types)
        self.assertTrue(len(no_absorb_layers) == 1)
        absorb_to_layer, no_absorb_layers = tg.get_absorb_to_layer(self.gptj, self.lm_input, op_types)
        self.assertTrue(len(no_absorb_layers) == 11)
        return absorb_to_layer, no_absorb_layers

    def test_rtn(self):
        fp32_model = copy.deepcopy(self.model)
        fp16_model = copy.deepcopy(self.model).to(torch.float16)
        model1 = rtn_quantize(fp32_model, num_bits=3, group_size=-1)
        self.assertTrue(isinstance(model1.fc1, torch.nn.Linear))
        weight_config = {
            # 'op_name': (bit, group_size, scheme)
            "fc1": {"bits": 8, "group_size": -1, "scheme": "sym"},
            "fc2": {
                "bits": 4,
                "group_size": 32,
                "scheme": "asym",
                "quantile": 0.95,  # not required.
            },
        }
        model2 = rtn_quantize(fp32_model, weight_config=weight_config)
        model2 = rtn_quantize(fp16_model, weight_config=weight_config, return_int=True)
        self.assertTrue(isinstance(model2.fc1, WeightOnlyLinear))

    def test_awq(self):
        example_inputs = torch.ones([1, 10], dtype=torch.long)
        from neural_compressor.adaptor.torch_utils.awq import ActAwareWeightQuant

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torchscript=True,
        )

        class LLMCalibDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(2):
                    yield example_inputs

        out1 = model(example_inputs)
        awq = ActAwareWeightQuant(model, dataloader=LLMCalibDataloader(), bits=8, group_size=-1)
        qdq_model = awq.quantize()
        out2 = qdq_model(example_inputs)
        # output data is up to 4, so use big atol=0.5
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=0.5))

        def calib_func(model):
            for i in range(2):
                model(self.lm_input)

        out1 = self.gptj(example_inputs)
        awq = ActAwareWeightQuant(self.gptj, calib_func=calib_func, example_inputs=self.lm_input, bits=8, group_size=-1)
        qdq_model = awq.quantize()
        out2 = qdq_model(example_inputs)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-2))

        # default awq_quantize is 4 bits, 32 group size, use big atol=1e-1
        qdq_model = awq_quantize(self.gptj, example_inputs=self.lm_input, calib_func=calib_func)
        out2 = qdq_model(example_inputs)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-1))


class TestGPTQWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )

    def test_gptq(self):
        import random

        class GPTQLLMDataLoader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(20):
                    length = random.randint(1, 1024)
                    yield torch.ones([1, length], dtype=torch.long)

        dataloader = GPTQLLMDataLoader()
        model = copy.deepcopy(self.gptj)
        weight_config = {
            "transformer.h.0.attn.k_proj": {
                "wbits": 4,
                "group_size": 128,
                "sym": True,
                "percdamp": 0.01,
                "perchannel": False,
            },
            "transformer.h.1.attn.k_proj": {
                "wbits": 3,
                "group_size": -1,
                "sym": False,
                "percdamp": 0.01,
                "act_order": True,
                "static_groups": True,
            },
            "transformer.h.2.attn.k_proj": {
                "wbits": 3,
                "group_size": 32,
                "sym": False,
                "percdamp": 0.01,
                "mse": True,
                "act_order": False,
            },
            "transformer.h.3.attn.k_proj": {
                "wbits": 3,
                "group_size": 256,
                "sym": False,
                "percdamp": 0.01,
                "mse": True,
                "act_order": False,
            },
        }
        quantizer = gptq_quantize(
            model, weight_config=weight_config, dataloader=dataloader, use_max_length=True, pad_max_length=512
        )
        self.assertTrue(isinstance(model, torch.nn.Module))

        model = copy.deepcopy(self.gptj)
        weight_config = {"wbits": 4}
        quantizer = gptq_quantize(
            model, weight_config=weight_config, dataloader=dataloader, use_max_length=False, pad_max_length=512
        )
        self.assertTrue(isinstance(model, torch.nn.Module))


class TestTEQWeightOnlyQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.gptj.seqlen = 512

    def generate_random_corpus(self, nsamples=32):
        meta_data = []
        for _ in range(nsamples):
            inp = torch.ones([1, 512], dtype=torch.long)
            tar = torch.ones([1, 512], dtype=torch.long)
            meta_data.append((inp, tar))
        return meta_data

    def train_func(self):
        pass

    def test_teq(self):
        dataloader = self.generate_random_corpus()
        model = copy.deepcopy(self.gptj)

        weight_config = {
            # 'op_name': (bit, group_size, scheme)
            "transformer.h.0.mlp.fc_in": {"bits": 8, "group_size": -1, "scheme": "sym"},
            "transformer.h.0.mlp.fc_out": {"bits": 4, "group_size": 32, "scheme": "asym"},
        }
        absorb_dict = {"transformer.h.0.mlp.fc_in": ["transformer.h.0.mlp.fc_out"]}
        extra_config = {"folding": True}

        model = teq_quantize(
            model,
            weight_config=weight_config,
            absorb_to_layer=absorb_dict,
            extra_config=extra_config,
            dataloader=dataloader,
        )
        self.assertTrue(isinstance(model, torch.nn.Module))


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


@unittest.skipIf(not auto_round_installed, "auto_round module is not installed")
class TestAutoRoundWeightOnlyQuant(unittest.TestCase):
    approach = "weight_only"

    @classmethod
    def setUpClass(self):
        self.dataloader = SimpleDataLoader()
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )
        self.gptj_no_jit = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
        )
        self.llm_dataloader = LLMDataLoader()
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autoround_int_quant(self):
        model = copy.deepcopy(self.gptj)
        device = "cpu"
        model = model
        out1 = model(self.lm_input)
        q_model, weight_config1 = autoround_quantize(
            model=model,
            tokenizer=self.tokenizer,
            n_samples=20,
            device=device,
            amp=False,
            seqlen=10,
            iters=10,
            scale_dtype="fp32",
        )
        q_model = q_model
        model = model
        out2 = model(self.lm_input)
        out3 = q_model(self.lm_input)
        self.assertTrue(torch.all(torch.isclose(out1[0], out2[0], atol=1e-1)))
        self.assertFalse(torch.all(out1[0] == out2[0]))
        self.assertTrue(torch.all(out2[0] == out3[0]))


if __name__ == "__main__":
    unittest.main()
