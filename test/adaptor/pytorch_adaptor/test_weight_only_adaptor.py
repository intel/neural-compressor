import copy
import os
import shutil
import unittest

import torch
import transformers

from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.adaptor.torch_utils.model_wrapper import MulLinear, WeightOnlyLinear
from neural_compressor.model import Model as INCModel
from neural_compressor.utils.load_huggingface import export_compressed_model
from neural_compressor.utils.pytorch import load


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(30, 50, bias=True)
        self.fc2 = torch.nn.Linear(50, 30)
        self.fc3 = torch.nn.Linear(30, 5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def eval_func(model):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        input = torch.randn(3, 30)
        # compute output
        output = model(input)
    return 0.0


class SimpleDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.randn([1, 30])


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestPytorchWeightOnlyAdaptor(unittest.TestCase):
    approach = "weight_only"

    @classmethod
    def setUpClass(self):
        self.dataloader = SimpleDataLoader()
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM",
            torchscript=True,
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

    def test_RTN_int_quant(self):
        input = torch.randn(3, 30)
        model = Model()
        out1 = model(input)
        conf = PostTrainingQuantConfig(
            approach="weight_only",
        )
        q_model = quantization.fit(model, conf)
        q_model.save("saved")
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))
        compressed_model = q_model.export_compressed_model(use_optimum_format=False)
        out3 = compressed_model(input)
        self.assertTrue("fc1.qweight" in compressed_model.state_dict().keys())
        self.assertTrue("fc1.qzeros" not in compressed_model.state_dict().keys())
        shape2 = compressed_model.state_dict()["fc1.scales"]
        self.assertTrue(torch.all(out3 == out2))

        # test huggingface popular int4 format
        model = Model()
        new_model = load("saved", model, weight_only=True)
        inc_model = INCModel(new_model)
        inc_model.export_compressed_model(qweight_config_path="saved/qconfig.json", use_optimum_format=True)
        out4 = inc_model.model(input)
        self.assertTrue("fc1.qzeros" in inc_model.model.state_dict().keys())
        model = Model()
        compressed_model = export_compressed_model(model, saved_dir="saved", use_optimum_format=True)
        self.assertTrue("fc1.qzeros" in inc_model.model.state_dict().keys())
        # output gap is because of torch.float16 is used in hf_format
        self.assertTrue(torch.allclose(out3, out4, atol=1e-3))

        model = Model()
        out1 = model(input)
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            recipes={
                # By default, enable_full_range is False and 4 bit sym will only use range [-7,7].
                "rtn_args": {"enable_full_range": True}
            },
        )
        q_model = quantization.fit(model, conf)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))
        compressed_model = q_model.export_compressed_model(use_optimum_format=False, enable_full_range=True)
        out3 = compressed_model(input)
        self.assertTrue(torch.all(out3 == out2))

        model = Model()
        out1 = model(input)
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "dtype": "int4",  # 1-8 bits
                    },
                },
            },
            recipes={
                # By default, enable_full_range is False and 4 bit sym will only use range [-7,7].
                # When enable_mse_search is set to True, enable clip for weight by checking mse.
                "rtn_args": {"enable_full_range": True, "enable_mse_search": True}
            },
        )
        q_model = quantization.fit(model, conf)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))

        model = Model()
        out1 = model(input)
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            recipes={
                # 0 means splitting output channel
                "rtn_args": {"group_dim": 0}
            },
        )
        q_model = quantization.fit(model, conf)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))

        model = Model()
        out1 = model(input)
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 8,  # 1-8 bits
                        "group_size": -1,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "RTN",
                    },
                },
            },
            recipes={
                # By default, enable_full_range is False and 4 bit sym will only use range [-7,7].
                "rtn_args": {"return_int": True}
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(isinstance(q_model.model.fc1, WeightOnlyLinear))
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))

        model = Model()
        out1 = model(input)
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # 1 - 1024 or higher
                        "scheme": "asym",
                        "algorithm": "RTN",
                    },
                },
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))

        model = Model()
        out1 = model(input)
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_name_dict={
                "fc1": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # 1 - 1024 or higher
                        "scheme": "sym",
                        "algorithm": "RTN",
                    },
                },
                "fc2": {  # re.match
                    "weight": {
                        "bits": 3,  # 1-8 bits
                        "group_size": 16,  # 1 - 1024 or higher
                        "scheme": "asym",
                        "algorithm": "RTN",
                    },
                },
                "fc3": {  # re.match
                    "weight": {
                        "dtype": "fp32",
                    },
                },
            },
        )
        q_model = quantization.fit(model, conf, eval_func=eval_func)
        out2 = q_model(input)
        self.assertTrue(torch.all(torch.isclose(out1, out2, atol=5e-1)))
        self.assertFalse(torch.all(out1 == out2))
        q_model.save("saved")

        new_model = load("saved", model, weight_only=True)
        out1 = new_model(input)
        self.assertTrue(torch.all(out1 == out2))

        model_size1 = os.path.getsize("saved/best_model.pt") / 1024
        print("FP32 Model size:{:.3f}M".format(model_size1))
        inc_model = INCModel(new_model)
        inc_model.export_compressed_model(use_optimum_format=False, qweight_config_path="saved/qconfig.json")
        torch.save(inc_model.state_dict(), "saved/tmp.pt")
        model_size2 = os.path.getsize("saved/tmp.pt") / 1024
        print("WeightOnlyLinear Model size:{:.3f}M".format(model_size2))
        self.assertTrue(isinstance(inc_model.model.fc1, WeightOnlyLinear))
        self.assertTrue(model_size1 / model_size2 > 2)

    def test_RTN_4bit_quant(self):
        for dtype in ["int4", "nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"]:
            model = copy.deepcopy(self.gptj)
            out1 = model(self.lm_input)
            conf = PostTrainingQuantConfig(
                approach="weight_only",
                op_type_dict={
                    ".*": {  # re.match
                        "weight": {
                            "dtype": dtype,  # select from int, nf4, or fp4
                            # nf4/fp4 have fixed bits and scheme.
                            "group_size": 64,  # -1 (per-channel)
                            "algorithm": "RTN",
                        },
                    },
                },
            )
            q_model = quantization.fit(model, conf)
            out2 = q_model(self.lm_input)
            self.assertTrue(torch.all(torch.isclose(out1[0], out2[0], atol=1e-1)))
            self.assertFalse(torch.all(out1[0] == out2[0]))
            compressed_model = q_model.export_compressed_model(use_optimum_format=False)
            out3 = compressed_model(self.lm_input)
            self.assertTrue(torch.all(out3[0] == out2[0]))

    def test_AWQ_quant(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # -1 (per-channel)
                        "scheme": "asym",
                        "algorithm": "AWQ",
                    },
                },
            },
            op_name_dict={
                ".*3.*": {  # re.match
                    "weight": {"dtype": "fp32"},
                },
                ".*4.*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # -1 (per-channel)
                        "scheme": "asym",
                        "algorithm": "RTN",
                    },
                },
                ".*lm_head": {  # re.match
                    "weight": {"dtype": "fp32"},
                },
            },
            recipes={
                "awq_args": {"enable_auto_scale": True, "enable_mse_search": True, "folding": False},
            },
        )
        fp32_model = copy.deepcopy(self.gptj)
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=self.llm_dataloader,
        )
        q_model.save("saved")
        input = torch.ones([1, 10], dtype=torch.long)
        out1 = q_model(input)
        from neural_compressor.utils.pytorch import load

        fp32_model = copy.deepcopy(self.gptj)
        reload_model = load("saved", fp32_model, weight_only=True)
        out2 = reload_model(input)
        q_model.export_compressed_model(use_optimum_format=False)
        out3 = q_model(input)
        # no idea about the gap at 1e-08, use allclose instead of out1==out2
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-05))
        self.assertTrue(torch.allclose(out1[0], out3[0], atol=1e-05))
        self.assertTrue(isinstance(q_model.model.transformer.h[0].mlp.fc_in, WeightOnlyLinear))
        self.assertTrue(isinstance(q_model.model.lm_head, torch.nn.Linear))

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # -1 (per-channel)
                        "scheme": "asym",
                        "algorithm": "AWQ",
                    },
                },
            },
            op_name_dict={
                ".*3.*": {  # re.match
                    "weight": {"dtype": "fp32"},
                },
                ".*4.*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # -1 (per-channel)
                        "scheme": "asym",
                        "algorithm": "RTN",
                    },
                },
                ".*lm_head": {  # re.match
                    "weight": {"dtype": "fp32"},
                },
            },
            recipes={
                "rtn_args": {"return_int": True},
                "awq_args": {"enable_auto_scale": True, "enable_mse_search": True, "folding": False},
            },
        )
        fp32_model = copy.deepcopy(self.gptj)
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=self.llm_dataloader,
        )
        self.assertTrue(isinstance(q_model.model.transformer.h[0].mlp.fc_out, MulLinear))
        self.assertTrue(isinstance(q_model.model.transformer.h[3].mlp.fc_out, torch.nn.Linear))
        self.assertTrue(isinstance(q_model.model.transformer.h[4].mlp.fc_out, WeightOnlyLinear))

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # -1 (per-channel)
                        "scheme": "asym",
                        "algorithm": "AWQ",
                    },
                },
            },
        )
        fp32_model = copy.deepcopy(self.gptj_no_jit)
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=self.llm_dataloader,
        )
        self.assertTrue(isinstance(q_model.model.transformer.h[0].mlp.fc_in, MulLinear))
        self.assertTrue(isinstance(q_model.model.transformer.h[0].mlp.fc_out, MulLinear))

    def test_AWQ_nf4_quant(self):
        input = torch.ones([1, 10], dtype=torch.long)
        fp32_model = copy.deepcopy(self.gptj)
        out1 = fp32_model(input)
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "dtype": "nf4",  # select from int, nf4, or fp4
                        # nf4/fp4 have fixed bits and scheme.
                        "group_size": 32,  # -1 (per-channel)
                        "algorithm": "RTN",
                    },
                },
            },
            op_name_dict={
                "lm_head": {  # re.match
                    "weight": {
                        "dtype": "fp32",
                    },
                },
            },
        )
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=self.llm_dataloader,
        )
        out2 = q_model(input)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-01))
        compressed_model = q_model.export_compressed_model(use_optimum_format=False)
        out3 = compressed_model(input)
        self.assertTrue(torch.all(out3[0] == out2[0]))

    def test_AWQ_util(self):
        from neural_compressor.adaptor.torch_utils.util import get_module_input_output

        class DemoModel(torch.nn.Module):
            def __init__(self):
                super(DemoModel, self).__init__()
                self.fc1 = torch.nn.Linear(3, 4)
                self.fc2 = torch.nn.Linear(4, 3)

            def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                return out

        tmp = torch.randn([3, 3])

        class DemoCalibDataloader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(3):
                    yield tmp

        module_hook_config = {"fc1": ["output"], "fc2": ["input", "output"]}
        model = DemoModel()
        out = model(tmp)
        values = get_module_input_output(model, module_hook_config, DemoCalibDataloader())
        self.assertTrue(torch.allclose(values["fc1"]["output"][0], values["fc2"]["input"][0]))
        self.assertTrue(torch.allclose(values["fc2"]["output"][0], out))

    def test_GPTQ_fixed_length_quant(self):
        class GPTQLLMDataLoader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(10):
                    yield torch.ones([1, 512], dtype=torch.long)

        class GPTQLLMDataLoaderList:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(10):
                    yield (torch.ones([1, 512], dtype=torch.long), torch.ones([1, 512], dtype=torch.long))

        class GPTQLLMDataLoaderDict:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(10):
                    yield {
                        "input_ids": torch.ones([1, 512], dtype=torch.long),
                        "attention_mask": torch.ones([1, 512], dtype=torch.long),
                    }

        dataloader = GPTQLLMDataLoader()
        dataloader_list = GPTQLLMDataLoaderList()
        dataloader_dict = GPTQLLMDataLoaderDict()

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 8,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "GPTQ",
                    },
                },
            },
            op_name_dict={
                ".*lm_head": {  # re.match
                    "weight": {"dtype": "fp32"},
                },
            },
            recipes={
                "gptq_args": {"percdamp": 0.01, "act_order": False, "use_max_length": True, "pad_max_length": 512},
            },
        )

        # case 1: tensor
        model_1 = copy.deepcopy(self.gptj)
        input = torch.ones([1, 512], dtype=torch.long)
        out0 = model_1(input)
        q_model = quantization.fit(
            model_1,
            conf,
            calib_dataloader=dataloader,
        )
        q_model.save("saved")
        out1 = q_model.model(input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=1e-02))
        compressed_model = q_model.export_compressed_model(use_optimum_format=False)
        out2 = compressed_model(input)
        torch.save(compressed_model.state_dict(), "saved/compressed_model.pt")
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-05))

        # # case 2: list or tuple
        model_2 = copy.deepcopy(self.gptj)
        input = torch.ones([1, 512], dtype=torch.long)
        conf.op_type_dict = {
            ".*": {  # re.match
                "weight": {
                    "bits": 4,  # 1-8 bits
                    "group_size": 8,  # -1 (per-channel)
                    "scheme": "asym",
                    "algorithm": "GPTQ",
                },
            },
        }
        q_model = quantization.fit(
            model_2,
            conf,
            calib_dataloader=dataloader_list,
        )
        q_model.save("saved")
        out1 = q_model.model(input)
        compressed_model = q_model.export_compressed_model(use_optimum_format=True)
        out2 = compressed_model(input)
        print(out1[0])
        print(out2[0])
        torch.save(compressed_model.state_dict(), "saved/compressed_model.pt")
        # hf_format uses fp16 for scale, so output atol is higher.
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=2e-04))

        # # case 2: list or tuple
        model_3 = copy.deepcopy(self.gptj)
        input = torch.ones([1, 512], dtype=torch.long)
        q_model = quantization.fit(
            model_3,
            conf,
            calib_dataloader=dataloader_dict,
        )
        q_model.save("saved")
        out1 = q_model.model(input)
        compressed_model = q_model.export_compressed_model(use_optimum_format=False)
        out2 = compressed_model(input)
        torch.save(compressed_model.state_dict(), "saved/compressed_model.pt")
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-05))

        print("GPTQ with fixed length Done")

    def test_GPTQ_unfixed_length_quant(self):
        import random

        class GPTQLLMDataLoader:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(10):
                    length = random.randint(1, 1024)
                    yield torch.ones([1, length], dtype=torch.long)

        class GPTQLLMDataLoaderList:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(10):
                    length = random.randint(1, 1024)
                    yield (torch.ones([1, length], dtype=torch.long), torch.ones([1, length], dtype=torch.long))

        class GPTQLLMDataLoaderDict:
            def __init__(self):
                self.batch_size = 1

            def __iter__(self):
                for i in range(10):
                    length = random.randint(1, 1024)
                    yield {
                        "input_ids": torch.ones([1, length], dtype=torch.long),
                        "attention_mask": torch.ones([1, length], dtype=torch.long),
                    }

        dataloader = GPTQLLMDataLoader()
        dataloader_list = GPTQLLMDataLoaderList()
        dataloader_dict = GPTQLLMDataLoaderDict()

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 8,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "GPTQ",
                    },
                },
            },
            op_name_dict={
                ".*lm_head": {  # re.match
                    "weight": {"dtype": "fp32"},
                },
            },
            recipes={
                "gptq_args": {"percdamp": 0.01, "act_order": False, "use_max_length": False, "pad_max_length": 512},
            },
        )

        # case 1: tensor
        model_1 = copy.deepcopy(self.gptj)
        input = torch.ones([1, 512], dtype=torch.long)
        out0 = model_1(input)
        q_model = quantization.fit(
            model_1,
            conf,
            calib_dataloader=dataloader,
        )
        q_model.save("saved")
        out1 = q_model.model(input)
        self.assertTrue(torch.allclose(out1[0], out0[0], atol=1e-02))
        compressed_model = q_model.export_compressed_model()
        out2 = compressed_model(input)
        torch.save(compressed_model.state_dict(), "saved/compressed_model.pt")
        # hf_format uses fp16 for scale, so output atol is higher.
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=2e-04))

        # # case 2: list or tuple
        model_2 = copy.deepcopy(self.gptj)
        input = torch.ones([1, 512], dtype=torch.long)
        q_model = quantization.fit(
            model_2,
            conf,
            calib_dataloader=dataloader_list,
        )
        q_model.save("saved")
        out1 = q_model.model(input)
        compressed_model = q_model.export_compressed_model(use_optimum_format=False)
        out2 = compressed_model(input)
        torch.save(compressed_model.state_dict(), "saved/compressed_model.pt")
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-05))

        # # case 2: list or tuple
        model_3 = copy.deepcopy(self.gptj)
        input = torch.ones([1, 512], dtype=torch.long)
        q_model = quantization.fit(
            model_3,
            conf,
            calib_dataloader=dataloader_dict,
        )
        q_model.save("saved")
        out1 = q_model.model(input)
        compressed_model = q_model.export_compressed_model()
        out2 = compressed_model(input)
        torch.save(compressed_model.state_dict(), "saved/compressed_model.pt")
        # hf_format uses fp16 for scale, so output atol is higher.
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=2e-04))

        print("GPTQ with unfixed length Done")

    def test_TEQ_quant(self):
        class teq_inc_loader(object):
            def __init__(self, nsamples=32):
                self.batch_size = 1
                self.nsamples = nsamples

            def __len__(self):
                return self.nsamples // self.batch_size

            def __iter__(self):
                for i in range(self.nsamples):
                    yield (torch.ones([1, 512], dtype=torch.long), torch.ones([1, 512], dtype=torch.long))

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "TEQ",
                    },
                },
            },
            op_name_dict={
                ".*lm_head": {  # re.match
                    "weight": {"dtype": "fp32"},
                },
            },
            recipes={
                "teq_args": {"folding": True},
            },
        )

        input = torch.ones([1, 512], dtype=torch.long)
        dataloader = teq_inc_loader()
        fp32_model = copy.deepcopy(self.gptj)
        out1 = fp32_model(input)
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=dataloader,
        )
        out2 = q_model.model(input)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-01))

    def test_AutoRound_quant(self):
        from neural_compressor.adaptor.torch_utils.auto_round import get_dataloader

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-GPTJForCausalLM", trust_remote_code=True
        )
        dataloader = get_dataloader(
            tokenizer, seqlen=10, seed=42, train_bs=8, dataset_split="train", dataset_name="NeelNanda/pile-10k"
        )
        fp32_model = copy.deepcopy(self.gptj)

        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,  # -1 (per-channel)
                        "scheme": "sym",
                        "algorithm": "AUTOROUND",
                    },
                },
                ".*lm_head": {  # re.match
                    "weight": {"dtype": "fp32"},
                },
            },
        )

        input = torch.ones([1, 512], dtype=torch.long)
        fp32_model = copy.deepcopy(self.gptj)
        out1 = fp32_model(input)
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=dataloader,
        )
        out2 = q_model.model(input)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-01))
        q_model.save("./test")
        print(q_model.autoround_config)


if __name__ == "__main__":
    unittest.main()
