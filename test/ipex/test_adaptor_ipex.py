import neural_compressor.adaptor.pytorch as nc_torch
import os
import shutil
import torch
import torch.utils.data as data
import unittest
from neural_compressor import set_workspace
from neural_compressor.experimental import common
from neural_compressor.utils.utility import LazyImport
from neural_compressor.conf.pythonic_config import config
from neural_compressor.utils.pytorch import load
from packaging.version import Version
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from neural_compressor import mix_precision
from neural_compressor.utils.utility import LazyImport
from neural_compressor.config import MixedPrecisionConfig

torch_utils = LazyImport("neural_compressor.adaptor.torch_utils")

os.environ["WANDB_DISABLED"] = "true"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "true"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

try:
    import intel_extension_for_pytorch as ipex
    TEST_IPEX = True
    IPEX_VERSION = Version(ipex.__version__)
except:
    TEST_IPEX = False

torch.manual_seed(9527)
assert TEST_IPEX, "Please install intel extension for pytorch"
# get torch and IPEX version
PT_VERSION = nc_torch.get_torch_version().release

class DummyDataloader(data.DataLoader):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.sequence_a = "intel-extension-for-transformers is based in SH"
        self.sequence_b = "Where is intel-extension-for-transformers based? NYC or SH"
        self.encoded_dict = self.tokenizer(self.sequence_a, self.sequence_b, return_tensors="pt")
        self.batch_size = 1

    def __len__(self):
        return 10

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        if index < 10:
            return self.encoded_dict
        
    def __iter__(self):
        for _ in range(10):
            yield self.encoded_dict

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)
        self.linear = torch.nn.Linear(224 * 224, 5)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(1, -1)
        x = self.linear(x)
        return x


def calib_func(model):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        input = torch.randn(1, 3, 224, 224)
        # compute output
        output = model(input)


class Dataloader:
    def __init__(self) -> None:
        self.batch_size=1
    def __iter__(self):
        yield torch.randn(1, 3, 224, 224)


@unittest.skipIf(PT_VERSION >= Version("1.12.0").release or PT_VERSION < Version("1.10.0").release,
                 "Please use Intel extension for Pytorch version 1.10 or 1.11")
class TestPytorchIPEX_1_10_Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config.quantization.backend = 'ipex'
        config.quantization.approach = 'post_training_static_quant'
        config.quantization.use_bf16 = False
        set_workspace("./saved")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_tuning_ipex(self):
        from neural_compressor.experimental import Quantization
        model = M()
        quantizer = Quantization(config)
        quantizer.model = model
        quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = True
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        dataloader = torch.utils.data.DataLoader(dataset)
        quantizer.calib_dataloader = dataloader
        quantizer.eval_dataloader = dataloader
        nc_model = quantizer.fit()
        nc_model.save('./saved')
        q_model = load('./saved', model, dataloader=dataloader)
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark(config)
        evaluator.model = q_model
        evaluator.b_dataloader = dataloader
        evaluator.fit('accuracy')

@unittest.skipIf(PT_VERSION < Version("1.12.0").release,
                 "Please use Intel extension for Pytorch version higher or equal to 1.12")
class TestPytorchIPEX_1_12_Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config.quantization.backend = 'ipex'
        config.quantization.accuracy_criterion.tolerable_loss = 0.0001
        config.quantization.accuracy_criterion.higher_is_better = False
        config.quantization.approach = 'post_training_static_quant'
        config.quantization.use_bf16 = False
        set_workspace("./saved")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_tuning_ipex(self):
        from neural_compressor.experimental import Quantization
        model = M()
        quantizer = Quantization(config)
        quantizer.model = model
        quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = False
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        dataloader = torch.utils.data.DataLoader(dataset)
        quantizer.calib_dataloader = dataloader
        quantizer.calib_func = calib_func
        quantizer.eval_dataloader = dataloader
        nc_model = quantizer.fit()
        sparsity = nc_model.report_sparsity()
        self.assertTrue(sparsity[-1] >= 0.0)
        nc_model.save('./saved')
        q_model = load('./saved', model, dataloader=dataloader)
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark(config)
        evaluator.model = q_model
        evaluator.b_dataloader = dataloader
        evaluator.fit('accuracy')

    def test_tuning_ipex_for_ipex_autotune_func(self):
        from neural_compressor.experimental import Quantization
        model = M()
        if PT_VERSION < Version("2.1").release:
            qconfig = ipex.quantization.default_static_qconfig
        else:
            qconfig = ipex.quantization.default_static_qconfig_mapping
        prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=torch.ones(1, 3, 224, 224), inplace=False)
        quantizer = Quantization(config)
        quantizer.model = prepared_model
        quantizer.conf.usr_cfg.tuning.exit_policy['max_trials'] = 5
        quantizer.conf.usr_cfg.tuning.exit_policy['timeout'] = 100
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        dataloader = torch.utils.data.DataLoader(dataset)
        quantizer.calib_dataloader = dataloader
        quantizer.eval_dataloader = dataloader
        nc_model = quantizer.fit()

    def test_copy_prepared_model(self):
        model = M()
        if PT_VERSION < Version("2.1").release:
            qconfig = ipex.quantization.default_static_qconfig
        else:
            qconfig = ipex.quantization.default_static_qconfig_mapping
        prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=torch.ones(1, 3, 224, 224), inplace=False)
        copy_model = torch_utils.util.auto_copy(prepared_model)
        self.assertTrue(isinstance(copy_model, torch.nn.Module))
    
    def test_bf16(self):
        from neural_compressor.experimental import Quantization
        model = M()
        if PT_VERSION < Version("2.1").release:
            qconfig = ipex.quantization.default_static_qconfig
        else:
            qconfig = ipex.quantization.default_static_qconfig_mapping
        prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=torch.ones(1, 3, 224, 224), inplace=False)
        config.quantization.use_bf16 = True
        config.quantization.performance_only = True
        quantizer = Quantization(config)
        quantizer.model = model
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        dataloader = torch.utils.data.DataLoader(dataset)
        quantizer.calib_dataloader = dataloader
        quantizer.eval_dataloader = dataloader
        nc_model = quantizer.fit()

    def test_example_inputs(self):
        from neural_compressor.experimental import Quantization
        model = M()
        config.quantization.example_inputs = torch.randn([1, 3, 224, 224])
        quantizer = Quantization(config)
        quantizer.model = model
        quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = False
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        dataloader = torch.utils.data.DataLoader(dataset)
        quantizer.calib_dataloader = dataloader
        nc_model = quantizer.fit()

    def test_new_API(self):
        model = M()
        from neural_compressor import PostTrainingQuantConfig, quantization
        op_type_dict = {
            "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
            "linear": {
                "weight": {
                    "dtype": ["int8"],
                    "scheme": ["sym"],
                    "granularity": ["per_channel"],
                    "algorithm": ["minmax"],
                },
                "activation": {
                    "dtype": ["int8"],
                    "scheme": ["sym"],
                    "granularity": ["per_tensor"],
                    "algorithm": ["kl"],
                },
            },
        }

        conf = PostTrainingQuantConfig(
            backend="ipex",
            op_type_dict=op_type_dict,
        )
        calib_dataloader = Dataloader()
        q_model = quantization.fit(
            model,
            conf,
            calib_dataloader=calib_dataloader,
        )
        q_model.save('./saved')

    def test_fallback_fused_op_type(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 1)
                self.linear = torch.nn.Linear(224 * 224, 5)

            def forward(self, a):
                x = self.conv(a)
                x += x
                x = x.view(1, -1)
                x = self.linear(x)
                return x
        
        model = M()
        from neural_compressor import PostTrainingQuantConfig, quantization
        op_type_dict = {
            "Conv2d&add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
        }

        conf = PostTrainingQuantConfig(
            backend="ipex",
            op_type_dict=op_type_dict,
        )
        calib_dataloader = Dataloader()
        q_model = quantization.fit(
            model,
            conf,
            calib_dataloader=calib_dataloader,
        )
        
    def test_tune_add(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 1)
                self.linear = torch.nn.Linear(224 * 224, 5)

            def forward(self, a):
                x = self.conv(a)
                x = x.view(1, -1)
                x += x
                x = self.linear(x)
                return x
        
        model = M()
        from neural_compressor import PostTrainingQuantConfig, quantization
        
        
        acc_lst = [1, 0.8, 1.1, 1.2]
        def fake_eval(model):
            res = acc_lst.pop(0)
            return res
            

        conf = PostTrainingQuantConfig(
            backend="ipex",
            quant_level=0
            )
        calib_dataloader = Dataloader()
        q_model = quantization.fit(
            model,
            conf,
            calib_dataloader=calib_dataloader,
            eval_func=fake_eval
        )

    def test_tune_add_with_recipe(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 1)
                self.linear = torch.nn.Linear(224 * 224, 5)

            def forward(self, a):
                x = self.conv(a)
                x = x.view(1, -1)
                x += x
                x = self.linear(x)
                return x
        
        model = M()
        from neural_compressor import PostTrainingQuantConfig, quantization
        
        
        acc_lst = [1, 0.8, 1.1, 1.2]
        def fake_eval(model):
            res = acc_lst.pop(0)
            return res
            

        conf = PostTrainingQuantConfig(
            backend="ipex",
            quant_level=0,
            recipes={'smooth_quant': True,
                     'smooth_quant_args': { 'alpha': 0.5}
                     }
            )
        calib_dataloader = Dataloader()
        q_model = quantization.fit(
            model,
            conf,
            calib_dataloader=calib_dataloader,
            eval_func=fake_eval
        )

    @unittest.skipIf(IPEX_VERSION.release < Version("2.1.0").release,
                 "Please use Intel extension for Pytorch version higher or equal to 2.1.0")
    def test_dict_inputs_for_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        dummy_dataloader = DummyDataloader()
        from neural_compressor import PostTrainingQuantConfig, quantization

        conf = PostTrainingQuantConfig(
            backend="ipex",
        )
        q_model = quantization.fit(
            model,
            conf,
            calib_dataloader=dummy_dataloader,
        )
        q_model.save('./saved')

    @unittest.skipIf(IPEX_VERSION.release < Version("2.1.0").release,
                 "Please use Intel extension for Pytorch version higher or equal to 2.1.0")
    def test_dict_inputs_for_model_calib_func(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        example_inputs = DummyDataloader()[0]
        from neural_compressor import PostTrainingQuantConfig, quantization

        def calib_func(p_model):
            p_model(**example_inputs)

        conf = PostTrainingQuantConfig(
            backend="ipex",
            example_inputs=example_inputs
        )
        q_model = quantization.fit(
            model,
            conf,
            calib_func=calib_func,
        )
        q_model.save('./saved')

class TestMixedPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.environ['FORCE_FP16'] = '1'
        os.environ['FORCE_BF16'] = '1'
        self.pt_model = M()

    @unittest.skipIf(IPEX_VERSION.release < Version("1.11.0").release,
      "Please use PyTroch 1.11 or higher version for mixed precision.")
    def test_mixed_precision_with_eval_func_ipex(self):
        torch = LazyImport("torch")
        def eval(model):
            return 0.5

        conf = MixedPrecisionConfig(backend="ipex", example_inputs=torch.randn(1, 3, 224, 224))
        output_model = mix_precision.fit(
            self.pt_model,
            conf,
            eval_func=eval,
        )
        self.assertTrue(isinstance(output_model._model, torch.jit.ScriptModule))
if __name__ == "__main__":
    unittest.main()
