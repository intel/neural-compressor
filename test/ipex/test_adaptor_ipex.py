import neural_compressor.adaptor.pytorch as nc_torch
import os
import shutil
import torch
import unittest
from neural_compressor.experimental import common
from packaging.version import Version
from neural_compressor.utils.utility import LazyImport
from neural_compressor import config
from neural_compressor.utils.pytorch import load
torch_utils = LazyImport("neural_compressor.adaptor.torch_utils")

try:
    import intel_extension_for_pytorch as ipex
    TEST_IPEX = True
except:
    TEST_IPEX = False

torch.manual_seed(9527)
assert TEST_IPEX, "Please install intel extension for pytorch"
# get torch and IPEX version
PT_VERSION = nc_torch.get_torch_version().release

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


@unittest.skipIf(PT_VERSION >= Version("1.12.0").release or PT_VERSION < Version("1.10.0").release,
                 "Please use Intel extension for Pytorch version 1.10 or 1.11")
class TestPytorchIPEX_1_10_Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        config.quantization.backend = 'ipex'
        config.quantization.approach = 'post_training_static_quant'
        config.quantization.use_bf16 = False

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
        qconfig = ipex.quantization.default_static_qconfig
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
        qconfig = ipex.quantization.default_static_qconfig
        prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=torch.ones(1, 3, 224, 224), inplace=False)
        copy_model = torch_utils.util.auto_copy(prepared_model)
        self.assertTrue(isinstance(copy_model, torch.nn.Module))
    
    def test_bf16(self):
        from neural_compressor.experimental import Quantization
        model = M()
        qconfig = ipex.quantization.default_static_qconfig
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

if __name__ == "__main__":
    unittest.main()
