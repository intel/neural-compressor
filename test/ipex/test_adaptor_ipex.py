import neural_compressor.adaptor.pytorch as nc_torch
import os
import shutil
import torch
import unittest
from neural_compressor.experimental import common
from packaging.version import Version

try:
    import intel_extension_for_pytorch as ipex
    TEST_IPEX = True
except:
    TEST_IPEX = False

torch.manual_seed(9527)
assert TEST_IPEX, "Please install intel extension for pytorch"
# get torch and IPEX version
PT_VERSION = nc_torch.get_torch_version()


def build_ipex_yaml():
    fake_yaml = '''
        model:
          name: imagenet
          framework: pytorch_ipex

        evaluation:
          accuracy:
            metric:
              topk: 1
          performance:
            warmup: 5
            iteration: 10

        tuning:
          accuracy_criterion:
            relative:  0.01
          exit_policy:
            timeout: 300
          random_seed: 9527
          workspace:
            path: saved
        '''
    with open('ipex_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_yaml)


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


@unittest.skipIf(PT_VERSION >= Version("1.12.0-rc1") or PT_VERSION < Version("1.10.0-rc1"),
                 "Please use Intel extension for Pytorch version 1.10 or 1.11")
class TestPytorchIPEX_1_10_Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_ipex_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('ipex_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_tuning_ipex(self):
        from neural_compressor.experimental import Quantization
        model = M()
        quantizer = Quantization('ipex_yaml.yaml')
        quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = True
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        quantizer.model = model
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        nc_model = quantizer.fit()
        nc_model.save('./saved')
        ipex_conf = ipex.quantization.QuantConf(configure_file="./saved/best_configure.json", )
        q_model = ipex.quantization.convert(model, ipex_conf, torch.randn(1, 3, 224, 224))
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark('ipex_yaml.yaml')
        evaluator.model = q_model
        evaluator.b_dataloader = common.DataLoader(dataset)
        evaluator.fit('accuracy')


@unittest.skipIf(PT_VERSION < Version("1.12.0-rc1"),
                 "Please use Intel extension for Pytorch version higher or equal to 1.12")
class TestPytorchIPEX_1_12_Adaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_ipex_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('ipex_yaml.yaml')
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('runs', ignore_errors=True)

    def test_tuning_ipex(self):
        from neural_compressor.experimental import Quantization
        model = M()
        quantizer = Quantization('ipex_yaml.yaml')
        quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = True
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        quantizer.model = model
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        nc_model = quantizer.fit()
        nc_model.save('./saved')
        qconfig = ipex.quantization.default_static_qconfig
        prepared_model = ipex.quantization.prepare(model,
                                                   qconfig,
                                                   example_inputs=torch.randn(1, 3, 224, 224),
                                                   inplace=False)
        prepared_model.load_qconf_summary(qconf_summary="./saved/best_configure.json")
        convert_model = ipex.quantization.convert(prepared_model)
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark('ipex_yaml.yaml')
        evaluator.model = convert_model
        evaluator.b_dataloader = common.DataLoader(dataset)
        evaluator.fit('accuracy')

    def test_tuning_ipex_for_ipex_autotune_func(self):
        from neural_compressor.experimental import Quantization
        model = M()
        qconfig = ipex.quantization.default_static_qconfig
        prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=torch.ones(1, 3, 224, 224), inplace=False)
        quantizer = Quantization('ipex_yaml.yaml')
        quantizer.conf.usr_cfg.tuning.exit_policy['max_trials'] = 20
        dataset = quantizer.dataset('dummy', (100, 3, 224, 224), label=True)
        quantizer.model = prepared_model
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        nc_model = quantizer.fit()
        nc_model.save('./saved')
        qconfig = ipex.quantization.default_static_qconfig
        prepared_model = ipex.quantization.prepare(model, qconfig, example_inputs=torch.ones(1, 3, 224, 224), inplace=False)
        prepared_model.load_qconf_summary(qconf_summary="./saved/best_configure.json")
        convert_model = ipex.quantization.convert(prepared_model)
        from neural_compressor.experimental import Benchmark
        evaluator = Benchmark('ipex_yaml.yaml')
        evaluator.model = convert_model
        evaluator.b_dataloader = common.DataLoader(dataset)
        evaluator.fit('accuracy')


if __name__ == "__main__":
    unittest.main()
