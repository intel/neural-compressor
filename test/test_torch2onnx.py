import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from torch.quantization import QuantStub, DeQuantStub
import unittest
import os
import onnx
from neural_compressor.adaptor.pytorch import PyTorchVersionMode
import neural_compressor.adaptor.pytorch as nc_torch
from neural_compressor.experimental import Quantization, common
import shutil
from distutils.version import StrictVersion


PT_VERSION = nc_torch.get_torch_version()
if PT_VERSION >= PyTorchVersionMode.PT18.value:
    FX_MODE = True
else:
    FX_MODE = False

ONNX111_VERSION = StrictVersion("1.11.0")
ONNX_VERSION = StrictVersion(onnx.__version__)
if ONNX_VERSION >= ONNX111_VERSION and PT_VERSION >= PyTorchVersionMode.PT110.value:
    BF16_MODE = True
else:
    BF16_MODE = False


fake_dyn_yaml = '''
    model:
      name: xxx
      framework: pytorch
    quantization:
      approach: post_training_dynamic_quant

    tuning:
      accuracy_criterion:
        relative: 0.01
      exit_policy:
        max_trials: 100
    '''


fake_ptq_yaml = '''
    model:
      name: xxx
      framework: pytorch
    quantization:
      approach: post_training_static_quant

    tuning:
      accuracy_criterion:
        relative: 0.01
      exit_policy:
        max_trials: 100
    '''


def build_pytorch_yaml():
    with open('ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_ptq_yaml)

    with open('dynamic_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_dyn_yaml)


def build_pytorch_fx_yaml():
    fake_fx_ptq_yaml = fake_ptq_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_fx_ptq_yaml)

    fake_fx_dyn_yaml = fake_dyn_yaml.replace('pytorch', 'pytorch_fx')
    with open('fx_dynamic_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_fx_dyn_yaml)


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv = nn.Conv2d(3, 1, 1)
        self.linear = nn.Linear(224 * 224, 5)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = x.view(1, -1)
        x = self.linear(x)
        x = self.dequant(x)
        return x


class DynamicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
    def forward(self, x):
        if x is not None:
            x = self.conv(x)
        return x


class DynamicControlModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.linear = nn.Linear(224 * 224, 1)
        self.dyn = DynamicModel()

    def forward(self, x):
        x = self.conv(x)
        x = self.dyn(x)
        x = self.bn(x)
        x = x.view(1, -1)
        x = self.linear(x)
        return x


example_inputs = torch.randn([1, 3, 224, 224])


class TestPytorchAdaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_pytorch_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('ptq_yaml.yaml')
        os.remove('dynamic_yaml.yaml')
        shutil.rmtree('runs', ignore_errors=True)
        os.remove('fp32-model.onnx')
        os.remove('int8-model.onnx')
        os.remove('int8-model.tmp')

    @unittest.skipIf(not BF16_MODE, "Unsupport BF16 Mode with ONNX Version Below 1.11")
    def test_bf16_onnx(self):
        model = M()
        from neural_compressor.experimental.common import Model
        inc_model = Model(model)
        inc_model.export_to_bf16_onnx(
            save_path='bf16-model.onnx',
            example_inputs=example_inputs,
            opset_version=14,
            dynamic_axes={"input": {0: "batch_size"},
                        "output": {0: "batch_size"}},
            do_constant_folding=True,
        )
        os.remove('bf16-model.onnx')

    def test_eager_quant(self):
        model = M()
        from neural_compressor.experimental.common import Model
        inc_model = Model(model)
        fp32_jit_model = inc_model.export_to_jit(example_inputs)
        inc_model.export_to_fp32_onnx(
            save_path='fp32-model.onnx',
            example_inputs=example_inputs,
            opset_version=11,
            dynamic_axes={"input": {0: "batch_size"},
                        "output": {0: "batch_size"}},
            do_constant_folding=True,
        )
        for fake_yaml in ['dynamic_yaml.yaml', ]:
            model = M()
            quantizer = Quantization(fake_yaml)
            quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = True
            dataset = quantizer.dataset('dummy', (10, 3, 224, 224), label=True)
            quantizer.model = model
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            q_model = quantizer.fit()

            int8_jit_model = q_model.export_to_jit(example_inputs)
            # INC will keep fallbacked fp32 modules when exporting onnx model
            if 'ptq_yaml.yaml':
                calib_dataloader = quantizer.calib_dataloader
            else:
                calib_dataloader = None
            q_model.export_to_int8_onnx(
                save_path='int8-model.onnx',
                example_inputs=example_inputs,
                opset_version=11,
                dynamic_axes={"input": {0: "batch_size"},
                            "output": {0: "batch_size"}},
                do_constant_folding=True,
                quant_format='QDQ',
                dtype='S8S8',
                fp32_model=model,
                calib_dataloader=calib_dataloader,
            )

    def test_input_tuple(self):
        from neural_compressor.adaptor.torch_utils.util import input2tuple
        input = {'input': [1,2,3,4], 'id': 0}
        output = input2tuple(input)
        self.assertTrue(type(output)==tuple)
        input = [1,2,3,4]
        output = input2tuple(input)
        self.assertTrue(type(output)==tuple)


@unittest.skipIf(not FX_MODE, "Unsupport Fx Mode with PyTorch Version Below 1.8")
class TestPytorchFXAdaptor(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_pytorch_fx_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fx_ptq_yaml.yaml')
        os.remove('fx_dynamic_yaml.yaml')
        shutil.rmtree('runs', ignore_errors=True)
        os.remove('int8-model.onnx')
        os.remove('int8-model.tmp')

    def test_fx_quant(self):
        for fake_yaml in ['fx_dynamic_yaml.yaml', 'fx_ptq_yaml.yaml']:
            model = DynamicControlModel()
            # run fx_quant in neural_compressor and save the quantized GraphModule
            quantizer = Quantization(fake_yaml)
            quantizer.conf.usr_cfg.tuning.exit_policy['performance_only'] = True
            dataset = quantizer.dataset('dummy', (10, 3, 224, 224), label=True)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.model = common.Model(model)
            q_model = quantizer.fit()

            int8_jit_model = q_model.export_to_jit(example_inputs)
            # INC will keep fallbacked fp32 modules when exporting onnx model
            if 'ptq_yaml.yaml':
                calib_dataloader = quantizer.calib_dataloader
            else:
                calib_dataloader = None
            q_model.export_to_int8_onnx(
                save_path='int8-model.onnx',
                example_inputs=example_inputs,
                opset_version=11,
                dynamic_axes={"input": {0: "batch_size"},
                            "output": {0: "batch_size"}},
                do_constant_folding=True,
                quant_format='QLinear',
                dtype='U8S8',
                fp32_model=model,
                calib_dataloader=calib_dataloader,
            )

if __name__ == "__main__":
    unittest.main()
