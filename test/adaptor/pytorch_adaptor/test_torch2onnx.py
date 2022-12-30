import copy
import numpy as np
import onnx
import os
import shutil
import torch
import torch.nn as nn
import unittest
import neural_compressor.adaptor.pytorch as nc_torch
from neural_compressor import quantization
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.experimental.data.datasets.dataset import Datasets
from packaging.version import Version
from torch.quantization import QuantStub, DeQuantStub


PT_VERSION = nc_torch.get_torch_version().release
if PT_VERSION >= Version("1.8.0").release:
    FX_MODE = True
else:
    FX_MODE = False

ONNX111_VERSION = Version("1.11.0")
ONNX_VERSION = Version(onnx.__version__)
if ONNX_VERSION >= ONNX111_VERSION and PT_VERSION >= Version("1.10.0").release:
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

def train_func(compression_manager, model, dataloader=None):
    compression_manager.callbacks.on_train_begin(dataloader)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    # switch to evaluate mode
    model.train()
    input = torch.randn(1, 3, 224, 224)
    # compute output
    output = model(input)
    loss = output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    compression_manager.callbacks.on_train_end()
    return model


def build_pytorch_yaml():
    with open('ptq_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_ptq_yaml)

    with open('dynamic_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_dyn_yaml)

    fake_qat_yaml = fake_ptq_yaml.replace(
        'post_training_static_quant',
        'quant_aware_training',
    )
    with open('qat_yaml.yaml', 'w', encoding="utf-8") as f:
        f.write(fake_qat_yaml)


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
        os.remove('qat_yaml.yaml')
        shutil.rmtree('runs', ignore_errors=True)

    @unittest.skipIf(not BF16_MODE, "Unsupport BF16 Mode with ONNX Version Below 1.11")
    def test_onnx(self):
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
        inc_model.export_to_fp32_onnx(
            save_path='fp32-model.onnx',
            example_inputs=example_inputs,
            opset_version=11,
            dynamic_axes={"input": {0: "batch_size"},
                        "output": {0: "batch_size"}},
            do_constant_folding=True,
        )
        os.remove('bf16-model.onnx')
        os.remove('fp32-model.onnx')

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

    def test_fx_quant(self):
        for approach in ['dynamic', 'static']:
            model = DynamicControlModel()
            # run fx_quant in neural_compressor and save the quantized GraphModule
            conf = PostTrainingQuantConfig(approach=approach)
            dataset = Datasets("pytorch")['dummy']((100, 3, 224, 224))
            dataloader = torch.utils.data.DataLoader(dataset)
            q_model = quantization.fit(model,
                                       conf,
                                       calib_dataloader=dataloader,
                                       eval_dataloader=dataloader
                                       )

            int8_jit_model = q_model.export_to_jit(example_inputs)
            # INC will keep fallbacked fp32 modules when exporting onnx model
            if approach == 'static':
                calib_dataloader = dataloader
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
                calib_dataloader=calib_dataloader
            )

if __name__ == "__main__":
    unittest.main()
