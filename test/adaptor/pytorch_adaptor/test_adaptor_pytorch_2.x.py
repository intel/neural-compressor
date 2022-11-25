import copy
import neural_compressor.adaptor.pytorch as nc_torch
import numpy as np
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import unittest
import os
from neural_compressor import PostTrainingQuantConfig, QuantizationAwareTrainingConfig
from neural_compressor.config import set_tensorboard, set_workspace
from neural_compressor.data import DATASETS, DATALOADERS
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.model import MODELS
from neural_compressor.experimental import Quantization, common
from neural_compressor.experimental.data.datasets.dataset import DATASETS
from neural_compressor import quantization
from neural_compressor.training import prepare_compression
from neural_compressor.utils.pytorch import load
from neural_compressor.utils.utility import recover
from neural_compressor.utils.utility import LazyImport
from torch.quantization import QuantStub, DeQuantStub
from packaging.version import Version


# improve lazy import UT coverage
resnet18 = LazyImport("torchvision.models.resnet18")
q_resnet18 = LazyImport("torchvision.models.quantization.resnet18")

PT_VERSION = nc_torch.get_torch_version().release
if PT_VERSION >= Version("1.8.0").release:
    FX_MODE = True
else:
    FX_MODE = False


dyn_op_name_list = {"decoder": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}}}

ptq_op_name_list = {
    "layer1.0.conv1": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    },
    "layer1.0.conv2": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    },
    "layer2.0.conv1": {
        "activation": {
            "dtype": ["uint8"],
            "algorithm": ["minmax"],
            "granularity": ["per_tensor"],
            "scheme": ["sym"]
        },
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["sym"]
        }
    },
    "layer3.0.conv1": {
        "activation": {
            "dtype": ["uint8"],
            "algorithm": ["kl"],
            "granularity": ["per_tensor"],
            "scheme": ["sym"]
        },
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["sym"]
        }
    },
    "layer1.0.add_relu": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    },
}

ptq_fx_op_name_list = {
    "layer1.0.conv1": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    },
    "layer1.0.conv2": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    },
    "layer2.0.conv1": {
        "activation": {
            "dtype": ["uint8"],
            "algorithm": ["minmax"],
            "granularity": ["per_tensor"],
            "scheme": ["sym"]
        },
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["sym"]
        }
    },
    "layer3.0.conv1": {
        "activation": {
            "dtype": ["uint8"],
            "algorithm": ["kl"],
            "granularity": ["per_tensor"],
            "scheme": ["sym"]
        },
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["sym"]
        }
    },
    "layer1.0.add_relu": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    },
    "conv.module": {
        "weight": {
            "dtype": ["fp32"]
        },
        "activation": {
            "dtype": ["fp32"]
        }
    },
    "default_qconfig": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    }
}

qat_op_name_list = {
    "layer1.0.conv1": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    },
    "layer1.0.conv2": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    },
    "layer2.0.conv1": {
        "activation": {
            "dtype": ["uint8"],
            "algorithm": ["minmax"],
            "granularity": ["per_tensor"],
            "scheme": ["sym"]
        },
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["sym"]
        }
    },
    "layer3.0.conv1": {
        "activation": {
            "dtype": ["uint8"],
            "algorithm": ["kl"],
            "granularity": ["per_tensor"],
            "scheme": ["sym"]
        },
        "weight": {
            "dtype": ["int8"],
            "algorithm": ["minmax"],
            "granularity": ["per_channel"],
            "scheme": ["sym"]
        }
    },
    "layer1.0.add_relu": {
        "activation": {
            "dtype": ["fp32"]
        },
        "weight": {
            "dtype": ["fp32"]
        }
    }
}




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


class FP32Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        times = x.size(1)
        if times == 1:
            return x + x
        return x


class DynamicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)
    def forward(self, x):
        if x is not None:
            x = self.conv(x)
        return x


class SubModel(torch.nn.Module):
    def __init__(self, bypass=True):
        super().__init__()
        self.quant = QuantStub()
        self.conv = nn.Conv2d(1, 1, 1)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.fp32 = FP32Model()
        self.norm = nn.LayerNorm([1, 224, 224])
        self.dequant = DeQuantStub()
        self.bypass = bypass

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.quant(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.dequant(x)
        if not self.bypass:
            x = self.fp32(x)
        x = self.norm(x)
        return x


class PartialQuantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv = nn.Conv2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.linear = nn.Linear(224 * 224, 1)
        self.dequant = DeQuantStub()
        self.sub = SubModel(bypass=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sub(x)
        x = self.quant(x)
        x = self.conv2(x)
        x = x.view(1, -1)
        x = self.linear(x)
        x = self.dequant(x)
        return x

class DynamicControlModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.linear = nn.Linear(224 * 224, 1)
        self.sub = SubModel()
        self.fp32 = FP32Model()
        self.dyn = DynamicModel()

    def forward(self, x):
        x = self.conv(x)
        x = self.dyn(x)
        x = self.bn(x)
        x = self.sub(x)
        x = self.fp32(x)
        x = x.view(1, -1)
        x = self.linear(x)
        return x


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken=10, ninp=512, nhid=256, nlayers=5, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        input = torch.ones((3, 10), dtype=torch.int32)
        h0 = torch.randn(2, 10, 256)
        c0 = torch.randn(2, 10, 256)
        hidden = (h0, c0)
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden


def eval_func(model):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        input = torch.randn(1, 3, 224, 224)
        # compute output
        output = model(input)
    return 0.0


def train_func(compression_manager, model, dataloader=None):
    compression_manager.callbacks.on_train_begin(dataloader=dataloader)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    # switch to evaluate mode
    model.train()
    input = torch.randn(1, 3, 224, 224)
    # compute output
    output = model(input)
    loss = output[0].mean() if isinstance(output, tuple) else output.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    compression_manager.callbacks.on_train_end()
    return model


def q_func(model):
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
    return model


class TestPytorchAdaptor(unittest.TestCase):
    model = q_resnet18()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_quantization_new_API(self):
        for fake_yaml in ["dynamic", "qat", "static"]:
            model = M()
            if fake_yaml == "qat":
                quant_conf = QuantizationAwareTrainingConfig(op_name_list=qat_op_name_list)
                compression_manager = prepare_compression(copy.deepcopy(model), quant_conf)
                q_model = train_func(compression_manager, compression_manager.model)
            else:
                dataset = DATASETS("pytorch")["dummy"]((100, 3, 224, 224))
                dataloader = DATALOADERS["pytorch"](dataset)
                if fake_yaml == "dynamic":
                    quant_conf = PostTrainingQuantConfig(approach="dynamic",
                                                    op_name_list=dyn_op_name_list)
                elif fake_yaml == "static":
                    quant_conf = PostTrainingQuantConfig(approach="static",
                                                    op_name_list=ptq_op_name_list)
                q_model = quantization.fit(
                    model,
                    quant_conf,
                    calib_dataloader=dataloader if fake_yaml == "static" else None)
            q_model.save("./saved")
            # Load configure and weights by neural_compressor.utils
            saved_model = load("./saved", model)
            shutil.rmtree("./saved", ignore_errors=True)

    def test_auto_quant(self):
        def eval_func(model):
            return 1

        model_origin = LSTMModel(
            ntoken = 10,
            ninp = 512,
            nhid = 256,
            nlayers = 2,
        )
        # run fx_quant in neural_compressor and save the quantized GraphModule
        quant_conf = PostTrainingQuantConfig(approach="auto")
        set_workspace("./saved")
        dataset = DATASETS("pytorch")["dummy"]((100, 3, 224, 224))
        dataloader = common.DataLoader(dataset)
        q_model = quantization.fit(model_origin,
                                   quant_conf,
                                   calib_dataloader=dataloader,
                                   eval_func=eval_func)
        q_model.save("./saved")
        model = common.Model(model_origin)
        model.workspace_path = "./saved"
        self.assertNotEqual(q_model, None)
        self.assertEqual(type(q_model._model.decoder),
                         type(model._model.decoder))
        shutil.rmtree("./saved", ignore_errors=True)

    def test_tensorboard(self):
        model = copy.deepcopy(self.model)
        model.eval().fuse_model()
        quant_conf = PostTrainingQuantConfig(approach="static",
                                        backend="pytorch")
        set_tensorboard(True)
        dataset = DATASETS("pytorch")["dummy"]((100, 3, 224, 224))
        dataloader = common.DataLoader(dataset)
        quantization.fit(
          model, quant_conf, calib_dataloader=dataloader, eval_func=eval_func
        )
        self.assertTrue(True if os.path.exists("runs/eval/baseline_acc0.0") else False)
        quantization.fit(model,
                   quant_conf,
                   calib_dataloader=dataloader,
                   eval_dataloader=dataloader)
        self.assertTrue(True if os.path.exists("runs/eval/baseline_acc0.0") else False)
        set_tensorboard(False)


@unittest.skipIf(not FX_MODE, "Unsupport Fx Mode with PyTorch Version Below 1.8")
class TestPytorchFXAdaptor(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_fx_quant(self):
        for fake_yaml in ["qat", "static"]:
            model_origin = resnet18()
            dataset = DATASETS("pytorch")["dummy"]((10, 3, 224, 224), label=True)
            dataloader = DATALOADERS["pytorch"](dataset)
            if fake_yaml == "qat":
                conf = QuantizationAwareTrainingConfig(
                  op_name_list=qat_op_name_list, backend="pytorch_fx"
                )
                compression_manager = prepare_compression(copy.deepcopy(model_origin), conf)
                q_model = train_func(compression_manager, compression_manager.model, dataloader)
            else:
                conf = PostTrainingQuantConfig(
                  op_name_list=ptq_fx_op_name_list, backend="pytorch_fx"
                )
                set_workspace("./saved")
                q_model = quantization.fit(model_origin,
                                           conf,
                                           calib_dataloader=dataloader,
                                           calib_func=eval_func)
            q_model.save("./saved")
            # Load configure and weights with neural_compressor.utils
            model_fx = load("./saved", model_origin,
                            **{"dataloader": torch.utils.data.DataLoader(dataset)})
            self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))

            if fake_yaml != "qat":
                # recover int8 model with only tune_cfg
                history_file = "./saved/history.snapshot"
                model_fx_recover = recover(model_origin, history_file, 0,
                                **{"dataloader": dataloader})
                self.assertEqual(model_fx.code, model_fx_recover.code)
            shutil.rmtree("./saved", ignore_errors=True)
        for fake_yaml in ["qat", "static"]:
            model_origin = M()
            # run fx_quant in neural_compressor and save the quantized GraphModule
            dataset = DATASETS("pytorch")["dummy"]((100, 3, 224, 224), label=True)
            dataloader = DATALOADERS["pytorch"](dataset)
            if fake_yaml == "qat":
                conf = QuantizationAwareTrainingConfig(
                  op_name_list=qat_op_name_list, backend="pytorch_fx"
                )
                compression_manager = prepare_compression(copy.deepcopy(model_origin), conf)
                q_model = train_func(compression_manager, compression_manager.model, dataloader)
                compression_manager.save("./saved")
            else:
                conf = PostTrainingQuantConfig(
                    op_name_list=ptq_fx_op_name_list, backend="pytorch_fx"
                )
                q_model = quantization.fit(model_origin,
                                           conf,
                                           calib_dataloader=dataloader)
                q_model.save("./saved")
            # Load configure and weights with neural_compressor.utils
            model_fx = load("./saved", model_origin,
                            **{"dataloader": torch.utils.data.DataLoader(dataset)})
            self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))
            shutil.rmtree("./saved", ignore_errors=True)

    @unittest.skipIf(PT_VERSION < Version("1.9.0").release,
      "Please use PyTroch 1.9 or higher version for dynamic quantization with pytorch_fx backend")
    def test_fx_dynamic_quant(self):
        origin_model = LSTMModel(
            ntoken = 10,
            ninp = 512,
            nhid = 256,
            nlayers = 5,
        )
        # run fx_quant in neural_compressor and save the quantized GraphModule
        origin_model.eval()
        conf = PostTrainingQuantConfig(approach="dynamic",
            op_name_list=ptq_fx_op_name_list, backend="pytorch_fx"
        )
        set_workspace("./saved")
        q_model = quantization.fit(origin_model, conf)
        q_model.save("./saved")

        # Load configure and weights by neural_compressor.utils
        model_fx = load("./saved", origin_model)
        self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))

        # Test the functionality of older model saving type
        state_dict = torch.load("./saved/best_model.pt")
        tune_cfg = state_dict.pop("best_configure")
        import yaml
        with open("./saved/best_configure.yaml", "w") as f:
            yaml.dump(tune_cfg, f, default_flow_style=False)
        torch.save(state_dict, "./saved/best_model_weights.pt")
        os.remove("./saved/best_model.pt")
        model_fx = load("./saved", origin_model)
        self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))

        # recover int8 model with only tune_cfg
        history_file = "./saved/history.snapshot"
        model_fx_recover = recover(origin_model, history_file, 0)
        self.assertEqual(model_fx.code, model_fx_recover.code)
        shutil.rmtree("./saved", ignore_errors=True)

    def test_default_dynamic_quant(self):
        def eval_func(model):
            return 1

        # Model Definition
        for fake_yaml in ["qat", "auto"]:
            model_origin = LSTMModel(
                ntoken = 10,
                ninp = 512,
                nhid = 256,
                nlayers = 2,
            )
            dataset = DATASETS("pytorch")["dummy"]((3, 10))
            dataloader = DATALOADERS["pytorch"](dataset)
            # run fx_quant in neural_compressor and save the quantized GraphModule
            if fake_yaml == "qat":
                conf = QuantizationAwareTrainingConfig(
                    op_name_list=qat_op_name_list, backend="pytorch_fx"
                )
                compression_manager = prepare_compression(copy.deepcopy(model_origin), conf)
                q_model = train_func(compression_manager, compression_manager.model, dataloader=dataloader)
                self.assertTrue("quantize" in str(type(q_model.model.encoder)))
                self.assertTrue("quantize" in str(type(q_model.model.rnn)))
            else:
                conf = PostTrainingQuantConfig(backend="pytorch_fx")
                q_model = quantization.fit(model_origin,
                                           conf,
                                           calib_dataloader=dataloader)
                self.assertTrue("quantize" in str(type(q_model.model.encoder)))
                self.assertTrue("quantize" in str(type(q_model.model.rnn)))

    def test_fx_sub_module_quant(self):
        for fake_yaml in ["qat", "static"]:
            model_origin = DynamicControlModel()
            dataset = DATASETS("pytorch")["dummy"]((1, 3, 224, 224))
            dataloader = DATALOADERS["pytorch"](dataset)
            # run fx_quant in neural_compressor and save the quantized GraphModule
            if fake_yaml == "qat":
                conf = QuantizationAwareTrainingConfig(
                    op_name_list=qat_op_name_list, backend="pytorch_fx"
                )
                compression_manager = prepare_compression(copy.deepcopy(model_origin), conf)
                q_model = train_func(compression_manager, compression_manager.model, dataloader)
            else:
                set_workspace("./saved")
                conf = PostTrainingQuantConfig(backend="pytorch_fx")
                q_model = quantization.fit(model_origin,
                                           conf,
                                           calib_dataloader=dataloader)
            q_model.save("./saved")
            # Load configure and weights with neural_compressor.utils
            model_fx = load("./saved/best_model.pt", model_origin,
                               **{"dataloader": torch.utils.data.DataLoader(dataset)
                              })
            self.assertTrue(isinstance(model_fx.sub, torch.fx.graph_module.GraphModule))

            if fake_yaml != "qat":
            # recover int8 model with only tune_cfg
                history_file = "./saved/history.snapshot"
                model_fx_recover = recover(model_origin, history_file, 0,
                                   **{"dataloader": torch.utils.data.DataLoader(dataset)
                                  })
                self.assertEqual(model_fx.sub.code, model_fx_recover.sub.code)
            shutil.rmtree("./saved", ignore_errors=True)

    @unittest.skipIf(PT_VERSION < Version("1.11.0").release,
      "Please use PyTroch 1.11 or higher version for mixed precision with pytorch_fx or pytorch backend")
    def test_mix_precision(self):
        model_origin = DynamicControlModel()
        # run fx_quant in neural_compressor and save the quantized GraphModule
        dataset = DATASETS("pytorch")["dummy"]((100, 3, 224, 224))
        dataloader = DATALOADERS["pytorch"](dataset)
        set_workspace=("./saved")
        conf = PostTrainingQuantConfig(op_name_list=ptq_fx_op_name_list, backend="pytorch_fx")
        q_model = quantization.fit(model_origin,
                                   conf,
                                   calib_dataloader=dataloader,
                                   calib_func = eval_func)
        tune_cfg = q_model.q_config
        tune_cfg["op"][("conv.module", "Conv2d")].clear()
        tune_cfg["op"][("conv.module", "Conv2d")] = \
            {"weight": {"dtype": "bf16"}, "activation": {"dtype": "bf16"}}
        tune_cfg["bf16_ops_list"].append(("conv.module", "Conv2d"))
        from neural_compressor.adaptor.torch_utils.bf16_convert import Convert
        q_model._model = Convert(q_model._model, tune_cfg)

        self.assertEqual(q_model._model.conv.module.module.weight.dtype, torch.bfloat16)
        self.assertEqual(q_model._model.conv.module.module.bias.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
