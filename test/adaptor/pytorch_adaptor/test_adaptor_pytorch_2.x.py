import copy
import os
import shutil
import unittest

import torch
import torch.nn as nn
from packaging.version import Version
from torch.quantization import DeQuantStub, QuantStub

import neural_compressor.adaptor.pytorch as nc_torch
from neural_compressor import (
    Metric,
    PostTrainingQuantConfig,
    QuantizationAwareTrainingConfig,
    quantization,
    set_workspace,
)
from neural_compressor.data import DATALOADERS, DataLoader, Datasets
from neural_compressor.training import fit, prepare_compression
from neural_compressor.utils.pytorch import load
from neural_compressor.utils.utility import LazyImport, recover

# improve lazy import UT coverage
resnet18 = LazyImport("torchvision.models.resnet18")

PT_VERSION = nc_torch.get_torch_version().release
if PT_VERSION >= Version("1.8.0").release:
    FX_MODE = True
else:
    FX_MODE = False


ptq_fx_op_name_list = {
    "layer1.0.conv1": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
    "layer1.0.conv2": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
    "layer2.0.conv1": {
        "activation": {"dtype": ["uint8"], "algorithm": ["minmax"], "granularity": ["per_tensor"], "scheme": ["sym"]},
        "weight": {"dtype": ["int8"], "algorithm": ["minmax"], "granularity": ["per_channel"], "scheme": ["sym"]},
    },
    "layer3.0.conv1": {
        "activation": {"dtype": ["uint8"], "algorithm": ["kl"], "granularity": ["per_tensor"], "scheme": ["sym"]},
        "weight": {"dtype": ["int8"], "algorithm": ["minmax"], "granularity": ["per_channel"], "scheme": ["sym"]},
    },
    "layer1.0.add_relu": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
    "conv.module": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
    "default_qconfig": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
}

qat_op_name_list = {
    "layer1.0.conv1": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
    "layer1.0.conv2": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
    "layer2.0.conv1": {
        "activation": {"dtype": ["uint8"], "algorithm": ["minmax"], "granularity": ["per_tensor"], "scheme": ["sym"]},
        "weight": {"dtype": ["int8"], "algorithm": ["minmax"], "granularity": ["per_channel"], "scheme": ["sym"]},
    },
    "layer3.0.conv1": {
        "activation": {"dtype": ["uint8"], "algorithm": ["kl"], "granularity": ["per_tensor"], "scheme": ["sym"]},
        "weight": {"dtype": ["int8"], "algorithm": ["minmax"], "granularity": ["per_channel"], "scheme": ["sym"]},
    },
    "layer1.0.add_relu": {"activation": {"dtype": ["fp32"]}, "weight": {"dtype": ["fp32"]}},
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


def train_func(model):
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
    return model


@unittest.skipIf(not FX_MODE, "Unsupported Fx Mode with PyTorch Version Below 1.8")
class TestPytorchFXAdaptor(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_fx_quant(self):
        for approach in ["qat", "static"]:
            model_origin = resnet18()
            dataset = Datasets("pytorch")["dummy"]((10, 3, 224, 224), label=True)
            dataloader = DATALOADERS["pytorch"](dataset)
            if approach == "qat":
                model = copy.deepcopy(model_origin)
                conf = QuantizationAwareTrainingConfig(op_name_dict=qat_op_name_list)
                compression_manager = prepare_compression(model, conf)
                compression_manager.callbacks.on_train_begin()
                model = compression_manager.model
                q_model = train_func(model)
                compression_manager.callbacks.on_train_end()
                compression_manager.save("./saved")
            else:
                conf = PostTrainingQuantConfig(op_name_dict=ptq_fx_op_name_list)
                conf.example_inputs = torch.randn([1, 3, 224, 224])
                set_workspace("./saved")
                q_model = quantization.fit(model_origin, conf, calib_dataloader=dataloader, eval_func=eval_func)
                q_model.save("./saved")
            # Load configure and weights with neural_compressor.utils
            model_fx = load("./saved", model_origin)
            self.assertTrue("quantize" in str(type(q_model.model.fc)))
            self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))

            shutil.rmtree("./saved", ignore_errors=True)

        for approach in ["qat", "static"]:
            model_origin = M()
            # run fx_quant in neural_compressor and save the quantized GraphModule
            dataset = Datasets("pytorch")["dummy"]((100, 3, 224, 224), label=True)
            dataloader = DATALOADERS["pytorch"](dataset)
            if approach == "qat":
                model = copy.deepcopy(model_origin)
                conf = QuantizationAwareTrainingConfig(op_name_dict=qat_op_name_list)
                compression_manager = prepare_compression(model, conf)
                q_model = fit(compression_manager=compression_manager, train_func=train_func, eval_func=eval_func)
                compression_manager.save("./saved")
            else:
                conf = PostTrainingQuantConfig(op_name_dict=ptq_fx_op_name_list)
                q_model = quantization.fit(model_origin, conf, calib_dataloader=dataloader)
                q_model.save("./saved")
            # Load configure and weights with neural_compressor.utils
            model_fx = load("./saved", model_origin)
            self.assertTrue("quantize" in str(type(model_fx.conv)))
            self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))
            shutil.rmtree("./saved", ignore_errors=True)

    def test_quantize_with_metric(self):
        model_origin = resnet18()
        dataset = Datasets("pytorch")["dummy"]((1, 3, 224, 224))
        dataloader = DATALOADERS["pytorch"](dataset)
        # run fx_quant in neural_compressor and save the quantized GraphModule
        conf = PostTrainingQuantConfig()
        q_model = quantization.fit(
            model_origin,
            conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_metric=Metric(name="topk", k=1),
        )
        self.assertTrue("quantize" in str(type(q_model.model.fc)))

    def test_quantize_with_calib_func(self):
        model_origin = resnet18()
        # run fx_quant in neural_compressor and save the quantized GraphModule
        conf = PostTrainingQuantConfig()
        q_model = quantization.fit(model_origin, conf, calib_func=eval_func, eval_func=eval_func)
        self.assertTrue("quantize" in str(type(q_model.model.fc)))

    @unittest.skipIf(
        PT_VERSION < Version("1.9.0").release,
        "Please use PyTroch 1.9 or higher version for dynamic quantization with pytorch_fx backend",
    )
    def test_fx_dynamic_quant(self):
        origin_model = LSTMModel(
            ntoken=10,
            ninp=512,
            nhid=256,
            nlayers=5,
        )
        # run fx_quant in neural_compressor and save the quantized GraphModule
        origin_model.eval()
        conf = PostTrainingQuantConfig(approach="dynamic", op_name_dict=ptq_fx_op_name_list)
        set_workspace("./saved")
        q_model = quantization.fit(copy.deepcopy(origin_model), conf)
        q_model.save("./saved")

        # Load configure and weights by neural_compressor.utils
        model_fx = load("./saved", copy.deepcopy(origin_model))
        self.assertTrue(isinstance(model_fx, torch.fx.graph_module.GraphModule))

        # Test the functionality of older model saving type
        state_dict = torch.load("./saved/best_model.pt")
        tune_cfg = state_dict.pop("best_configure")
        import yaml

        with open("./saved/best_configure.yaml", "w") as f:
            yaml.dump(tune_cfg, f, default_flow_style=False)
        torch.save(state_dict, "./saved/best_model_weights.pt")
        os.remove("./saved/best_model.pt")
        model_fx = load("./saved", copy.deepcopy(origin_model))
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
        for approach in ["qat", "auto"]:
            model_origin = LSTMModel(
                ntoken=10,
                ninp=512,
                nhid=256,
                nlayers=2,
            )
            dataset = Datasets("pytorch")["dummy"]((3, 10))
            dataloader = DATALOADERS["pytorch"](dataset)
            # run fx_quant in neural_compressor and save the quantized GraphModule
            if approach == "qat":
                model = copy.deepcopy(model_origin)
                conf = QuantizationAwareTrainingConfig(op_name_dict=qat_op_name_list)
                compression_manager = prepare_compression(model, conf)
                compression_manager.callbacks.on_train_begin()
                model = compression_manager.model.model
                train_func(model)
                compression_manager.callbacks.on_train_end()
                self.assertTrue("quantize" in str(type(model.encoder)))
                self.assertTrue("quantize" in str(type(model.rnn)))
            else:
                conf = PostTrainingQuantConfig(approach="auto")
                q_model = quantization.fit(model_origin, conf, calib_dataloader=dataloader)
                self.assertTrue("quantize" in str(type(q_model.model.encoder)))
                self.assertTrue("quantize" in str(type(q_model.model.rnn)))

    def test_fx_sub_module_quant(self):
        for approach in ["qat", "static"]:
            model_origin = DynamicControlModel()
            dataset = Datasets("pytorch")["dummy"]((1, 3, 224, 224))
            dataloader = DATALOADERS["pytorch"](dataset)
            # run fx_quant in neural_compressor and save the quantized GraphModule
            if approach == "qat":
                model = copy.deepcopy(model_origin)
                conf = QuantizationAwareTrainingConfig(op_name_dict=qat_op_name_list)
                compression_manager = prepare_compression(model, conf)
                compression_manager.callbacks.on_train_begin()
                model = compression_manager.model
                q_model = train_func(model)
                compression_manager.callbacks.on_train_end()
                compression_manager.save("./saved")
            else:
                set_workspace("./saved")
                conf = PostTrainingQuantConfig()
                q_model = quantization.fit(model_origin, conf, calib_dataloader=dataloader)
                q_model.save("./saved")
            # Load configure and weights with neural_compressor.utils
            model_fx = load(
                "./saved/best_model.pt", model_origin, **{"dataloader": torch.utils.data.DataLoader(dataset)}
            )
            self.assertTrue(isinstance(model_fx.sub, torch.fx.graph_module.GraphModule))

            if approach != "qat":
                # recover int8 model with only tune_cfg
                history_file = "./saved/history.snapshot"
                model_fx_recover = recover(
                    model_origin, history_file, 0, **{"dataloader": torch.utils.data.DataLoader(dataset)}
                )
                self.assertEqual(model_fx.sub.code, model_fx_recover.sub.code)
            shutil.rmtree("./saved", ignore_errors=True)

    @unittest.skipIf(
        PT_VERSION < Version("1.11.0").release,
        "Please use PyTroch 1.11 or higher version for mixed precision with pytorch_fx or pytorch backend",
    )
    def test_mix_precision(self):
        model_origin = DynamicControlModel()
        # run fx_quant in neural_compressor and save the quantized GraphModule
        dataset = Datasets("pytorch")["dummy"]((100, 3, 224, 224))
        dataloader = DataLoader("pytorch", dataset)
        set_workspace("./saved")
        conf = PostTrainingQuantConfig(op_name_dict=ptq_fx_op_name_list)
        q_model = quantization.fit(model_origin, conf, calib_dataloader=dataloader, calib_func=eval_func)
        tune_cfg = q_model.q_config
        tune_cfg["op"][("conv.module", "Conv2d")].clear()
        tune_cfg["op"][("conv.module", "Conv2d")] = {"weight": {"dtype": "bf16"}, "activation": {"dtype": "bf16"}}
        tune_cfg["bf16_ops_list"].append(("conv.module", "Conv2d"))
        from neural_compressor.adaptor.torch_utils.bf16_convert import Convert

        q_model._model = Convert(q_model._model, tune_cfg)

        self.assertEqual(q_model._model.conv.module.module.weight.dtype, torch.bfloat16)
        self.assertEqual(q_model._model.conv.module.module.bias.dtype, torch.bfloat16)

    def test_hawq_metric(self):
        # Test for hawq metric
        import torchvision

        from neural_compressor.adaptor.torch_utils.hawq_metric import hawq_top
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.model.torch_model import PyTorchFXModel
        from neural_compressor.quantization import fit

        ori_model = torchvision.models.resnet18()
        pt_model = PyTorchFXModel(ori_model)
        dataset = Datasets("pytorch")["dummy"](((16, 3, 224, 224)))
        dataloader = DATALOADERS["pytorch"](dataset)
        q_model = fit(ori_model, conf=PostTrainingQuantConfig(), calib_dataloader=dataloader)
        op_to_traces = hawq_top(
            fp32_model=pt_model, q_model=q_model, dataloader=dataloader, criterion=None, enable_act=True
        )
        self.assertIsNotNone(op_to_traces)


@unittest.skipIf(not FX_MODE, "Unsupported Fx Mode with PyTorch Version Below 1.8")
class TestPyTorchBlockDetector(unittest.TestCase):
    def test_block_detector(self):
        from transformers import BertModel

        from neural_compressor.adaptor.torch_utils.pattern_detector import (
            BLOCK_PATTERNS,
            TransformerBasedModelBlockPatternDetector,
        )

        model = BertModel.from_pretrained("bert-base-uncased")
        detector = TransformerBasedModelBlockPatternDetector(model, BLOCK_PATTERNS)
        result = detector.detect_block()
        assert len(result["attention_blocks"]), 12
        assert len(result["ffn_blocks"]), 12

        found_attention_op = False
        found_dense_op = False
        for block in ["attention_blocks"]:
            for op in block:
                if "dense" in op:
                    found_dense_op = True
                    break

        for block in ["ffn_blocks"]:
            for op in block:
                if "attention" in op:
                    found_attention_op = True
                    break
        assert not found_attention_op
        assert not found_dense_op


if __name__ == "__main__":
    unittest.main()
