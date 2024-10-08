import shutil

import pytest
import torch
import torch.testing._internal.common_quantization as torch_test_quant_common

from neural_compressor.common.utils import logger
from neural_compressor.torch.export import export
from neural_compressor.torch.quantization import (
    DynamicQuantConfig,
    StaticQuantConfig,
    convert,
    get_default_dynamic_config,
    get_default_static_config,
    prepare,
    quantize,
)
from neural_compressor.torch.utils import GT_OR_EQUAL_TORCH_VERSION_2_5, TORCH_VERSION_2_2_2, get_torch_version

torch.manual_seed(0)


@pytest.fixture
def force_not_import_ipex(monkeypatch):
    def _is_ipex_imported():
        return False

    monkeypatch.setattr("neural_compressor.torch.quantization.config.is_ipex_imported", _is_ipex_imported)
    monkeypatch.setattr("neural_compressor.torch.quantization.algorithm_entry.is_ipex_imported", _is_ipex_imported)
    monkeypatch.setattr("neural_compressor.torch.export.pt2e_export.is_ipex_imported", _is_ipex_imported)


class TestPT2EQuantization:
    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    @staticmethod
    def get_toy_model():
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                x = a / (torch.abs(a) + 1)
                if b.sum() < 0:
                    b = b * -1
                return x * b

        inp1 = torch.randn(10)
        inp2 = torch.randn(10)
        example_inputs = (inp1, inp2)
        bar = Bar()
        return bar, example_inputs

    @staticmethod
    def build_model_include_conv_and_linear():
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, 5)
                self.pool = torch.nn.MaxPool2d(2, 2)
                self.conv2 = torch.nn.Conv2d(6, 16, 5)
                self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
                self.fc2 = torch.nn.Linear(120, 84)

            def forward(self, x):
                x = self.conv1(x)
                x = self.pool(torch.nn.functional.relu(x))
                x = self.conv2(x)
                x = self.pool(torch.nn.functional.relu(x))
                x = x.view(-1, 16 * 5 * 5)
                x = torch.nn.functional.relu(self.fc1(x))
                x = torch.nn.functional.relu(self.fc2(x))

                return x

        model = Model()
        example_inputs = (torch.randn(1, 3, 32, 32),)
        return model, example_inputs

    @staticmethod
    def build_simple_torch_model_and_example_inputs():
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(20, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc1(x)
                x = torch.nn.functional.relu(x)
                x = self.fc2(x)
                return x

        model = SimpleModel()
        example_inputs = (torch.randn(10, 10),)
        exported_model = export(model, example_inputs=example_inputs)
        return exported_model, example_inputs

    @pytest.mark.skipif(get_torch_version() <= TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0")
    @pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
    def test_quantize_simple_model(self, granularity, force_not_import_ipex):
        from neural_compressor.torch.quantization import StaticQuantConfig

        model, example_inputs = self.build_simple_torch_model_and_example_inputs()
        float_model_output = model(*example_inputs)
        quant_config = None

        def calib_fn(model):
            for i in range(4):
                model(*example_inputs)

        quant_config = StaticQuantConfig(w_granularity=granularity)
        q_model = quantize(model=model, quant_config=quant_config, run_fn=calib_fn)
        from torch._inductor import config

        config.freezing = True
        q_model_out = q_model(*example_inputs)
        assert torch.allclose(float_model_output, q_model_out, atol=1e-2), "Quantization failed!"

        # test save and load
        q_model.save(
            example_inputs=example_inputs,
            output_dir="./saved_results",
        )
        from neural_compressor.torch.quantization import load

        loaded_quantized_model = load("./saved_results")
        loaded_q_model_out = loaded_quantized_model(*example_inputs)
        assert torch.equal(loaded_q_model_out, q_model_out)

        opt_model = torch.compile(q_model)
        out = opt_model(*example_inputs)
        logger.warning("out shape is %s", out.shape)
        assert out is not None

    @pytest.mark.skipif(not GT_OR_EQUAL_TORCH_VERSION_2_5, reason="Requires torch>=2.5")
    def test_quantize_simple_model_with_set_local(self, force_not_import_ipex):
        model, example_inputs = self.build_simple_torch_model_and_example_inputs()
        float_model_output = model(*example_inputs)
        quant_config = None

        def calib_fn(model):
            for i in range(4):
                model(*example_inputs)

        quant_config = get_default_static_config()
        quant_config.set_local("fc1", StaticQuantConfig(w_dtype="fp32", act_dtype="fp32"))
        q_model = quantize(model=model, quant_config=quant_config, run_fn=calib_fn)

        # check the half node
        expected_node_occurrence = {
            # Only quantize the `fc2`
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
        }
        expected_node_occurrence = {
            torch_test_quant_common.NodeSpec.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        node_in_graph = self.get_node_in_graph(q_model)
        for node, cnt in expected_node_occurrence.items():
            assert node_in_graph.get(node, 0) == cnt, f"Node {node} should occur {cnt} times, but {node_in_graph[node]}"

        from torch._inductor import config

        config.freezing = True
        q_model_out = q_model(*example_inputs)
        assert torch.allclose(float_model_output, q_model_out, atol=1e-2), "Quantization failed!"
        opt_model = torch.compile(q_model)
        out = opt_model(*example_inputs)
        assert out is not None

    @pytest.mark.skipif(get_torch_version() <= TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0")
    @pytest.mark.parametrize("is_dynamic", [False, True])
    def test_prepare_and_convert_on_simple_model(self, is_dynamic, force_not_import_ipex):
        model, example_inputs = self.build_simple_torch_model_and_example_inputs()
        quant_config = None

        def calib_fn(model):
            for i in range(2):
                model(*example_inputs)

        if is_dynamic:
            quant_config = get_default_dynamic_config()
        else:
            quant_config = get_default_static_config()

        prepared_model = prepare(model, quant_config=quant_config)
        calib_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"

        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(q_model)
        out = opt_model(*example_inputs)
        logger.warning("out shape is %s", out.shape)
        assert out is not None

    @pytest.mark.skipif(get_torch_version() <= TORCH_VERSION_2_2_2, reason="Requires torch>=2.3.0")
    def test_prepare_and_convert_on_llm(self, force_not_import_ipex):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
        example_inputs = (input_ids,)
        model = export(model, example_inputs=example_inputs)

        quant_config = get_default_static_config()
        # prepare
        prepare_model = prepare(model, quant_config)
        # calibrate
        for i in range(2):
            prepare_model(*example_inputs)
        # convert
        converted_model = convert(prepare_model)
        # inference
        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(converted_model)
        out = opt_model(*example_inputs)
        assert out.logits is not None

    @staticmethod
    def get_node_in_graph(graph_module):
        # Copied from torch.testing._internal.common_quantization
        nodes_in_graph = {}
        node_list = []
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        for node in graph_module.graph.nodes:
            n = None
            if node.op == "call_function" or node.op == "call_method":
                n = torch_test_quant_common.NodeSpec(node.op, node.target)
            elif node.op == "call_module":
                n = torch_test_quant_common.NodeSpec(node.op, type(modules[node.target]))

            if n is not None:
                node_list.append(n)
                if n in nodes_in_graph:
                    nodes_in_graph[n] += 1
                else:
                    nodes_in_graph[n] = 1
        return nodes_in_graph

    @pytest.mark.skipif(not GT_OR_EQUAL_TORCH_VERSION_2_5, reason="Requires torch>=2.5")
    def test_mixed_fp16_and_int8(self, force_not_import_ipex):
        model, example_inputs = self.build_model_include_conv_and_linear()
        model = export(model, example_inputs=example_inputs)

        quant_config = get_default_static_config()
        quant_config.set_local(torch.nn.Linear, StaticQuantConfig(w_dtype="fp16", act_dtype="fp16"))
        # prepare
        prepare_model = prepare(model, quant_config)
        # calibrate
        for i in range(2):
            prepare_model(*example_inputs)
        # convert
        converted_model = convert(prepare_model)

        # check the half node
        expected_node_occurrence = {
            # 4 `aten.to` for each `aten.linear`
            torch.ops.aten.to.dtype: 8,
            torch.ops.aten.linear.default: 2,
        }
        expected_node_occurrence = {
            torch_test_quant_common.NodeSpec.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        node_in_graph = self.get_node_in_graph(converted_model)
        for node, cnt in expected_node_occurrence.items():
            assert node_in_graph.get(node, 0) == cnt, f"Node {node} should occur {cnt} times, but {node_in_graph[node]}"

        # inference
        from torch._inductor import config

        config.freezing = True
        opt_model = torch.compile(converted_model)
        out = opt_model(*example_inputs)
        assert out is not None
