# Inspired by the quantization transformation, the bias idea it to
# use `subgraph_rewriter.replace_pattern` to replace the linear pattern
# with a new pattern that uses half precision.


import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.fx import subgraph_rewriter
from torch.fx.experimental.proxy_tensor import make_fx

torch.manual_seed(0)


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
    return model, example_inputs


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


def linear_fn_bias(i, w, b):
    return torch.nn.functional.linear(i, w, b)


def linear_fn(i, w):
    return torch.nn.functional.linear(i, w)


linear_pattern = make_fx(linear_fn, pre_dispatch=True)(torch.randn(0, 0), torch.randn(0, 0))
print(linear_pattern.print_readable())

linear_pattern_bias = make_fx(linear_fn_bias, pre_dispatch=True)(torch.randn(0, 0), torch.randn(0, 0), torch.randn(0))
print(linear_pattern_bias.print_readable())


def linear_fp16_replace(i, w, bias):
    fp_i = i.half()
    fp_w = w.half()
    fp_bias = bias.half()
    fp_r = torch.nn.functional.linear(fp_i, fp_w, fp_bias)
    fp_r_fp32 = fp_r.float()
    return fp_r_fp32


linear_replace = torch.fx.symbolic_trace(linear_fp16_replace)


linear_replace_pattern_bias = make_fx(linear_fp16_replace, pre_dispatch=True)(
    torch.randn(0, 0), torch.randn(0, 0), torch.randn(0)
)


def replace_with_vis(g, search_pattern, replace_pattern):
    print(f"search pattern, {search_pattern.print_readable(False)}")
    print(f"replace pattern: {replace_pattern.print_readable(False)}")
    subgraph_rewriter.replace_pattern(g, search_pattern, replace_pattern)


@torch.no_grad()
def test_quantizer_on_simple_model():
    model, example_inputs = build_model_include_conv_and_linear()

    quantizer = X86InductorQuantizer()
    # quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())
    quantizer.set_module_type_qconfig(torch.nn.Conv2d, xiq.get_default_x86_inductor_quantization_config())

    exported_model = capture_pre_autograd_graph(model, example_inputs)

    # prepare
    prepare_model = prepare_pt2e(exported_model, quantizer)
    # calibrate
    for i in range(2):
        prepare_model(*example_inputs)
    # convert
    converted_model = convert_pt2e(prepare_model)

    from neural_compressor.torch.algorithms.pt2e_quant.passes import half_precision_convert

    half_precision_convert(gm=converted_model, target_dtype=torch.float16, node_list=[])

    for node in converted_model.graph.nodes:
        if meta := getattr(node, "meta"):
            if quantization_annotation := meta.get("quantization_annotation"):
                print(f"node: {node} quantization_annotation {quantization_annotation._annotated}")
            else:
                print(f"node: {node} !!!!!!!!! no quantization_annotation")

    # float 32 result:
    out = model(*example_inputs)
    print(f"int8 + float32: {out}")

    # Option1
    # replace_with_vis(converted_model, linear_pattern_bias, linear_replace)
    # Option 2 (Rec.)
    replace_with_vis(converted_model, linear_pattern_bias, linear_replace_pattern_bias)

    new_out = converted_model(*example_inputs)
    print(f"int8 + float16: {new_out}")
    eps = 1e-3
    count_diff = (out - new_out).abs().gt(eps).sum()
    print(f"count diff: {count_diff}")
    max_diff = (out - new_out).abs().max()
    print(f"max diff: {max_diff}")

    # inference
    from torch._inductor import config

    config.freezing = True
    opt_model = torch.compile(converted_model)
    # opt_model = converted_model
    opt_out = opt_model(*example_inputs)
    # logger.warning("out shape is %s", out.shape)
    print(f"opt_out: {opt_out}")
    diff_opt = (new_out - opt_out).abs().max()
    print(f"diff_opt: {diff_opt}")
    assert torch.allclose(new_out, opt_out, atol=1e-5)
    assert out is not None


# test_quantizer_on_simple_model()
