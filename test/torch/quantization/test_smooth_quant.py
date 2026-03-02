import copy
import os
import shutil

import pytest
import torch
from packaging.version import Version

from neural_compressor.torch.quantization import SmoothQuantConfig, convert, get_default_sq_config, prepare, quantize
from neural_compressor.torch.utils import is_ipex_available

if is_ipex_available():
    import intel_extension_for_pytorch as ipex
os.environ["FORCE_FP32"] = "1"


class Model(torch.nn.Module):
    device = torch.device("cpu")

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


model = Model()
example_inputs = torch.rand([1, 3])


def run_fn(model):
    for i in range(10):
        model(example_inputs)


@pytest.mark.skipif(not Version(torch.__version__) < Version("2.9.0"), reason="only for torch<2.9.0 [ipex]")
class TestSmoothQuant:
    def teardown_class(self):
        shutil.rmtree("saved_results", ignore_errors=True)

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_smooth_quant_default(self):
        fp32_model = copy.deepcopy(model)
        quant_config = get_default_sq_config()
        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"

        fp32_model = copy.deepcopy(model)
        example_dict = {"x": example_inputs}
        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_dict)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_smooth_quant_fallback(self):
        fp32_model = copy.deepcopy(model)
        quant_config = get_default_sq_config()
        # fallback by op_type
        quant_config.set_local(torch.nn.Linear, SmoothQuantConfig(w_dtype="fp32", act_dtype="fp32"))
        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"

        for op, op_info in q_model.tune_cfg[" "]["q_op_infos"].items():
            if op_info["op_type"] == "<class 'torch.nn.modules.linear.Linear'>":
                dtype = q_model.tune_cfg[" "]["q_op_infos"][op]["input_tensor_infos"][0]["force_dtype"]
                assert dtype == "torch.float32", "Failed to fallback linear op, please check!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    @pytest.mark.parametrize(
        "act_sym, act_algo, alpha, scale_sharing",
        [
            (True, "kl", 0.1, True),
            (True, "minmax", 0.1, False),
            (False, "kl", 0.5, False),
            (False, "minmax", 0.5, True),
            (True, "minmax", 0.1, True),
            (False, "kl", 0.5, False),
        ],
    )
    def test_sq_linear_params(self, act_sym, act_algo, alpha, scale_sharing):
        fp32_model = copy.deepcopy(model)
        quant_config = SmoothQuantConfig(act_sym=act_sym, act_algo=act_algo, alpha=alpha, scale_sharing=scale_sharing)

        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"
        output1 = fp32_model(example_inputs)
        output2 = q_model(example_inputs)
        assert torch.allclose(output1, output2, atol=2e-2), "Accuracy gap atol > 0.02 is unexpected. Please check."

        example_dict = {"x": example_inputs}
        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_dict)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"
        output1 = fp32_model(example_inputs)
        output2 = q_model(example_inputs)
        assert torch.allclose(output1, output2, atol=2e-2), "Accuracy gap atol > 0.02 is unexpected. Please check."

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_sq_ipex_accuracy(self):
        qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5)
        user_model = copy.deepcopy(model)
        user_model = ipex.quantization.prepare(user_model.eval(), qconfig, example_inputs=example_inputs, inplace=True)

        def run_fn(model):
            model(example_inputs)

        run_fn(user_model)
        user_model.save_qconf_summary(qconf_summary="ipex.json")
        with torch.no_grad():
            user_model = ipex.quantization.convert(user_model.eval(), inplace=True).eval()
            user_model(example_inputs)
            user_model = torch.jit.trace(user_model.eval(), example_inputs, strict=False)
            user_model = torch.jit.freeze(user_model.eval())
            user_model(example_inputs)
            user_model(example_inputs)
        ipex_out = user_model(example_inputs)

        fp32_model = copy.deepcopy(model)
        quant_config = get_default_sq_config()
        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")

        inc_out = q_model(example_inputs)
        # set a big atol to avoid random issue
        assert torch.allclose(inc_out, ipex_out, atol=2e-02), "Unexpected result. Please double check."

        from neural_compressor.torch.algorithms.smooth_quant import recover_model_from_json

        fp32_model = copy.deepcopy(model)
        ipex_model = recover_model_from_json(fp32_model, "ipex.json", example_inputs=example_inputs)
        ipex_out = ipex_model(example_inputs)
        assert torch.allclose(inc_out, ipex_out, atol=2e-02), "Unexpected result. Please double check."

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_sq_save_load(self):
        fp32_model = copy.deepcopy(model)
        quant_config = get_default_sq_config()
        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")
        inc_out = q_model(example_inputs)

        from neural_compressor.torch.algorithms.smooth_quant import recover_model_from_json
        from neural_compressor.torch.quantization import load

        # load using saved model
        loaded_model = load("saved_results")
        loaded_out = loaded_model(example_inputs)
        # set a big atol to avoid random issue
        assert torch.allclose(inc_out, loaded_out, atol=2e-02), "Unexpected result. Please double check."

        # compare saved json file
        fp32_model = copy.deepcopy(model)
        loaded_model = recover_model_from_json(fp32_model, "saved_results/qconfig.json", example_inputs=example_inputs)
        loaded_out = loaded_model(example_inputs)
        assert torch.allclose(inc_out, loaded_out, atol=1e-05), "Unexpected result. Please double check."

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_smooth_quant_with_quantize_API(self):
        fp32_model = copy.deepcopy(model)
        quant_config = get_default_sq_config()
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

        fp32_model = copy.deepcopy(model)
        example_dict = {"x": example_inputs}
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_dict)
        assert q_model is not None, "Quantization failed!"

        quant_config.folding = True
        fp32_model = copy.deepcopy(model)
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

        fp32_model = copy.deepcopy(model)
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_dict)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_sq_save_load_with_quantize_API(self):
        fp32_model = copy.deepcopy(model)
        quant_config = get_default_sq_config()
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"
        q_model.save("saved_results")
        inc_out = q_model(example_inputs)

        from neural_compressor.torch.algorithms.smooth_quant import recover_model_from_json
        from neural_compressor.torch.quantization import load

        # load using saved model
        loaded_model = load("saved_results")
        loaded_out = loaded_model(example_inputs)
        # set a big atol to avoid random issue
        assert torch.allclose(inc_out, loaded_out, atol=2e-02), "Unexpected result. Please double check."

        # compare saved json file
        fp32_model = copy.deepcopy(model)
        # quant_config.folding = True is not allowed to recover with json because it will update model weights
        loaded_model = recover_model_from_json(fp32_model, "saved_results/qconfig.json", example_inputs=example_inputs)
        loaded_out = loaded_model(example_inputs)
        assert torch.allclose(inc_out, loaded_out, atol=1e-05), "Unexpected result. Please double check."

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    def test_smooth_quant_mixed_precision(self):
        fp32_model = copy.deepcopy(model)
        quant_config = get_default_sq_config()  # do mixed_precison by default.

        # prepare/convert API
        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"

        quant_config.excluded_precisions = ["bf16"]
        prepared_model = prepare(fp32_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(prepared_model)
        q_model = convert(prepared_model)
        assert q_model is not None, "Quantization failed!"

        # quantize API
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

        quant_config.excluded_precisions = ["bf16"]
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

        quant_config.folding = True
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"

    @pytest.mark.skipif(not is_ipex_available(), reason="Requires IPEX")
    @pytest.mark.parametrize(
        "shared_criterion, do_blockwise, folding",
        [
            ("max", True, True),
            ("max", False, False),
            ("mean", True, False),
            ("mean", False, True),
            ("min", True, False),
            ("min", False, True),
        ],
    )
    def test_smooth_quant_auto(self, shared_criterion, do_blockwise, folding):
        fp32_model = copy.deepcopy(model)

        def run_fn(model):
            for i in range(100):
                model(example_inputs)

        quant_config = SmoothQuantConfig(
            alpha="auto",
            alpha_min=0.45,
            alpha_max=0.55,
            alpha_step=0.01,
            shared_criterion=shared_criterion,
            do_blockwise=do_blockwise,
            folding=folding,
        )
        q_model = quantize(fp32_model, quant_config=quant_config, run_fn=run_fn, example_inputs=example_inputs)
        assert q_model is not None, "Quantization failed!"
        output1 = fp32_model(example_inputs)
        output2 = q_model(example_inputs)
        assert torch.allclose(output1, output2, atol=2e-2), "Accuracy gap atol > 0.02 is unexpected. Please check."
