from unittest.mock import patch

import pytest
import torch

import neural_compressor.torch.utils.environ as inc_torch_env
from neural_compressor.torch.utils.utility import get_double_quant_config_dict


def get_gpt_j():
    import transformers

    tiny_gptj = transformers.AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-GPTJForCausalLM",
        torchscript=True,
    )
    return tiny_gptj


def build_simple_torch_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc1 = torch.nn.Linear(8, 30)
            self.fc2 = torch.nn.Linear(30, 60)
            self.fc3 = torch.nn.Linear(60, 30)
            self.fc4 = torch.nn.Linear(30, 50)

        def forward(self, x):
            out = self.fc1(x)
            out = self.fc2(out)
            out = self.fc3(out)
            out = self.fc4(out)
            return out

    model = Model()
    return model


from neural_compressor.torch.utils.utility import fetch_module, set_module


class TestTorchUtils:
    def setup_class(self):
        self.model = get_gpt_j()

    def teardown_class(self):
        pass

    @pytest.mark.parametrize(
        "module_name",
        [
            "transformer.h.2.mlp.fc_in",
            "transformer.nonexistent_attr",
        ],
    )
    def test_fetch_set_module(self, module_name):
        # fetch
        result = fetch_module(self.model, module_name)
        if "nonexistent_attr" in module_name:
            assert result is None, "result should be None"
        else:
            assert isinstance(result, torch.nn.Linear), "fetched module should be Linear"
            assert result.bias is not None, "The bias of fetched module should not be None."
        # set
        new_value = torch.nn.Linear(32, 128, bias=False)
        set_module(self.model, module_name, new_value)
        result = fetch_module(self.model, module_name)
        print(result)
        assert result.bias is None, "The bias of new module should be None."

    def test_get_model_info(self):
        from neural_compressor.torch.utils.utility import get_model_info

        white_module_list = [torch.nn.Linear]
        model_info = get_model_info(build_simple_torch_model(), white_module_list)
        assert len(model_info) == 4, "The length of model_info should be 4."

    @pytest.mark.parametrize("double_quant_type", ["BNB_NF4", "GGML_TYPE_Q4_K"])
    def test_double_quant_config_dict(self, double_quant_type):
        config_dict = get_double_quant_config_dict(double_quant_type)
        assert isinstance(config_dict, dict), "The returned object should be a dict."


class TestPackingWithNumba:

    @patch.object(inc_torch_env, "_is_tbb_installed", lambda: False)
    def test_tbb_not_installed(self):
        assert inc_torch_env.is_tbb_available() is False, "`is_tbb_available` should return False."
        assert inc_torch_env.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."

    @patch.object(inc_torch_env, "_is_tbb_installed", lambda: True)
    @patch.object(inc_torch_env, "_is_tbb_configured", lambda: False)
    def test_tbb_installed_but_not_configured_right(self):
        assert inc_torch_env.is_tbb_available() is False, "`is_tbb_available` should return False."
        assert inc_torch_env.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."

    @patch.object(inc_torch_env, "is_numba_available", lambda: False)
    def test_numba_not_installed(self):
        assert inc_torch_env.can_pack_with_numba() is False, "`can_pack_with_numba` should return False."
