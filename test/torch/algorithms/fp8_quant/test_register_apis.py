import shutil
from typing import Any, Dict, List, Optional

import pytest
import torch

from neural_compressor.torch.algorithms.fp8_quant import (
    ModuleConfig,
    ModuleExtraConfig,
    ModuleInfo,
    ModuleType,
    ObserverBase,
    PatchedModuleBase,
    ScalingMethodBase,
    register_observer,
    register_patched_module,
    register_scaling_methods,
)
from neural_compressor.torch.algorithms.fp8_quant._core.fp_utils import invert_scales
from neural_compressor.torch.algorithms.fp8_quant._core.quant_dequant import (
    DequantOutput,
    QuantDequantNone,
    QuantInput,
)
from neural_compressor.torch.algorithms.fp8_quant.model_configs import (
    OBSERVER_PARAMS,
    OBSERVER_TYPES,
    PATCHED_MODULE_TABLE,
    get_patched_module_table,
)
from neural_compressor.torch.algorithms.fp8_quant.observer import register_module_config_for_observer
from neural_compressor.torch.algorithms.fp8_quant.scaling_method_base import SCALING_METHODS_TABLE


class _NewModuleAlphaForTest(torch.nn.Module):
    _type_str = "new_module_1"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class _NewModuleBetaForTest(torch.nn.Module):
    _type_str = "new_module_2"

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@torch.no_grad()
def test_register_patched_module():
    @register_patched_module(_NewModuleAlphaForTest)
    class PatchedNewModuleForTest(PatchedModuleBase):
        def __init__(
            self,
            mod: torch.nn.Module,
            parent: torch.nn.Module,
            mod_extra_config: ModuleExtraConfig,
            name: Optional[str] = None,
        ):
            super().__init__(mod, parent, mod_extra_config, name)

        @classmethod
        def get_type(cls) -> str:
            return _NewModuleAlphaForTest._type_str

        @classmethod
        def get_module_type(cls) -> ModuleType:
            return ModuleType(num_inputs=1, param_names=[], num_outputs=1, required_output=False)

        @classmethod
        def get_module_config(cls) -> ModuleConfig:
            return ModuleConfig(inputs=(None,), outputs=(None,), params=None)

    assert _NewModuleAlphaForTest.__name__ in get_patched_module_table()


@torch.no_grad()
def test_register_patched_module_multiple_types():
    @register_patched_module([_NewModuleAlphaForTest, _NewModuleBetaForTest])
    class PatchedNewModuleForTest(PatchedModuleBase):
        def __init__(
            self,
            mod: torch.nn.Module,
            parent: torch.nn.Module,
            mod_extra_config: ModuleExtraConfig,
            name: Optional[str] = None,
        ):
            super().__init__(mod, parent, mod_extra_config, name)

        @classmethod
        def get_type(cls) -> str:
            return _NewModuleAlphaForTest._type_str

        @classmethod
        def get_module_type(cls) -> ModuleType:
            return ModuleType(num_inputs=1, param_names=[], num_outputs=1, required_output=False)

        @classmethod
        def get_module_config(cls) -> ModuleConfig:
            return ModuleConfig(inputs=(None,), outputs=(None,), params=None)

    assert _NewModuleAlphaForTest.__name__ in get_patched_module_table()
    assert _NewModuleBetaForTest.__name__ in get_patched_module_table()


@torch.no_grad()
@pytest.mark.parametrize("device_types", [["hpu", "cpu"], "hpu", "xpu", "cuda", "cpu"])
def test_register_patched_module_multiple_device_types(device_types):
    @register_patched_module(supported_float_module_types=_NewModuleAlphaForTest, device_types=device_types)
    class PatchedNewModuleForTest(PatchedModuleBase):
        def __init__(
            self,
            mod: torch.nn.Module,
            parent: torch.nn.Module,
            mod_extra_config: ModuleExtraConfig,
            name: Optional[str] = None,
        ):
            super().__init__(mod, parent, mod_extra_config, name)

        @classmethod
        def get_type(cls) -> str:
            return _NewModuleAlphaForTest._type_str

        @classmethod
        def get_module_type(cls) -> ModuleType:
            return ModuleType(num_inputs=1, param_names=[], num_outputs=1, required_output=False)

        @classmethod
        def get_module_config(cls) -> ModuleConfig:
            return ModuleConfig(inputs=(None,), outputs=(None,), params=None)

    devic_types = [device_types] if isinstance(device_types, str) else device_types
    for device_type in devic_types:
        assert _NewModuleAlphaForTest.__name__ in get_patched_module_table(
            device_type
        ), f"Expected module to be registered for device type {device_type}"


def _model_has_module(model, module_type):
    for module in model.modules():
        if isinstance(module, module_type):
            return True
    return False


@register_module_config_for_observer(
    module_name="linear", inputs_param=({"dim": 0},), outputs_param=({"dim": -1},), weight_param={"dim": -1}
)
@register_observer(observer_type="fake")
class FakeObserver(ObserverBase):
    def __init__(self, name, mod, d_shape=None, params=None, device="hpu"):
        super().__init__(name=name, mod=mod, device=device)
        self.state = torch.zeros((1, 1), device=self.device)
        self.dim = params.get("dim", None)

    def measure(self, x):
        self.update_state(x)
        self.used = True

    def update_state(self, x):
        pass

    def is_used(self):
        return self.used


def change_to_cur_file_dir():
    import os

    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    os.chdir(current_directory)


## TODO enable after SW-217369
@pytest.mark.skip(reason="This test is temporarily disabled")
class TestRegisterAPIs:
    def teardown_class(self):
        shutil.rmtree("test_outputs", ignore_errors=True)
        shutil.rmtree("hqt_output", ignore_errors=True)

    @torch.no_grad()
    def test_register_new_module_e2e(self):
        from pathlib import Path

        import habana_frameworks.torch.core as htcore

        class NewModuleForTest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight1 = torch.nn.Parameter(torch.randn(10, 10))
                self.weight2 = torch.nn.Parameter(torch.randn(10, 10))

            def forward(self, x):
                output = torch.matmul(x, self.weight1) + torch.matmul(x, self.weight2)
                return output

        @register_patched_module(supported_float_module_types=NewModuleForTest, device_types="hpu")
        class PatchedNewModuleForTest(PatchedModuleBase):
            def __init__(
                self,
                mod: torch.nn.Module,
                parent: torch.nn.Module,
                mod_extra_config: ModuleExtraConfig,
                *args,
                **kwargs,
            ):
                super().__init__(mod, parent, mod_extra_config, *args, **kwargs)

            @classmethod
            def get_type(cls) -> str:
                return NewModuleForTest.__name__

            @classmethod
            def get_module_type(cls) -> ModuleType:
                return ModuleType(
                    num_inputs=1,
                    param_names=["weight1", "weight2"],
                    num_outputs=1,
                    required_output=True,
                )

            @classmethod
            def get_module_config(cls) -> ModuleConfig:
                return ModuleConfig(inputs=(None,), outputs=(None,), params=None)

            def forward_measure(self, input):
                from neural_compressor.torch.algorithms.fp8_quant._core.common import (
                    measure_input,
                    measure_output,
                )

                measure_input((input,), observer=self._mod_extra_config.inputs)
                output = self.orig_mod(input)
                measure_output((output,), self._mod_extra_config.outputs)
                return output

            def forward_quant(self, input):
                # Just test the forward_quant is called, not the actual implementation
                print(f"Calling forward_quant for {self.__class__.__name__}")
                return input

        patched_module_info: ModuleInfo = PatchedNewModuleForTest.get_module_info()
        patched_module_type: ModuleType = PatchedNewModuleForTest.get_module_type()

        @register_scaling_methods(patched_module_info.type, "act_maxabs_pts_weight_maxabs_pts_pow2")
        class ScalingMethodForPatchedNewModuleForTest(ScalingMethodBase):
            @staticmethod
            def generate_op_scale_method(
                mod: torch.nn.Module,
                measurement: Optional[ModuleConfig] = None,
                params: Optional[Dict[str, Any]] = None,
                device: Optional[str] = "hpu",
            ) -> ModuleConfig:
                hp_dtype = params["hp_dtype"]
                input_scales = [
                    torch.tensor(1.0, dtype=hp_dtype, device=device) for _ in range(patched_module_type.num_inputs)
                ]
                output_scales = [
                    torch.tensor(1.0, dtype=hp_dtype, device=device) for _ in range(patched_module_type.num_outputs)
                ]
                weight_scales = {
                    param_name: torch.tensor(1.0, dtype=hp_dtype, device=device)
                    for param_name in patched_module_type.param_names
                }
                return ModuleConfig(inputs=input_scales, outputs=output_scales, params=weight_scales)

            @staticmethod
            def op_scales_to_mod_config(
                mod: torch.nn.Module, scales: ModuleConfig, params: Dict[str, Any], device: Optional[str] = "hpu"
            ) -> ModuleConfig:
                scales_inv = invert_scales(scales)
                lp_dtype = params["lp_dtype"]
                hp_dtype = params["hp_dtype"]
                input_configs = [QuantInput(s_inv, lp_dtype, hp_dtype) for s_inv in scales_inv.inputs]
                output_configs = [DequantOutput(s_inv, lp_dtype, hp_dtype) for s_inv in scales_inv.outputs]
                weight_configs = {
                    param_name: QuantInput(s_inv, lp_dtype, hp_dtype) for param_name, s_inv in scales_inv.params.items()
                }
                config = ModuleConfig(inputs=input_configs, outputs=output_configs, params=weight_configs)
                return config

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner_mod = NewModuleForTest()

            def forward(self, x):
                return self.inner_mod(x)

        device = "hpu"
        model = TinyModel()
        model.eval()
        model = model.to(device).to(torch.bfloat16)
        htcore.hpu_initialize()
        from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare

        cur_path = Path(__file__).parent
        config_file_path = cur_path / "test_jsons/test_measure.json"
        measure_config = FP8Config.from_json_file(config_file_path)
        model = prepare(model, measure_config)
        print(model)
        # Check if the `PatchedNewModuleForTest` is registered in the model
        assert _model_has_module(model, PatchedNewModuleForTest), "Register new module failed"
        inner_mod_mod_extra_config = model.inner_mod._mod_extra_config
        cur_module_type = PatchedNewModuleForTest.get_module_type()

        # Check if the patched module has the correct number of inputs, outputs and params
        assert (
            len(inner_mod_mod_extra_config.inputs) == cur_module_type.num_inputs
        ), f"Expected {cur_module_type.num_inputs} observers for inputs, but got {len(inner_mod_mod_extra_config.inputs)}"
        assert (
            len(inner_mod_mod_extra_config.outputs) == cur_module_type.num_outputs
        ), f"Expected {cur_module_type.num_outputs} observers for outputs, but got {len(inner_mod_mod_extra_config.outputs)}"
        assert len(inner_mod_mod_extra_config.params) == len(
            cur_module_type.param_names
        ), f"Expected {len(cur_module_type.param_names)} observers for params, but got {len(inner_mod_mod_extra_config.params)}"

        # Forward the model
        example_input = torch.randn(10, 10).to(device).to(torch.bfloat16)
        with torch.no_grad():
            out = model(example_input)
        assert out is not None, "Run observed model failed"
        finalize_calibration(model)
        # Convert model
        model = TinyModel()
        model.eval()
        model = model.to(device).to(torch.bfloat16)
        quant_config = FP8Config.from_json_file(cur_path / "test_jsons/test_pow2_quant.json")
        model = convert(model, quant_config)
        print(model)
        with torch.no_grad():
            out = model(example_input)
        assert out is not None, "Run quantized model failed"

    @torch.no_grad()
    def test_register_observer(self):
        import habana_frameworks.torch.core as htcore

        from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import PatchedLinear
        from neural_compressor.torch.quantization import FP8Config, finalize_calibration, prepare

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(30, 50)
                self.fc2 = torch.nn.Linear(50, 50)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                output = self.fc1(x)
                output = self.fc2(output)
                output = self.relu(output)
                return output

        change_to_cur_file_dir()

        assert "maxabs" in OBSERVER_TYPES, "maxabs observer should be in OBSERVER_TYPES."
        assert "fake" in OBSERVER_TYPES, "fake observer should be in OBSERVER_TYPES."
        assert "fake" in OBSERVER_PARAMS, "fake observer parameter should be in OBSERVER_PARAMS."
        assert "linear" in OBSERVER_PARAMS["fake"], "linear should be in OBSERVER_PARAMS['fake']."

        device = "hpu"
        model = TinyModel()
        model = model.eval().to(device)
        htcore.hpu_initialize()

        config = FP8Config.from_json_file("test_jsons/test_fake_measure.json")
        model = prepare(model, config)
        assert model.fc1._mod_extra_config.inputs[0].dim == 0, "dim of fc1 input should be 0."
        assert model.fc1._mod_extra_config.params["weight"].dim == -1, "dim of fc1 weight should be -1."

        model(torch.rand((1, 30)).to(device))
        model(torch.rand((1, 30)).to(device))
        finalize_calibration(model)
        assert isinstance(model.fc1, PatchedLinear), "fc1 is not observed."

        with pytest.raises(ValueError):

            @register_observer(observer_type="fake2")
            @register_module_config_for_observer(
                module_name="linear", inputs_param=({"dim": 0},), outputs_param=({"dim": -1},), weight_param={"dim": -1}
            )
            class FakeObserver2(ObserverBase):
                def __init__(self, name, mod, d_shape=None, params=None, device="hpu"):
                    super().__init__(name=name, mod=mod, device=device)
                    self.state = torch.zeros((1, 1), device=self.device)
                    self.dim = params.get("dim", None)

                def measure(self, x):
                    self.update_state(x)
                    self.used = True

                def update_state(self, x):
                    pass

                def is_used(self):
                    return self.used
