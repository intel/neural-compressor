import os

os.environ["LOGLEVEL"] = "DEBUG"

import subprocess
import sys

import jax.numpy as jnp
import keras
import pytest
from keras_hub.models import Gemma3CausalLM

from neural_compressor.jax import DynamicQuantConfig, StaticQuantConfig, quantize_model
from neural_compressor.jax.utils.utility import print_model


def _build_sequential_model(dynamic, const_vars):
    keras.utils.set_random_seed(0)
    model_dtype = jnp.dtype("float32")
    model = keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(4, activation="linear", dtype=model_dtype),
            keras.layers.Dense(2, activation="linear", dtype=model_dtype),
            keras.layers.EinsumDense("ab,bc->ac", output_shape=2, dtype=model_dtype),
        ]
    )
    if dynamic:
        config = DynamicQuantConfig(
            weight_dtype="int8",
            activation_dtype="int8",
            const_scale=const_vars,
            const_weight=const_vars,
        )
        return quantize_model(model, config)
    else:
        config = StaticQuantConfig(
            weight_dtype="int8",
            activation_dtype="int8",
            const_scale=const_vars,
            const_weight=const_vars,
        )

        def calib_fn(model):
            _ = model(jnp.arange(1, 5).reshape((1, 4)))

        return quantize_model(model, config, calib_fn)


def _build_gemma_model(dynamic):
    model_path = os.environ.get("MODELS_PATH", "/tf_dataset/jax") + "/gemma3_instruct_270m"
    gemma = Gemma3CausalLM.from_preset(model_path, dtype="bfloat16")

    def calib_fn(model):
        _ = model.generate("How many hydrogen elements are in water?", max_length=100)

    if dynamic:
        config = DynamicQuantConfig(
            weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", const_scale=False, const_weight=False
        )
        _calib_fn = None
    else:
        config = StaticQuantConfig(
            weight_dtype="fp8_e4m3", activation_dtype="fp8_e4m3", const_scale=False, const_weight=False
        )
        _calib_fn = calib_fn

    return quantize_model(gemma, config, _calib_fn)


def print_sequential_model_fn(dynamic, const_vars):
    model = _build_sequential_model(dynamic, const_vars)
    print_model(model)


def print_gemma_model_fn(dynamic):
    model = _build_gemma_model(dynamic)
    print_model(model)


class LayerRepresentation:
    def __init__(
        self, class_name, name=None, val_a_zero_point=False, val_a_scale=False, val_w_scale=False, layer_ref=None
    ):
        self.class_name = class_name
        self.name = name
        self.val_a_zero_point = val_a_zero_point
        self.val_a_scale = val_a_scale
        self.val_w_scale = val_w_scale
        self.layer_ref = layer_ref

    def compare_with_description(self, dsc, const_vars):
        self._validate_class_name(dsc)

        if self.name is not None:
            self._validate_name(dsc)

        if self.val_a_zero_point:
            self._validate_a_zero_point(dsc, const_vars)

        if self.val_a_scale:
            self._validate_a_scale(dsc, const_vars)

        if self.val_w_scale:
            self._validate_w_scale(dsc, const_vars)

    def _validate_class_name(self, dsc):
        class_name = dsc.split()[0]
        assert self.class_name == class_name

    def _validate_name(self, dsc):
        name = dsc.split()[1]
        assert self.name == name

    def _validate_a_zero_point(self, dsc, const_vars):
        i_beg = dsc.find("a_zero_point")
        i_end = dsc.find("]", i_beg)
        a_zero_point = dsc[i_beg : i_end + 1]

        assert a_zero_point.startswith(
            f"a_zero_point{'(attr)' if const_vars else ''}=["
        ), f"a_zero_point is missing or incorrect: {a_zero_point}"
        values = a_zero_point[a_zero_point.find("[") + 1 : -1].split()
        assert len(values) > 0
        if self.layer_ref:
            values = jnp.array([float(v) for v in values])
            ref = self.layer_ref.a_zero_point if const_vars else self.layer_ref.a_zero_point.value
            jnp.allclose(values, ref, atol=1e-5)

    def _validate_a_scale(self, dsc, const_vars):
        i_beg = dsc.find("a_scale")
        i_end = dsc.find("]", i_beg)
        a_scale = dsc[i_beg : i_end + 1]

        assert a_scale.startswith(
            f"a_scale{'(attr)' if const_vars else ''}=["
        ), f"a_scale is missing or incorrect: {a_scale}"
        values = a_scale[a_scale.find("[") + 1 : -1].split()
        assert len(values) > 0
        if self.layer_ref:
            values = jnp.array([float(v) for v in values])
            ref = self.layer_ref.a_scale if const_vars else self.layer_ref.a_scale.value
            jnp.allclose(values, ref, atol=1e-5)

    def _validate_w_scale(self, dsc, const_vars):
        i_beg = dsc.find("w_scale")
        i_end = dsc.find("]", i_beg)
        w_scale = dsc[i_beg : i_end + 1]

        assert w_scale.startswith(
            f"w_scale{'(attr)' if const_vars else ''}=["
        ), f"w_scale is missing or incorrect: {w_scale}"
        values = w_scale[w_scale.rfind("[") + 1 : -1].split()
        assert len(values) > 0
        if self.layer_ref:
            values = jnp.array([float(v) for v in values])
            ref = self.layer_ref.w_scale if const_vars else self.layer_ref.w_scale.value
            jnp.allclose(values, ref, atol=1e-5)


def _layer_descriptions(log):
    def _remove_logger_prefix(line):
        i = line.find("]")
        i = line.find("]", i + 1)
        return line[i + 1 :]

    in_representation = False
    dsc = str()
    for line in log.decode(encoding="utf-8").split("\n"):
        print(line.strip())
        if in_representation:
            if dsc == str():
                # first layer
                dsc = line
                continue
            if "[DEBUG]" in line or line == str():
                # next layer or last empty line
                yield _remove_logger_prefix(dsc)
                dsc = line
            else:
                # continuation of previous layer in the next line
                dsc += line
        else:
            in_representation = line.endswith("internal representation:")


def _expected_sequential_layers(dynamic, const_vars):
    model = _build_sequential_model(dynamic, const_vars)
    if dynamic:
        return [
            LayerRepresentation(class_name="KerasQuantizedModelWrapper"),
            LayerRepresentation(class_name="InputLayer", name=".input_layer"),
            LayerRepresentation(
                class_name="QDynamicDense", name=".dense", val_w_scale=True, layer_ref=model._flatten_layers()[2]
            ),
            LayerRepresentation(class_name="DynamicQDQLayer", name=".dense.input_qdq"),
            LayerRepresentation(
                class_name="QDynamicDense", name=".dense_1", val_w_scale=True, layer_ref=model._flatten_layers()[4]
            ),
            LayerRepresentation(class_name="DynamicQDQLayer", name=".dense_1.input_qdq"),
            LayerRepresentation(
                class_name="QDynamicEinsumDense",
                name=".einsum_dense",
                val_w_scale=True,
                layer_ref=model._flatten_layers()[6],
            ),
            LayerRepresentation(class_name="DynamicQDQLayer", name=".einsum_dense.input_qdq"),
        ]
    else:
        return [
            LayerRepresentation(class_name="KerasQuantizedModelWrapper"),
            LayerRepresentation(class_name="InputLayer", name=".input_layer"),
            LayerRepresentation(
                class_name="QStaticDense",
                name=".dense",
                val_a_zero_point=True,
                val_a_scale=True,
                val_w_scale=True,
                layer_ref=model._flatten_layers()[2],
            ),
            LayerRepresentation(
                class_name="QStaticDense",
                name=".dense_1",
                val_a_zero_point=True,
                val_a_scale=True,
                val_w_scale=True,
                layer_ref=model._flatten_layers()[3],
            ),
            LayerRepresentation(
                class_name="QStaticEinsumDense",
                name=".einsum_dense",
                val_a_zero_point=True,
                val_a_scale=True,
                val_w_scale=True,
                layer_ref=model._flatten_layers()[4],
            ),
        ]


@pytest.mark.parametrize("dynamic", [False, True], ids=["dynamic=False", "dynamic=True"])
@pytest.mark.parametrize("const_vars", [False, True], ids=["const_vars=False", "const_vars=True"])
@pytest.mark.smoke_test_if("const_vars=False-dynamic=False", "const_vars=True-dynamic=True")
def test_print_sequential_model(dynamic, const_vars):
    cmd = (
        "import runpy\n"
        f"mod = runpy.run_path({__file__!r})\n"
        f"mod['print_sequential_model_fn']({dynamic}, {const_vars})\n"
    )
    process = subprocess.run([sys.executable, "-c", cmd], capture_output=True)
    descriptions = _layer_descriptions(process.stderr)

    for layer in _expected_sequential_layers(dynamic, const_vars):
        layer.compare_with_description(next(descriptions), const_vars)

    with pytest.raises(StopIteration):
        next(descriptions)  # empty line
        next(descriptions)  # StopIteration


@pytest.mark.parametrize("dynamic", [False, True], ids=["dynamic=False", "dynamic=True"])
def test_print_gemma(dynamic):
    def _traverse_layers(layer):
        yield layer
        for l in layer._layers:
            yield from _traverse_layers(l)

    cmd = "import runpy\n" f"mod = runpy.run_path({__file__!r})\n" f"mod['print_gemma_model_fn']({dynamic})\n"
    process = subprocess.run([sys.executable, "-c", cmd], capture_output=True)
    descriptions = _layer_descriptions(process.stderr)
    gemma_layers = _traverse_layers(_build_gemma_model(dynamic))

    for layer in gemma_layers:
        LayerRepresentation(
            class_name=layer.__class__.__name__,
            name=None,
            val_a_zero_point=hasattr(layer, "a_zero_point"),
            val_a_scale=hasattr(layer, "a_scale"),
            val_w_scale=hasattr(layer, "w_scale"),
            layer_ref=layer,
        ).compare_with_description(next(descriptions), const_vars=False)

    with pytest.raises(StopIteration):
        next(descriptions)  # empty line
        next(descriptions)  # StopIteration
