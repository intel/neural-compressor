#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gc
import math
import os
import re
from collections import OrderedDict, UserDict
from functools import partial

import yaml
from packaging.version import Version

from neural_compressor.utils.utility import dump_elapsed_time

from ..data.dataloaders.base_dataloader import BaseDataLoader
from ..utils import logger
from ..utils.utility import GLOBAL_STATE, MODE, CpuInfo, LazyImport, Statistics
from .adaptor import Adaptor, adaptor_registry
from .query import QueryBackendCapability

torch = LazyImport("torch")
json = LazyImport("json")
hvd = LazyImport("horovod.torch")
torch_utils = LazyImport("neural_compressor.adaptor.torch_utils")
ipex = LazyImport("intel_extension_for_pytorch")

REDUCE_RANGE = False if CpuInfo().vnni else True
logger.debug("Reduce range is {}".format(str(REDUCE_RANGE)))


def get_torch_version():
    try:
        torch_version = torch.__version__.split("+")[0]
    except ValueError as e:  # pragma: no cover
        assert False, "Got an unknown version of torch: {}".format(e)
    version = Version(torch_version)
    return version


def get_ipex_version():
    try:
        ipex_version = ipex.__version__.split("+")[0]
    except ValueError as e:  # pragma: no cover
        assert False, "Got an unknown version of intel_extension_for_pytorch: {}".format(e)
    version = Version(ipex_version)
    return version


def get_torch_white_list(approach):
    version = get_torch_version()
    import torch.quantization as tq

    if version.release < Version("1.7.0").release:  # pragma: no cover
        white_list = (
            set(tq.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING.keys())
            if approach == "post_training_dynamic_quant"
            else tq.default_mappings.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
        )
    elif version.release < Version("1.8.0").release:  # pragma: no cover
        white_list = (
            set(tq.quantization_mappings.get_dynamic_quant_module_mappings().keys())
            if approach == "post_training_dynamic_quant"
            else tq.quantization_mappings.get_qconfig_propagation_list()
        )
    else:
        white_list = (
            set(tq.quantization_mappings.get_default_dynamic_quant_module_mappings().keys())
            if approach == "post_training_dynamic_quant"
            else tq.quantization_mappings.get_default_qconfig_propagation_list()
        )
    return white_list


def pytorch_forward_wrapper(
    model,
    input,
    conf=None,
    backend="default",
    running_mode="inference",
):
    version = get_torch_version()
    from .torch_utils.util import forward_wrapper

    if (
        version.release < Version("1.12.0").release and backend == "ipex" and running_mode == "calibration"
    ):  # pragma: no cover
        with ipex.quantization.calibrate(conf, default_recipe=True):  # pylint: disable=E1101
            output = forward_wrapper(model, input)
    else:
        output = forward_wrapper(model, input)
    return output


def get_example_inputs(model, dataloader):
    version = get_torch_version()
    from .torch_utils.util import move_input_device

    # Suggest set dataloader like calib_dataloader
    if dataloader is None:
        return None
    device = next(model.parameters()).device
    try:
        for idx, (input, label) in enumerate(dataloader):
            input = move_input_device(input, device)
            output = pytorch_forward_wrapper(model, input)
            if isinstance(input, (dict, UserDict)):  # pragma: no cover
                assert version.release >= Version("1.12.0").release, "INC support IPEX version >= 1.12.0"
                if "label" in input.keys():
                    input.pop("label")
                if version.release <= Version("2.0.1").release:
                    return tuple(input.values())
                else:
                    return dict(input)

            if isinstance(input, (list, tuple)):
                return tuple(input)
            if isinstance(input, torch.Tensor):
                return input
            break
    except Exception as e:  # pragma: no cover
        for idx, input in enumerate(dataloader):
            input = move_input_device(input, device)
            output = pytorch_forward_wrapper(model, input)
            if isinstance(input, (dict, UserDict)):  # pragma: no cover
                assert version.release >= Version("1.12.0").release, "INC support IPEX version >= 1.12.0"
                if "label" in input.keys():
                    input.pop("label")
                if version.release <= Version("2.0.1").release:
                    return tuple(input.values())
                else:
                    return dict(input)
            if isinstance(input, list) or isinstance(input, tuple):
                return tuple(input)
            if isinstance(input, torch.Tensor):
                return input
            break
    if idx == 0:
        assert False, "Please checkout the example_inputs format."


def get_ops_recursively(model, prefix, ops={}):
    """This is a helper function for `graph_info`,
        and it will get all ops from model.

    Args:
        model (object): input model
        prefix (string): prefix of op name
        ops (dict): dict of ops from model {op name: type}.
    Returns:
        None
    """
    version = get_torch_version()
    if version.release < Version("1.7.0").release:  # pragma: no cover
        white_list = (
            set(torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING.values())
            | set(torch.quantization.default_mappings.DEFAULT_QAT_MODULE_MAPPING.values())
            | set(torch.quantization.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING.values())
            | set(torch.quantization.default_mappings.DEFAULT_MODULE_MAPPING.keys())
            | set(torch.quantization.default_mappings.DEFAULT_QAT_MODULE_MAPPING.keys())
            | set(torch.quantization.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING.keys())
            | torch.quantization.default_mappings._INCLUDE_QCONFIG_PROPAGATE_LIST
        )
    elif version.release < Version("1.8.0").release:  # pragma: no cover
        white_list = torch.quantization.get_compare_output_module_list()
    else:
        white_list = torch.quantization.get_default_compare_output_module_list()

    for name, child in model.named_children():
        op_name = prefix + "." + name if prefix != "" else name
        if (
            type(child) in white_list
            and not isinstance(child, torch.nn.Sequential)
            and type(child) != torch.quantization.stubs.DeQuantStub
        ):
            ops[op_name] = (
                unify_op_type_mapping[str(child.__class__.__name__)]
                if str(child.__class__.__name__) in unify_op_type_mapping
                else str(child.__class__.__name__)
            )
        get_ops_recursively(child, op_name, ops)


def _cfg_to_qconfig(tune_cfg, observer_type="post_training_static_quant"):
    """Convert tune configure to quantization config for each op.

    Args:
        tune_cfg (dict): dictionary of tune configure for each op
        observer_type (str, optional): specify observer type, Default is 'ptq_static',
                                       options: 'ptq_dynamic', 'qat'.

    Returns:
        op_qcfgs (dict): dictionary of quantization configure for each op

    tune_cfg should be a format like below:
    {
      'fuse': {'int8': [['CONV2D', 'RELU', 'BN'], ['CONV2D', 'RELU']],
               'fp32': [['CONV2D', 'RELU', 'BN']]},
      'calib_iteration': 10,
      'op': {
         ('op1', 'CONV2D'): {
           'activation':  {'dtype': 'uint8',
                           'algorithm': 'minmax',
                           'scheme':'sym',
                           'granularity': 'per_tensor'},
           'weight': {'dtype': 'int8',
                      'algorithm': 'kl',
                      'scheme':'asym',
                      'granularity': 'per_channel'}
         },
         ('op2', 'RELU): {
           'activation': {'dtype': 'int8',
           'scheme': 'asym',
           'granularity': 'per_tensor',
           'algorithm': 'minmax'}
         },
         ('op3', 'CONV2D'): {
           'activation':  {'dtype': 'fp32'},
           'weight': {'dtype': 'fp32'}
         },
         ...
      }
    }
    """
    op_qcfgs = OrderedDict()
    op_qcfgs["bf16_ops_list"] = []
    for key in tune_cfg["op"]:
        value = tune_cfg["op"][key]
        assert isinstance(value, dict)
        assert "activation" in value
        if ("weight" in value and value["weight"]["dtype"] == "fp32") or (
            "weight" not in value and value["activation"]["dtype"] == "fp32"
        ):
            op_qcfgs[key[0]] = None
        elif ("weight" in value and value["weight"]["dtype"] == "bf16") or (
            "weight" not in value and value["activation"]["dtype"] == "bf16"
        ):
            op_qcfgs["bf16_ops_list"].append(key)
            op_qcfgs[key[0]] = None
        else:
            if "weight" in value:
                weight = value["weight"]
                scheme = weight["scheme"]
                granularity = weight["granularity"]
                algorithm = weight["algorithm"]
                dtype = weight["dtype"]
                if observer_type == "quant_aware_training" and key[1] not in [
                    "Embedding",
                    "EmbeddingBag",
                    "LSTM",
                    "GRU",
                    "LSTMCell",
                    "GRUCell",
                    "RNNCell",
                ]:
                    weights_fake_quantize = _fake_quantize(algorithm, scheme, granularity, dtype)
                else:
                    weights_observer = _observer(algorithm, scheme, granularity, dtype)
            else:
                if observer_type == "quant_aware_training":
                    weights_fake_quantize = torch.quantization.default_weight_fake_quant
                else:
                    weights_observer = torch.quantization.default_per_channel_weight_observer

            activation = value["activation"]
            scheme = activation["scheme"]
            granularity = activation["granularity"]
            algorithm = activation["algorithm"]
            dtype = activation["dtype"]
            compute_dtype = (
                activation["compute_dtype"]
                if "compute_dtype" in activation and activation["compute_dtype"] is not None
                else "uint8"
            )

            if observer_type == "quant_aware_training":
                if key[1] in ["LSTM", "GRU", "LSTMCell", "GRUCell", "RNNCell"]:
                    activation_observer = _observer(
                        algorithm, scheme, granularity, dtype, "post_training_dynamic_quant", compute_dtype
                    )

                elif key[1] not in ["Embedding", "EmbeddingBag"]:
                    activation_fake_quantize = _fake_quantize(algorithm, scheme, granularity, dtype, compute_dtype)

                else:
                    activation_observer = _observer(algorithm, scheme, granularity, dtype, observer_type, compute_dtype)
            elif value["activation"]["quant_mode"] == "static":
                activation_observer = _observer(
                    algorithm, scheme, granularity, dtype, "post_training_static_quant", compute_dtype
                )
            elif value["activation"]["quant_mode"] == "dynamic":
                activation_observer = _observer(
                    algorithm, scheme, granularity, dtype, "post_training_dynamic_quant", compute_dtype
                )

            version = get_torch_version()
            if observer_type == "quant_aware_training":
                if key[1] in ["LSTM", "GRU", "LSTMCell", "GRUCell", "RNNCell", "Embedding", "EmbeddingBag"]:
                    if version.release >= Version("1.11.0").release:
                        if key[1] in ["Embedding", "EmbeddingBag"]:
                            qconfig = torch.quantization.float_qparams_weight_only_qconfig
                        else:
                            qconfig = torch.quantization.per_channel_dynamic_qconfig
                    else:
                        qconfig = torch.quantization.QConfigDynamic(
                            activation=activation_observer, weight=weights_observer
                        )
                else:
                    qconfig = torch.quantization.QConfig(
                        activation=activation_fake_quantize, weight=weights_fake_quantize
                    )
            elif value["activation"]["quant_mode"] == "static":
                qconfig = torch.quantization.QConfig(activation=activation_observer, weight=weights_observer)
            else:
                if version.release < Version("1.6.0").release:  # pragma: no cover
                    qconfig = torch.quantization.QConfigDynamic(weight=weights_observer)
                elif version.release >= Version("1.11.0").release:
                    if key[1] in ["Embedding", "EmbeddingBag"]:
                        qconfig = torch.quantization.float_qparams_weight_only_qconfig
                    else:
                        qconfig = torch.quantization.per_channel_dynamic_qconfig
                else:
                    qconfig = torch.quantization.QConfigDynamic(activation=activation_observer, weight=weights_observer)

            op_qcfgs[key[0]] = qconfig

    return op_qcfgs


def _cfgs_to_fx_cfgs(op_cfgs, observer_type="post_training_static_quant"):
    """Convert quantization config to a format that meets the requirements of torch.fx.

        Args:
            op_cfgs (dict): dictionary of quantization configure for each op
            observer_type (str, optional): specify observer type, Default is 'ptq_static',
                                           options: 'ptq_dynamic', 'qat'.

        Returns:
            fx_op_cfgs (dict): dictionary of quantization configure that meets
                               the requirements of torch.fx

    example: fx_op_cfgs = {"": default_qconfig,
                           "module_name": [("layer4.1.conv2", per_channel_weight_qconfig)]}
    """
    version = get_torch_version()
    if observer_type == "post_training_dynamic_quant":
        model_qconfig = torch.quantization.default_dynamic_qconfig
    elif observer_type == "quant_aware_training":
        model_qconfig = (
            torch.quantization.QConfig(
                activation=torch.quantization.FakeQuantize.with_args(
                    dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=REDUCE_RANGE
                ),
                weight=torch.quantization.default_weight_fake_quant,
            )
            if version.release < Version("1.10.0").release
            else torch.quantization.QConfig(
                activation=torch.quantization.FusedMovingAvgObsFakeQuantize.with_args(
                    dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=REDUCE_RANGE
                ),
                weight=torch.quantization.default_fused_per_channel_wt_fake_quant,
            )
        )
    else:
        model_qconfig = torch.quantization.QConfig(
            activation=torch.quantization.HistogramObserver.with_args(reduce_range=REDUCE_RANGE),
            weight=torch.quantization.default_per_channel_weight_observer,
        )

    if version.release >= Version("1.13.0").release:  # pragma: no cover
        from torch.ao.quantization import QConfigMapping

        fx_op_cfgs = QConfigMapping()
        if observer_type != "post_training_dynamic_quant":
            fx_op_cfgs.set_global(model_qconfig)
    else:
        fx_op_cfgs = dict()
        if observer_type != "post_training_dynamic_quant":
            fx_op_cfgs[""] = model_qconfig
        op_tuple_cfg_list = []

    for key, value in op_cfgs.items():
        if key == "default_qconfig":
            if version.release >= Version("1.13.0").release:  # pragma: no cover
                fx_op_cfgs.set_global(value)  # pylint: disable=E1101
            else:
                fx_op_cfgs[""] = value
            continue
        if version.release >= Version("1.13.0").release:  # pragma: no cover
            fx_op_cfgs.set_module_name(key, value)  # pylint: disable=E1101
        else:
            op_tuple = (key, value)
            op_tuple_cfg_list.append(op_tuple)

    if version.release < Version("1.13.0").release:  # pragma: no cover
        fx_op_cfgs["module_name"] = op_tuple_cfg_list
    elif observer_type != "post_training_dynamic_quant":
        from torch.ao.quantization import get_default_qconfig_mapping

        for name, q_config in get_default_qconfig_mapping().to_dict()["object_type"]:
            fx_op_cfgs.set_object_type(name, q_config)  # pylint: disable=E1101

    return fx_op_cfgs


def _observer(algorithm, scheme, granularity, dtype, observer_type="post_training_static_quant", compute_dtype="uint8"):
    """Construct an observer module, In forward, observer will update the statistics of
       the observed Tensor. And they should provide a `calculate_qparams` function
       that computes the quantization parameters given the collected statistics.

    Args:
        algorithm (string): What algorithm for computing the quantization parameters based on.
        scheme (string): Quantization scheme to be used.
        granularity (string): What granularity to computing the quantization parameters,
                              per channel or per tensor.
        dtype (string): Quantized data type
        observer_type (string): Observer type, default is 'post_training_static_quant'.

    Returns:
        oberser (object)
    """
    from .torch_utils.util import _get_signed_and_bits, calculate_quant_min_max, match_datatype_pattern

    if observer_type == "post_training_dynamic_quant" and get_torch_version().release >= Version("1.6.0").release:
        return torch.quantization.default_dynamic_quant_observer

    compute_dtype_dict = {"int8": torch.qint8, "uint8": torch.quint8, "None": None}
    if compute_dtype in compute_dtype_dict:
        compute_dtype = compute_dtype_dict[compute_dtype]
    else:  # pragma: no cover
        assert False, "Unsupported compute_dtype with {}".format(compute_dtype)

    quant_min, quant_max = None, None
    dtype_dict = {"int8": torch.qint8, "uint8": torch.quint8, "fp32": torch.float}
    if dtype in dtype_dict:
        torch_dtype = dtype_dict[dtype]
    else:  # pragma: no cover
        # TODO to handle int4
        if match_datatype_pattern(dtype):
            logger.info(
                (f"Currently, PyTorch does not natively support {dtype}," + "it will simulate its numerics instead.")
            )
            unsigned, num_bits = _get_signed_and_bits(dtype)
            torch_dtype = torch.quint8 if unsigned else torch.qint8
            quant_min, quant_max = calculate_quant_min_max(unsigned, num_bits)
            logger.info(
                (
                    f"For {dtype}, replace it with {torch_dtype} and "
                    + f"set quant_min: {quant_min}, quant_max: {quant_max}"
                )
            )
        else:  # pragma: no cover
            assert False, "Unsupported dtype with {}".format(dtype)

    if algorithm == "placeholder" or torch_dtype == torch.float:  # pragma: no cover
        return (
            torch.quantization.PlaceholderObserver
            if get_torch_version().release < Version("1.8.0").release
            else torch.quantization.PlaceholderObserver.with_args(dtype=torch_dtype, compute_dtype=compute_dtype)
        )
    if algorithm == "minmax":
        if granularity == "per_channel":
            observer = torch.quantization.PerChannelMinMaxObserver
            if scheme == "sym":
                qscheme = torch.per_channel_symmetric
            elif scheme == "asym_float":
                qscheme = torch.per_channel_affine_float_qparams
            else:
                qscheme = torch.per_channel_affine
        else:
            assert granularity == "per_tensor"
            observer = torch.quantization.MinMaxObserver
            if scheme == "sym":
                qscheme = torch.per_tensor_symmetric
            else:
                assert scheme == "asym"
                qscheme = torch.per_tensor_affine
    else:
        assert algorithm == "kl"
        observer = torch.quantization.HistogramObserver
        assert granularity == "per_tensor"
        if scheme == "sym":
            qscheme = torch.per_tensor_symmetric
        else:
            assert scheme == "asym"
            qscheme = torch.per_tensor_affine

    return observer.with_args(
        qscheme=qscheme,
        dtype=torch_dtype,
        reduce_range=(REDUCE_RANGE and scheme == "asym"),
        quant_min=quant_min,
        quant_max=quant_max,
    )


def _fake_quantize(algorithm, scheme, granularity, dtype, compute_dtype="uint8"):
    """Construct a fake quantize module, In forward, fake quantize module will update
       the statistics of the observed Tensor and fake quantize the input.
       They should also provide a `calculate_qparams` function
       that computes the quantization parameters given the collected statistics.

    Args:
        algorithm (string): What algorithm for computing the quantization parameters based on.
        scheme (string): Quantization scheme to be used.
        granularity (string): What granularity to computing the quantization parameters,
                              per channel or per tensor.
        dtype (string): Quantized data type

    Return:
        fake quantization (object)
    """
    version = get_torch_version()
    if scheme == "asym_float" and version.release >= Version("1.7.0").release:  # pragma: no cover
        return torch.quantization.default_float_qparams_observer
    if algorithm == "placeholder" or dtype == "fp32":  # pragma: no cover
        return _observer(algorithm, scheme, granularity, dtype, compute_dtype=compute_dtype)
    fake_quant = (
        torch.quantization.FakeQuantize
        if version.release < Version("1.10.0").release
        else torch.quantization.FusedMovingAvgObsFakeQuantize
    )
    if algorithm == "minmax":
        if granularity == "per_channel":
            observer = torch.quantization.MovingAveragePerChannelMinMaxObserver
            if scheme == "sym":
                qscheme = torch.per_channel_symmetric
            else:
                assert scheme == "asym"
                qscheme = torch.per_channel_affine
        else:
            assert granularity == "per_tensor"
            observer = torch.quantization.MovingAverageMinMaxObserver
            if scheme == "sym":
                qscheme = torch.per_tensor_symmetric
            else:
                assert scheme == "asym"
                qscheme = torch.per_tensor_affine
    else:  # pragma: no cover
        # Histogram observer is too slow for quantization aware training
        assert algorithm == "kl"
        observer = torch.quantization.HistogramObserver
        assert granularity == "per_tensor"
        if scheme == "sym":
            qscheme = torch.per_tensor_symmetric
        else:
            assert scheme == "asym"
            qscheme = torch.per_tensor_affine

    if dtype == "int8":
        qmin = -128
        qmax = 127
        dtype = torch.qint8
    else:
        assert dtype == "uint8"
        qmin = 0
        qmax = 255
        dtype = torch.quint8

    return fake_quant.with_args(
        observer=observer,
        quant_min=qmin,
        quant_max=qmax,
        dtype=dtype,
        qscheme=qscheme,
        reduce_range=(REDUCE_RANGE and scheme == "asym"),
    )


def _propagate_qconfig(model, op_qcfgs, is_qat_convert=False, approach="post_training_static_quant"):
    """Propagate qconfig through the module hierarchy and assign `qconfig`
       attribute on each leaf module.

    Args:
        model (object): input model
        op_qcfgs (dict): dictionary that maps from name or type of submodule to
                         quantization configuration, qconfig applies to all submodules of a
                         given module unless qconfig for the submodules are specified (when
                         the submodule already has qconfig attribute)
        is_qat_convert (bool): flag that specified this function is used to QAT prepare
                               for pytorch 1.7 or above.
        approach (str): quantization approach
    Return:
        None, module is modified inplace with qconfig attached
    """
    fallback_ops = []
    _propagate_qconfig_recursively(model, "", op_qcfgs)

    if approach != "post_training_dynamic_quant":
        for k, v in op_qcfgs.items():
            if v is None and not is_qat_convert:
                fallback_ops.append(k)

        if fallback_ops and not is_qat_convert:
            _fallback_quantizable_ops_recursively(model, "", fallback_ops, op_qcfgs)


def _propagate_qconfig_recursively(model, prefix, op_qcfgs, qconfig_parent=None):
    """This is a helper function for `propagate_qconfig`

    Args:
        model (object): input model
        prefix (string): prefix of op name
        op_qcfgs (dict): dictionary that maps from name or type of submodule to
                        quantization configuration
        qconfig_parent (object, optional): qconfig of parent module

    Returns:
        None
    """
    for name, child in model.named_children():
        op_name = prefix + name
        child.qconfig = qconfig_parent
        qconfig_son = None
        if op_name in op_qcfgs:
            child.qconfig = op_qcfgs[op_name]
            # for submodules of fused module, like nn.ConvBnRelu2d.
            qconfig_son = child.qconfig
        elif type(child) == torch.quantization.DeQuantStub:
            version = get_torch_version()
            if version.release >= Version("1.8.0").release:
                child.qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.MinMaxObserver.with_args(reduce_range=REDUCE_RANGE),
                    weight=torch.quantization.default_per_channel_weight_observer,
                )
        _propagate_qconfig_recursively(child, op_name + ".", op_qcfgs, qconfig_son)


def _find_quantized_op_num(module, op_qcfgs, prefix="", op_count=0):
    """This is a helper function for `_fallback_quantizable_ops_recursively`

    Args:
        model (object): input model
        op_cfgs (dict): dictionary of quantization configure for each op
        prefix (str): prefix of op name
        op_count (int, optional): count the quantizable op quantity in this module

    Returns:
        the quantizable op quantity in this module
    """
    for name_tmp, child_tmp in module.named_children():
        op_name = prefix + "." + name_tmp if prefix != "" else name_tmp
        if op_name in op_qcfgs.keys() and type(child_tmp) != torch.quantization.QuantStub:
            op_count += 1
        else:
            op_count = _find_quantized_op_num(child_tmp, op_qcfgs, op_name, op_count)
    return op_count


def _fallback_quantizable_ops_recursively(model, prefix, fallback_ops, op_qcfgs):
    """Handle all fallback ops(fp32 ops)

    Args:
        model (object): input model
        prefix (string): the prefix of op name
        fallback_ops (list): list of fallback ops(fp32 ops)
        op_cfgs (dict): dictionary of quantization configure for each op

    Returns:
        None
    """

    class DequantQuantWrapper(torch.nn.Module):
        """A wrapper class that wraps the input module, adds DeQuantStub and
           surround the call to module with call to dequant.
           this is used by fallback layer when the data type of quantized op
           is  input:int8/output:int8.

        This is used by the fallback utility functions to add the dequant and
        quant modules, before `convert` function `QuantStub` will just be observer,
        it observes the input tensor, after `convert`, `QuantStub`
        will be swapped to `nnq.Quantize` which does actual quantization. Similarly
        for `DeQuantStub`.
        """

        def __init__(self, module, observer=None):
            super(DequantQuantWrapper, self).__init__()
            if not module.qconfig and observer:
                weights_observer = observer("minmax", "asym", "per_channel", "int8")
                activation_observer = observer("minmax", "sym", "per_tensor", "uint8")
                module.qconfig = torch.quantization.QConfig(activation=activation_observer, weight=weights_observer)
            self.add_module("quant", torch.quantization.QuantStub(module.qconfig))
            self.add_module("dequant", torch.quantization.DeQuantStub())
            self.add_module("module", module)
            version = get_torch_version()
            if version.release >= Version("1.8.0").release:
                self.dequant.qconfig = module.qconfig
            module.qconfig = None
            self.train(module.training)

        def forward(self, X):
            X = self.dequant(X)
            X = self.module(X)
            return self.quant(X)

        def add(self, x, y):
            # type: (Tensor, Tensor) -> Tensor
            x = self.dequant(x)
            y = self.dequant(y)
            r = self.module.add(x, y)
            return self.quant(r)

        def add_scalar(self, x, y):
            # type: (Tensor, float) -> Tensor
            x = self.dequant(x)
            r = self.module.add_scalar(x, y)
            return self.quant(r)

        def mul(self, x, y):
            # type: (Tensor, Tensor) -> Tensor
            x = self.dequant(x)
            y = self.dequant(y)
            r = self.module.mul(x, y)
            return self.quant(r)

        def mul_scalar(self, x, y):
            # type: (Tensor, float) -> Tensor
            x = self.dequant(x)
            r = self.module.mul_scalar(x, y)
            return self.quant(r)

        def cat(self, x, dim=0):
            # type: (List[Tensor], int) -> Tensor
            X = [self.dequant(x_) for x_ in x]
            r = self.module.cat(X, dim)
            return self.quant(r)

        def add_relu(self, x, y):
            # type: (Tensor, Tensor) -> Tensor
            x = self.dequant(x)
            y = self.dequant(y)
            r = self.module.add_relu(x, y)
            return self.quant(r)

    for name, child in model.named_children():
        op_name = prefix + "." + name if prefix != "" else name
        if op_name in fallback_ops:
            child.qconfig = None
            quantize_op_num = _find_quantized_op_num(model, op_qcfgs, prefix=prefix)
            if quantize_op_num == 1:
                found = False
                for name_tmp, child_tmp in model.named_children():
                    if isinstance(child_tmp, torch.quantization.QuantStub) or isinstance(
                        child_tmp, torch.quantization.DeQuantStub
                    ):
                        model._modules[name_tmp] = torch.nn.Identity()
                        found = True
                if not found:
                    model._modules[name] = DequantQuantWrapper(child, observer=_observer)
            else:
                model._modules[name] = DequantQuantWrapper(child, observer=_observer)
        else:
            _fallback_quantizable_ops_recursively(child, op_name, fallback_ops, op_qcfgs)


@adaptor_registry
class TemplateAdaptor(Adaptor):
    """Tample adaptor of PyTorch framework.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """

    def __init__(self, framework_specific_info):
        super(TemplateAdaptor, self).__init__(framework_specific_info)
        import torch.quantization as tq

        self.version = get_torch_version()
        # set torch random seed
        random_seed = framework_specific_info["random_seed"]
        torch.manual_seed(random_seed)

        self.bf16_ops = []
        self.use_bf16 = framework_specific_info.get("use_bf16", True)
        self.device = framework_specific_info["device"]
        self.backend = framework_specific_info.get("backend", "default")
        self.q_dataloader = framework_specific_info["q_dataloader"]
        self.q_func = framework_specific_info.get("q_func", None)
        self.benchmark = GLOBAL_STATE.STATE == MODE.BENCHMARK
        self.workspace_path = framework_specific_info["workspace_path"]
        self.is_baseline = False if GLOBAL_STATE.STATE == MODE.BENCHMARK else True
        self.query_handler = None
        self.approach = ""
        self.pre_optimized_model = None
        self.sub_module_list = None
        self.default_qconfig = framework_specific_info.get("default_qconfig", None)
        self.performance_only = framework_specific_info.get("performance_only", False)
        self.example_inputs = framework_specific_info.get("example_inputs", None)
        if isinstance(self.example_inputs, (list, tuple)):
            self.example_inputs = tuple(self.example_inputs)
        elif isinstance(self.example_inputs, (dict, UserDict)):
            self.example_inputs = dict(self.example_inputs)
        if "recipes" in framework_specific_info:
            self.recipes = framework_specific_info["recipes"]
        else:
            self.recipes = None

        if "approach" in framework_specific_info:  # pragma: no cover
            self.approach = framework_specific_info["approach"]
            if framework_specific_info["approach"] in ["post_training_static_quant", "post_training_auto_quant"]:
                if self.version.release < Version("1.7.0").release:  # pragma: no cover
                    self.q_mapping = tq.default_mappings.DEFAULT_MODULE_MAPPING
                elif self.version.release < Version("1.8.0").release:  # pragma: no cover
                    self.q_mapping = tq.quantization_mappings.get_static_quant_module_mappings()
                else:
                    self.q_mapping = tq.quantization_mappings.get_default_static_quant_module_mappings()
            elif framework_specific_info["approach"] == "quant_aware_training":
                if self.version.release < Version("1.7.0").release:  # pragma: no cover
                    self.q_mapping = tq.default_mappings.DEFAULT_QAT_MODULE_MAPPING
                elif self.version.release < Version("1.8.0").release:  # pragma: no cover
                    self.q_mapping = tq.quantization_mappings.get_qat_module_mappings()
                else:
                    self.q_mapping = tq.quantization_mappings.get_default_qat_module_mappings()
            elif framework_specific_info["approach"] == "post_training_dynamic_quant":
                if self.version.release < Version("1.7.0").release:
                    self.q_mapping = tq.default_mappings.DEFAULT_DYNAMIC_MODULE_MAPPING
                elif self.version.release < Version("1.8.0").release:
                    self.q_mapping = tq.quantization_mappings.get_dynamic_quant_module_mappings()
                else:
                    self.q_mapping = tq.quantization_mappings.get_default_dynamic_quant_module_mappings()
            elif framework_specific_info["approach"] == "post_training_weight_only":
                pass
            else:
                if not self.benchmark:
                    assert False, "Unsupported approach: {}".format(self.approach)

        # TODO: will be removed once 'op_type_dict' and 'op_name_dicts'
        # for quant_aware_training can be handled in strategy
        self.qat_optype_wise = framework_specific_info.get("qat_optype_wise", None)
        self.qat_op_wise = framework_specific_info.get("qat_op_wise", None)

        self.fp32_results = []
        self.fp32_preds_as_label = False

        if self.version.release >= Version("1.8").release:
            static_quant_mapping = tq.quantization_mappings.get_default_static_quant_module_mappings()
            self.fused_op_list = [static_quant_mapping[key] for key in static_quant_mapping if "intrinsic." in str(key)]
        self.fused_dict = {}

    def calib_func(self, model, dataloader, tmp_iterations, conf=None):
        try:
            for idx, (input, label) in enumerate(dataloader):
                output = pytorch_forward_wrapper(
                    model, input, backend=self.backend, conf=conf, running_mode="calibration"
                )
                if idx >= tmp_iterations - 1:
                    break
        except Exception as e:
            for idx, input in enumerate(dataloader):
                output = pytorch_forward_wrapper(
                    model, input, backend=self.backend, conf=conf, running_mode="calibration"
                )
                if idx >= tmp_iterations - 1:
                    break

    def model_calibration(self, q_model, dataloader, iterations=1, conf=None, calib_sampling_size=1):
        assert iterations > 0
        with torch.no_grad():
            if isinstance(dataloader, BaseDataLoader):
                batch_size = dataloader.batch_size
                try:
                    for i in range(batch_size):
                        if calib_sampling_size % (batch_size - i) == 0:
                            calib_batch_size = batch_size - i
                            if i != 0:
                                logger.warning(
                                    "Reset `calibration.dataloader.batch_size` field "
                                    "to {}".format(calib_batch_size) + " to make sure the sampling_size is "
                                    "divisible exactly by batch size"
                                )
                            break
                    tmp_iterations = int(math.ceil(calib_sampling_size / calib_batch_size))
                    dataloader.batch(calib_batch_size)
                    self.calib_func(q_model, dataloader, tmp_iterations, conf)
                except Exception:  # pragma: no cover
                    logger.warning("Fail to forward with batch size={}, set to {} now.".format(batch_size, 1))
                    dataloader.batch(1)
                    self.calib_func(q_model, dataloader, calib_sampling_size, conf)
            else:  # pragma: no cover
                dataloader_batch_size = getattr(dataloader, "batch_size") or getattr(dataloader, "total_batch_size")
                if hasattr(dataloader, "batch_size") and calib_sampling_size % dataloader_batch_size != 0:
                    logger.warning(
                        "Please note that calibration sampling size {} "
                        "isn't divisible exactly by batch size {}. "
                        "So the real sampling size is {}.".format(
                            calib_sampling_size, dataloader_batch_size, dataloader_batch_size * iterations
                        )
                    )

                self.calib_func(q_model, dataloader, iterations, conf)

    def eval_func(self, model, dataloader, postprocess, metrics, measurer, iteration, conf=None):
        results = []
        try:
            for idx, (input, label) in enumerate(dataloader):
                if measurer is not None:
                    measurer.start()

                output = pytorch_forward_wrapper(model, input, backend=self.backend, conf=conf)
                if self.device != "cpu":  # pragma: no cover
                    output = output.to("cpu")
                    label = label.to("cpu")
                if measurer is not None:
                    measurer.end()
                if postprocess is not None:
                    output, label = postprocess((output, label))
                if metrics:
                    for metric in metrics:
                        if not hasattr(metric, "compare_label") or (
                            hasattr(metric, "compare_label") and metric.compare_label
                        ):
                            metric.update(output, label)

                    # If distributed dataloader, gather all outputs to update metric
                    if getattr(dataloader, "distributed", False) or isinstance(
                        dataloader.sampler, torch.utils.data.distributed.DistributedSampler
                    ):
                        hvd.init()
                        for metric in metrics:
                            metric.hvd = hvd

                if self.fp32_preds_as_label:
                    self.fp32_results.append(output) if self.is_baseline else results.append(output)
                if idx + 1 == iteration:
                    break
        except Exception as e:  # pragma: no cover
            logger.warning("The dataloader didn't include label, will try input without label!")
            for idx, input in enumerate(dataloader):
                if isinstance(input, dict) or isinstance(input, UserDict):
                    if not self.benchmark:
                        assert (
                            "label" in input or "labels" in input
                        ), "The dataloader must include label to measure the metric!"
                        label = input["label"].to("cpu") if "label" in input else input["labels"].to("cpu")
                elif not self.benchmark:
                    assert False, "The dataloader must include label to measure the metric!"

                if measurer is not None:
                    measurer.start()

                output = pytorch_forward_wrapper(model, input, backend=self.backend, conf=conf)

                if measurer is not None:
                    measurer.end()

                if self.device != "cpu" and not self.benchmark:  # pragma: no cover
                    if isinstance(output, dict) or isinstance(input, UserDict):
                        for key in output:
                            output[key] = output[key].to("cpu")
                    elif isinstance(output, list) or isinstance(output, tuple):
                        for tensor in output:
                            tensor = tensor.to("cpu")
                    else:
                        output = output.to("cpu")

                if postprocess is not None and not self.benchmark:
                    output, label = postprocess((output, label))

                if metrics and not self.benchmark:
                    for metric in metrics:
                        if not hasattr(metric, "compare_label") or (
                            hasattr(metric, "compare_label") and metric.compare_label
                        ):
                            metric.update(output, label)

                    # If distributed dataloader, gather all outputs to update metric
                    if getattr(dataloader, "distributed", False) or isinstance(
                        dataloader.sampler, torch.utils.data.distributed.DistributedSampler
                    ):
                        hvd.init()
                        for metric in metrics:
                            metric.hvd = hvd

                if self.fp32_preds_as_label:
                    self.fp32_results.append(output) if self.is_baseline else results.append(output)
                if idx + 1 == iteration:
                    break
        return results

    def model_eval(self, model, dataloader, postprocess=None, metrics=None, measurer=None, iteration=-1, conf=None):
        with torch.no_grad():
            if metrics:
                for metric in metrics:
                    metric.reset()
            if isinstance(dataloader, BaseDataLoader) and not self.benchmark:
                try:
                    results = self.eval_func(model, dataloader, postprocess, metrics, measurer, iteration, conf)
                except Exception:  # pragma: no cover
                    logger.warning(
                        "Fail to forward with batch size={}, set to {} now.".format(dataloader.batch_size, 1)
                    )
                    dataloader.batch(1)
                    results = self.eval_func(model, dataloader, postprocess, metrics, measurer, iteration, conf)
            else:  # pragma: no cover
                results = self.eval_func(model, dataloader, postprocess, metrics, measurer, iteration, conf)

        if self.fp32_preds_as_label:
            if self.is_baseline:
                results = torch_utils.util.collate_torch_preds(self.fp32_results)
                reference = results
            else:
                reference = torch_utils.util.collate_torch_preds(self.fp32_results)
                results = torch_utils.util.collate_torch_preds(results)
            for metric in metrics:
                if hasattr(metric, "compare_label") and not metric.compare_label:
                    metric.update(results, reference)

        acc = 0 if metrics is None else [metric.result() for metric in metrics]
        return acc if not isinstance(acc, list) or len(acc) > 1 else acc[0]

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """

        raise NotImplementedError

    def _get_quantizable_ops(self, model):
        """This is a helper function to get all quantizable ops from model.

        Args:
            model (object): input model which is PyTorch model

        Returns:
            q_capability (dictionary): tuning capability for each op from model.
        """
        quantizable_ops = []
        self.block_wise = []
        self._get_quantizable_ops_recursively(model, "", quantizable_ops)
        q_capability = {}
        q_capability["block_wise"] = None
        q_capability["optypewise"] = OrderedDict()
        q_capability["opwise"] = OrderedDict()
        # add block ops
        if self.block_wise:
            logger.debug(f"*** Found {len(self.block_wise)} blocks: {self.block_wise}")
        q_capability["block_wise"] = self.block_wise[::-1] if self.block_wise else None

        quant_datatypes = self.query_handler.get_quant_datatypes()
        if self.approach == "quant_aware_training":
            capability_pair = [(self.query_handler.get_quantization_capability()["quant_aware"], "static")]
            fp32_config = {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}}
            # Ignore LayerNorm, InstanceNorm3d and Embedding quantizable ops,
            # due to huge accuracy regression in PyTorch.
            if isinstance(self, PyTorch_IPEXAdaptor):
                additional_skipped_module_classes = {}
            else:
                additional_skipped_module_classes = {"LayerNorm", "InstanceNorm3d", "Dropout"}
            no_fp32_ops = {"QuantStub"}
            for pair in capability_pair:
                capability, mode = pair
                for q_op in quantizable_ops:
                    if q_op not in q_capability["opwise"]:
                        q_capability["opwise"][q_op] = []
                    if q_op[1] not in q_capability["optypewise"]:
                        q_capability["optypewise"][q_op[1]] = []

                    op_cfg = (
                        copy.deepcopy(capability[q_op[1]])
                        if q_op[1] in capability
                        else copy.deepcopy(capability["default"])
                    )

                    op_cfg["activation"]["quant_mode"] = (
                        mode if q_op[1] not in ["LSTM", "GRU", "LSTMCell", "GRUCell", "RNNCell"] else "dynamic"
                    )

                    # skip the op that only include fp32
                    if q_op[1] not in additional_skipped_module_classes:
                        if op_cfg not in q_capability["opwise"][q_op]:
                            q_capability["opwise"][q_op].append(op_cfg)
                        if op_cfg not in q_capability["optypewise"][q_op[1]]:
                            q_capability["optypewise"][q_op[1]].append(op_cfg)

                    if q_op[1] not in no_fp32_ops:
                        if fp32_config not in q_capability["opwise"][q_op]:
                            q_capability["opwise"][q_op].append(fp32_config)
                        if fp32_config not in q_capability["optypewise"][q_op[1]]:
                            q_capability["optypewise"][q_op[1]].append(fp32_config)
        elif self.approach == "post_training_weight_only":
            capability_pair = [(self.query_handler.get_quantization_capability("weight_only_integer"), "weight_only")]
            fp32_config = {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}}
            for pair in capability_pair:
                capability, mode = pair
                for q_op in quantizable_ops:
                    if q_op not in q_capability["opwise"]:
                        q_capability["opwise"][q_op] = []
                    if q_op[1] not in q_capability["optypewise"]:
                        q_capability["optypewise"][q_op[1]] = []
                    op_cfg = (
                        copy.deepcopy(capability[q_op[1]])
                        if q_op[1] in capability
                        else copy.deepcopy(capability["default"])
                    )
                    op_cfg["activation"]["quant_mode"] = mode
                    if op_cfg not in q_capability["opwise"][q_op]:
                        q_capability["opwise"][q_op].append(op_cfg)
                        q_capability["opwise"][q_op].append(fp32_config)
                    if op_cfg not in q_capability["optypewise"][q_op[1]]:
                        q_capability["optypewise"][q_op[1]].append(op_cfg)
                        q_capability["optypewise"][q_op[1]].append(fp32_config)
        else:
            if "weight_only_integer" in quant_datatypes:  # TODO: need to enhance
                quant_datatypes.remove("weight_only_integer")
            for datatype in quant_datatypes:
                if self.approach == "post_training_dynamic_quant":
                    capability_pair = [
                        (self.query_handler.get_quantization_capability(datatype).get("dynamic", {}), "dynamic")
                    ]
                elif self.approach == "post_training_static_quant":
                    capability_pair = [
                        (self.query_handler.get_quantization_capability(datatype).get("static", {}), "static")
                    ]
                else:
                    capability_pair = [
                        (self.query_handler.get_quantization_capability(datatype).get("static", {}), "static"),
                        (self.query_handler.get_quantization_capability(datatype).get("dynamic", {}), "dynamic"),
                    ]

                fp32_config = {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}}
                # Ignore LayerNorm, InstanceNorm3d and Embedding quantizable ops,
                # due to huge accuracy regression in PyTorch.
                if isinstance(self, PyTorch_IPEXAdaptor):
                    additional_skipped_module_classes = {}
                else:
                    additional_skipped_module_classes = {"LayerNorm", "InstanceNorm3d", "Dropout"}
                no_fp32_ops = {"QuantStub"}
                for pair in capability_pair:
                    capability, mode = pair
                    for q_op in quantizable_ops:
                        op_cfg = None
                        if q_op not in q_capability["opwise"]:
                            q_capability["opwise"][q_op] = []
                        if q_op[1] not in q_capability["optypewise"]:
                            q_capability["optypewise"][q_op[1]] = []

                        if mode == "static" and q_op[1] in ["LSTM", "GRU", "LSTMCell", "GRUCell", "RNNCell"]:
                            continue

                        op_cfg = (
                            copy.deepcopy(capability[q_op[1]])
                            if q_op[1] in capability
                            else copy.deepcopy(capability.get("default", fp32_config))
                        )

                        op_cfg["activation"]["quant_mode"] = (
                            mode if q_op[1] not in ["LSTM", "GRU", "LSTMCell", "GRUCell", "RNNCell"] else "dynamic"
                        )

                        # skip the op that only include fp32
                        if q_op[1] not in additional_skipped_module_classes:
                            if op_cfg not in q_capability["opwise"][q_op]:
                                q_capability["opwise"][q_op].append(op_cfg)
                            if op_cfg not in q_capability["optypewise"][q_op[1]]:
                                q_capability["optypewise"][q_op[1]].append(op_cfg)

                        if q_op[1] not in no_fp32_ops:
                            if fp32_config not in q_capability["opwise"][q_op]:
                                q_capability["opwise"][q_op].append(fp32_config)
                            if fp32_config not in q_capability["optypewise"][q_op[1]]:
                                q_capability["optypewise"][q_op[1]].append(fp32_config)

        # get bf16 capability
        if (
            self.use_bf16
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
            and (self.version.release >= Version("1.11.0").release)
            and self.approach != "post_training_weight_only"
        ):
            self.bf16_ops = self.query_handler.get_op_types_by_precision("bf16")
            bf16_ops = []
            self._get_bf16_ops_recursively(model, "", bf16_ops)
            mixed_capability = self._combine_capability(bf16_ops, q_capability)
            return mixed_capability
        return q_capability

    def _get_bf16_ops_recursively(self, model, prefix, bf16_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            bf16_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """

        for name, child in model.named_children():
            op_name = prefix + "." + name if prefix != "" else name
            if (
                str(child.__class__.__name__) in self.bf16_ops
                and type(child) != torch.nn.Sequential
                and type(child) != torch.quantization.stubs.DeQuantStub
            ):
                bf16_ops.append(
                    (
                        op_name,
                        (
                            unify_op_type_mapping[str(child.__class__.__name__)]
                            if str(child.__class__.__name__) in unify_op_type_mapping
                            else str(child.__class__.__name__)
                        ),
                    )
                )
            elif self.is_fused_module(child):
                continue
            else:
                self._get_bf16_ops_recursively(child, op_name, bf16_ops)

    def _combine_capability(self, bf16_ops, q_capability):
        bf16_config = {"activation": {"dtype": "bf16"}, "weight": {"dtype": "bf16"}}
        fp32_config = {"activation": {"dtype": "fp32"}, "weight": {"dtype": "fp32"}}
        for bf16_op in bf16_ops:
            if bf16_op in q_capability["opwise"] and bf16_config not in q_capability["opwise"][bf16_op]:
                q_capability["opwise"][bf16_op].append(bf16_config)
            else:
                q_capability["opwise"][bf16_op] = [bf16_config, fp32_config]
                if bf16_op[1] not in q_capability["optypewise"]:
                    q_capability["optypewise"][bf16_op[1]] = [bf16_config, fp32_config]
            if bf16_op[1] in q_capability["optypewise"] and bf16_config not in q_capability["optypewise"][bf16_op[1]]:
                q_capability["optypewise"][bf16_op[1]].append(bf16_config)
        return q_capability

    def get_fused_list(self, model):
        """This is a helper function to get fused op list.

        Args:
            model (object): input model

        Returns:
            dict of op list
        """
        fused_dict = {}
        for op_name, child in model.named_modules():
            if type(child) in self.fused_op_list:
                in_fused_loop = False
                is_fused_module = False
                type_name = str(child).split("(")[0]
                prefix_index = op_name.rfind(".")
                fp32_int8_ops = []
                for fp32_op_name, module in self.pre_optimized_model.model.named_modules():
                    fp32_type_name = str(module).split("(")[0]
                    prefix_fp32_index = fp32_op_name.rfind(".")
                    if not is_fused_module:
                        is_fused_module = self.is_fused_module(module)
                        if is_fused_module:
                            in_fused_loop = True
                            continue
                    if is_fused_module and in_fused_loop:
                        if op_name == fp32_op_name[: fp32_op_name.rfind(".")]:
                            fp32_int8_ops.append(fp32_op_name)
                            continue
                        else:
                            is_fused_module = False
                            in_fused_loop = False
                    elif op_name == fp32_op_name and not in_fused_loop:
                        in_fused_loop = True
                        fp32_int8_ops.append(fp32_op_name)
                    elif (
                        in_fused_loop
                        and op_name[: prefix_index if prefix_index > -1 else 0]
                        == fp32_op_name[: prefix_fp32_index if prefix_fp32_index > -1 else 0]
                    ):
                        if "BatchNorm" in str(type(module)):
                            fp32_int8_ops.append(fp32_op_name)
                            continue
                        elif fp32_type_name in type_name.split(".")[-1][-len(fp32_type_name) - 2 :]:
                            fp32_int8_ops.append(fp32_op_name)
                            in_fused_loop = False
                            break
                        else:
                            in_fused_loop = False
                            break
                    elif in_fused_loop:
                        in_fused_loop = False
                        break
                if len(fp32_int8_ops) > 1:
                    fused_dict.update({op_name: fp32_int8_ops})
        return fused_dict

    def inspect_tensor(
        self,
        model,
        dataloader,
        op_list=None,
        iteration_list=None,
        inspect_type="activation",
        save_to_disk=False,
        save_path=None,
        quantization_cfg=None,
    ):
        assert self.version.release >= Version("1.8").release, "Inspect_tensor only support torch 1.8 or above!"
        from torch import dequantize

        from neural_compressor.utils.utility import dump_data_to_local

        is_quantized = model.is_quantized
        op_list_ = []
        fp32_int8_map = {}
        for op_name in op_list:
            op_list_.append(op_name)
            for key in self.fused_dict:
                if op_name in self.fused_dict[key]:
                    op_list_.pop()
                    fp32_int8_map[op_name] = {"activation": self.fused_dict[key][-1], "weight": self.fused_dict[key][0]}
                    if not is_quantized:
                        op_list_.append(self.fused_dict[key][-1])
                    elif key not in op_list_:
                        op_list_.append(key)
                    break

        assert min(iteration_list) > 0, "Iteration number should great zero, 1 means first iteration."
        iterations = max(iteration_list) if iteration_list is not None else -1
        new_model = self._pre_eval_hook(model, op_list=op_list_, iteration_list=iteration_list)
        self.evaluate(new_model, dataloader, iteration=iterations)
        observer_dict = {}
        ret = {}
        if inspect_type == "activation" or inspect_type == "all":
            if self.version.release >= Version("2.0.0").release:
                from torch.quantization.quantize import _get_observer_dict as get_observer_dict
            else:
                from torch.quantization import get_observer_dict
            ret["activation"] = []
            get_observer_dict(new_model.model, observer_dict)
            if iteration_list is None:
                iteration_list = [1]
            for i in iteration_list:
                summary = OrderedDict()
                for key in observer_dict:
                    if isinstance(observer_dict[key], torch.nn.modules.linear.Identity):
                        continue
                    op_name = key.replace(".activation_post_process", "")
                    if len(observer_dict[key].get_tensor_value()) == 0:
                        continue
                    value = observer_dict[key].get_tensor_value()[i]
                    if op_name in op_list:
                        if type(value) is list:
                            summary[op_name] = {}
                            for index in range(len(value)):
                                summary[op_name].update(
                                    {
                                        op_name
                                        + ".output"
                                        + str(index): (
                                            dequantize(value[index]).numpy()
                                            if value[index].is_quantized
                                            else value[index].numpy()
                                        )
                                    }
                                )
                        else:
                            summary[op_name] = {
                                op_name + ".output0": dequantize(value).numpy() if value.is_quantized else value.numpy()
                            }
                    else:
                        if bool(self.fused_dict):
                            if is_quantized:
                                for a in fp32_int8_map:
                                    if op_name == a:
                                        tensor_name = fp32_int8_map[a]["weight"]
                                        if type(value) is list:
                                            summary[tensor_name] = {}
                                            for index in range(len(value)):
                                                summary[tensor_name].update(
                                                    {
                                                        tensor_name
                                                        + ".output"
                                                        + str(index): (
                                                            dequantize(value[index]).numpy()
                                                            if value[index].is_quantized
                                                            else value[index].numpy()
                                                        )
                                                    }
                                                )
                                        else:
                                            summary[tensor_name] = {
                                                tensor_name
                                                + ".output0": (
                                                    dequantize(value).numpy() if value.is_quantized else value.numpy()
                                                )
                                            }
                            else:
                                for a in fp32_int8_map:  # pragma: no cover
                                    if op_name == fp32_int8_map[a]["activation"]:
                                        tensor_name = fp32_int8_map[a]["weight"]
                                        if type(value) is list:
                                            summary[tensor_name] = {}
                                            for index in range(len(value)):
                                                summary[tensor_name].update(
                                                    {
                                                        tensor_name
                                                        + ".output"
                                                        + str(index): (
                                                            dequantize(value[index]).numpy()
                                                            if value[index].is_quantized
                                                            else value[index].numpy()
                                                        )
                                                    }
                                                )
                                        else:
                                            summary[tensor_name] = {
                                                tensor_name
                                                + ".output0": (
                                                    dequantize(value).numpy() if value.is_quantized else value.numpy()
                                                )
                                            }

                ret["activation"].append(summary)

        if inspect_type == "weight" or inspect_type == "all":
            ret["weight"] = {}
            state_dict = new_model._model.state_dict()

            for key in state_dict:
                if not isinstance(state_dict[key], torch.Tensor):
                    continue
                if "weight" not in key and "bias" not in key:
                    continue

                op = key[: key.rfind(".")]
                op = op.replace("._packed_params", "")

                if op in op_list:
                    if op in ret["weight"]:
                        ret["weight"][op].update(
                            {
                                key: (
                                    dequantize(state_dict[key]).numpy()
                                    if state_dict[key].is_quantized
                                    else state_dict[key].detach().numpy()
                                )
                            }
                        )
                    else:
                        ret["weight"][op] = {
                            key: (
                                dequantize(state_dict[key]).numpy()
                                if state_dict[key].is_quantized
                                else state_dict[key].detach().numpy()
                            )
                        }
                else:
                    if bool(self.fused_dict):
                        if is_quantized:
                            for a in fp32_int8_map:
                                if op == a:
                                    tensor_name = fp32_int8_map[a]["weight"]
                                    if tensor_name in ret["weight"]:
                                        ret["weight"][tensor_name].update(
                                            {
                                                key: (
                                                    dequantize(state_dict[key]).numpy()
                                                    if state_dict[key].is_quantized
                                                    else state_dict[key].detach().numpy()
                                                )
                                            }
                                        )
                                    else:
                                        ret["weight"][tensor_name] = {
                                            key: (
                                                dequantize(state_dict[key]).numpy()
                                                if state_dict[key].is_quantized
                                                else state_dict[key].detach().numpy()
                                            )
                                        }
                                    break
        else:
            ret["weight"] = None

        if save_to_disk:
            if not save_path:
                save_path = self.workspace_path
            dump_data_to_local(ret, save_path, "inspect_result.pkl")

        return ret

    def _pre_eval_hook(self, model, op_list=None, iteration_list=None):
        """The function is used to do some preprocession before evaluation phase.
           Here, it used to add hook for dump output tensor for quantizable ops.

        Args:
             model (object): input model

        Returns:
              model (object): model with hook
        """
        from abc import ABCMeta

        def _with_args(cls_or_self, **kwargs):
            r"""Wrapper that allows creation of class factories.

            This can be useful when there is a need to create classes with the same
            constructor arguments, but different instances.

            Example::

                >>> Foo.with_args = classmethod(_with_args)
                >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
                >>> foo_instance1 = foo_builder()
                >>> foo_instance2 = foo_builder()
                >>> id(foo_instance1) == id(foo_instance2)
                False
            """

            class _PartialWrapper(object):
                def __init__(self, p):
                    self.p = p

                def __call__(self, *args, **keywords):
                    return self.p(*args, **keywords)

                def __repr__(self):
                    return self.p.__repr__()

                with_args = _with_args

            r = _PartialWrapper(partial(cls_or_self, **kwargs))
            return r

        ABC = ABCMeta(str("ABC"), (object,), {})  # compatible with Python 2 *and* 3:

        class _RecordingObserver(ABC, torch.nn.Module):
            """The module is mainly for debug and records the tensor values during runtime.

            Args:
                iteration_list (list, optional): indexes of iteration which to dump tensor.
            """

            def __init__(self, iteration_list=None, **kwargs):
                super(_RecordingObserver, self).__init__(**kwargs)
                self.output_tensors_dict = OrderedDict()
                self.current_iter = 1
                self.iteration_list = iteration_list

            def forward(self, x):
                if (self.iteration_list is None and self.current_iter == 1) or (
                    self.iteration_list is not None and self.current_iter in self.iteration_list
                ):
                    if type(x) is tuple or type(x) is list:
                        self.output_tensors_dict[self.current_iter] = [
                            i.to("cpu") if i.device != "cpu" else i.clone() for i in x
                        ]
                    else:
                        self.output_tensors_dict[self.current_iter] = x.to("cpu") if x.device != "cpu" else x.clone()
                self.current_iter += 1
                return x

            @torch.jit.export
            def get_tensor_value(self):
                return self.output_tensors_dict

            with_args = classmethod(_with_args)

        def _observer_forward_hook(module, input, output):
            """Forward hook that calls observer on the output.

            Args:
                module (object): input module
                input (object): module input
                output (object): module output

            Returns:
                module output tensor (object)
            """
            return module.activation_post_process(output)

        def _add_observer_(module, op_list=None, prefix=""):
            """Add observer for the leaf child of the module.

               This function insert observer module to all leaf child module that
               has a valid qconfig attribute.

            Args:
                module (object): input module with qconfig attributes for all the leaf modules that
                                 we want to dump tensor
                op_list (list, optional): list of ops which to be dumped in module
                prefix (string): name of module

            Returns:
                None, module is modified inplace with added observer modules and forward_hooks
            """
            for name, child in module.named_children():
                op_name = name if prefix == "" else prefix + "." + name
                if isinstance(child, torch.nn.quantized.FloatFunctional) and (op_list is None or op_name in op_list):
                    if (
                        hasattr(child, "qconfig")
                        and child.qconfig is not None
                        and (op_list is None or op_name in op_list)
                    ):
                        child.activation_post_process = child.qconfig.activation()
                elif (
                    hasattr(child, "qconfig") and child.qconfig is not None and (op_list is None or op_name in op_list)
                ):
                    # observer and hook will be gone after we swap the module
                    child.add_module("activation_post_process", child.qconfig.activation())
                    child.register_forward_hook(_observer_forward_hook)
                else:
                    _add_observer_(child, op_list, op_name)

        def _propagate_qconfig_helper(
            module, qconfig_dict, white_list=None, qconfig_parent=None, prefix="", fused=False
        ):
            """This is a helper function for `propagate_qconfig_`

            Args:
                module (object): input module
                qconfig_dict (dictionary): dictionary that maps from name of submodule to
                                           quantization configuration
                white_list (list, optional): list of quantizable modules
                qconfig_parent (object, optional): config of parent module, we will fallback to
                                                   this config when there is no specified config
                                                   for current module
                prefix (string, optional): corresponding prefix of the current module,
                                           used as key in qconfig_dict
                fused (bool, optional): Indicates whether the module is fused or not

            Return:
                None, module is modified inplace with qconfig attached
            """
            module.qconfig = qconfig_parent
            if hasattr(module, "_modules"):
                for name, child in module.named_children():
                    module_prefix = prefix + "." + name if prefix else name
                    _propagate_qconfig_helper(child, qconfig_dict, white_list, qconfig_parent, module_prefix)

        def _prepare(model, inplace=True, op_list=[], white_list=None):
            """The model will be attached with observer or fake quant modules, and qconfig
               will be propagated.

            Args:
                model (object): input model to be modified in-place
                inplace (bool, optional): carry out model transformations in-place,
                                          the original module is mutated
                op_list (list, optional): list of ops which to be dumped in module
                white_list (list, optional): list of quantizable modules

            Returns:
                model (object): model with qconfig
            """
            if not inplace:
                model = copy.deepcopy(model)
            _propagate_qconfig_helper(model, qconfig_dict={}, white_list=white_list, qconfig_parent=model.qconfig)
            # sanity check common API misusage
            if not any(hasattr(m, "qconfig") and m.qconfig for m in model.modules()):  # pragma: no cover
                logger.warn(
                    "None of the submodule got qconfig applied. Make sure you "
                    "passed correct configuration through `qconfig_dict` or "
                    "by assigning the `.qconfig` attribute directly on submodules"
                )
            _add_observer_(model, op_list=op_list)
            return model

        model = model if model.is_quantized else copy.deepcopy(model)
        model._model.qconfig = torch.quantization.QConfig(
            weight=torch.quantization.default_debug_observer,
            activation=_RecordingObserver.with_args(iteration_list=iteration_list),
        )
        _prepare(model._model, op_list=op_list)

        return model

    def is_fused_module(self, module):
        """This is a helper function for `_propagate_qconfig_helper` to detect
           if this module is fused.

        Args:
            module (object): input module

        Returns:
            (bool): is fused or not
        """
        op_type = str(type(module))
        if "fused" in op_type:
            return True
        else:
            return False

    def calculate_hessian_trace(self, fp32_model, dataloader, q_model, criterion, enable_act=False):
        """Calculate hessian trace.

        Args:
            fp32_model: The original fp32 model.
            criterion: The loss function for calculate the hessian trace. # loss = criterion(output, target)
            dataloader: The dataloader for calculate the gradient.
            q_model: The INT8 AMAP model.
            enable_act: Enabling quantization error or not.

        Return:
            hessian_trace(Dict[Tuple, float]), key: (op_name, op_type); value: hessian trace.
        """
        from .torch_utils.hawq_metric import hawq_top

        op_to_traces = hawq_top(
            fp32_model=fp32_model, dataloader=dataloader, q_model=q_model, criterion=criterion, enable_act=enable_act
        )
        return op_to_traces

    def smooth_quant(
        self,
        model,
        dataloader,
        calib_iter,
        alpha=0.5,
        folding=False,
        percentile=None,
        op_types=None,
        scales_per_op=None,
        force_re_smooth=False,
        record_max_info=False,
        weight_clip=True,
        auto_alpha_args={
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "alpha_step": 0.1,
            "shared_criterion": "mean",
            "do_blockwise": False,
        },
        default_alpha=0.5,
    ):
        """Convert the model by smooth quant.

        Args:
            model: origin FP32 model
            dataloader: calib dataloader
            calib_iter: calib iters
            alpha: smooth alpha in SmoothQuant, 1.0 will fallback to SPIQ
            folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
            percentile:Percentile of calibration to remove outliers, not supported now
            op_types: The op types whose input tensor will be dumped
            scales_per_op: True, each op will have an individual scale, mainly for accuracy
                           False, ops with the same input will share a scale, mainly for performance
            record_max_info: whether record the max info in model for alpha tuning.
            weight_clip: Whether to clip weight when calculating scales; by default it is on.
            auto_alpha_args: Hyperparameters used to set the alpha search space in SQ auto-tuning.
                            By default the search space is 0.0-1.0 with step_size 0.1.
                            do_blockwise determines whether to do blockwise auto-tuning.
            default_alpha: A hyperparameter that is used in SQ auto-tuning; by default it is 0.5.

        Returns:
            model: A modified fp32 model, inplace=True.
        """
        # Note: we should make sure smoothquant is only executed once with inplacing fp32 model.
        if hasattr(model._model, "_smoothquant_optimized") and model._model._smoothquant_optimized:
            logger.info("The model is already optimized by SmoothQuant algorithm, skip it.")
            return model
        if self.__class__.__name__ == "PyTorch_IPEXAdaptor" and self.version.release < Version("2.1").release:
            if folding is None:
                folding = True
                logger.info("IPEX version >= 2.1 is required for SmoothQuant folding=False, reset folding=True.")
            else:
                assert folding, "IPEX version >= 2.1 is required for SmoothQuant folding=False."

        if not hasattr(self, "sq") or force_re_smooth:
            from neural_compressor.adaptor.torch_utils.waq import TorchSmoothQuant

            self.sq = TorchSmoothQuant(
                model._model, dataloader=dataloader, example_inputs=self.example_inputs, q_func=self.q_func
            )
        kwargs = {}  ## different backends may have different default values
        self.sq.record_max_info = record_max_info  # whether record the max info of input and weight.
        if op_types is not None:
            kwargs["op_types"] = op_types
        if percentile is not None:
            kwargs["percentile"] = percentile
        if scales_per_op is not None:
            kwargs["scales_per_op"] = scales_per_op
        if alpha == "auto":
            auto_alpha_args["init_alpha"] = default_alpha
        model._model = self.sq.transform(
            alpha=alpha,
            folding=folding,
            calib_iter=calib_iter,
            weight_clip=weight_clip,
            auto_alpha_args=auto_alpha_args,
            **kwargs,
        )
        if self.sq.record_max_info:
            model.sq_max_info = self.sq.max_value_info
            model.sq_scale_info = self.sq.sq_scale_info
        return model

    def _apply_pre_optimization(self, model, tune_cfg, recover=False):
        """Update model parameters based on tune_cfg.

        Args:
            model (torch.nn.Module): smoothquant optimized model.
            tune_cfg (dict): optimization config.
            recover (dict): recover pre-optimization change.

        Returns:
            model: pre-optimized model.
        """
        q_model = model._model
        sq_max_info = model.sq_max_info
        if sq_max_info:
            from neural_compressor.adaptor.torch_utils.waq import TorchSmoothQuant

            tsq = TorchSmoothQuant(q_model, None)
            alpha = tune_cfg["recipe_cfgs"]["smooth_quant_args"]["alpha"]
            for op_name, info in sq_max_info.items():
                if alpha == "auto":
                    alpha = info["alpha"]
                absorb_layer = op_name
                absorbed_layer = info["absorbed_layer"]
                input_minmax = info["input_minmax"]
                weight_max = info["weight_max"]
                if self.sq.weight_clip:
                    weight_max = weight_max.clamp(min=1e-5)
                abs_input_max = torch.max(torch.abs(input_minmax[0]), torch.abs(input_minmax[1]))
                input_power = torch.pow(abs_input_max, alpha)
                weight_power = torch.pow(weight_max, 1 - alpha)
                scale = torch.clip(input_power / weight_power, min=1e-5)
                with torch.no_grad():
                    if recover:
                        scale = 1.0 / scale
                    for layer in absorbed_layer:
                        tsq._scale_layer_weight(layer, scale)
                    tsq._absorb_scales(absorb_layer, 1.0 / scale)
                logger.debug(f"Current smoothquant scale of {op_name} is {scale}, alpha is {alpha}")

    def qdq_quantize(self, model, tune_cfg):
        """Insert quant, dequant pairs before linear to simulate quantization.

        Args:
            model (torch.nn.Module): smoothquant optimized model.
            tune_cfg (dict): quantization config.

        Returns:
            model: qdq quantized model.
        """
        q_model = model._model
        from neural_compressor.adaptor.torch_utils.waq import get_module, set_module

        from .torch_utils.model_wrapper import QDQLinear, SQLinearWrapper

        smoothquant_scale_info = {}
        fallback_op_name_list = []
        stats_result = {}
        stats_result["Linear(failed when SQ)"] = {"INT8(QDQ)": 0, "BF16": 0, "FP32": 0}
        for (op_name, op_type), qconfig in tune_cfg["op"].items():
            if op_type == "Linear" and qconfig["weight"]["dtype"] != "int8":
                fallback_op_name_list.append(op_name)

        sq_max_info = model.sq_max_info
        if sq_max_info:
            assert (
                not q_model._smoothquant_optimized
            ), "The model is already optimized by smoothquant, cannot apply new alpha."
            for _, info in sq_max_info.items():
                alpha = info["alpha"]
                absorbed_layer = info["absorbed_layer"]
                input_minmax = info["input_minmax"]
                weight_max = info["weight_max"]
                if self.sq.weight_clip:
                    weight_max = weight_max.clamp(min=1e-5)
                abs_input_max = torch.max(torch.abs(input_minmax[0]), torch.abs(input_minmax[1]))
                input_power = torch.pow(abs_input_max, alpha)
                weight_power = torch.pow(weight_max, 1 - alpha)
                scale = torch.clip(input_power / weight_power, min=1e-5)
                if torch.isnan(scale).any() or torch.isinf(scale).any():
                    stats_result["Linear(failed when SQ)"]["FP32"] += 1
                    continue  # for peft model,lora_B weights is 0.
                for op_name in absorbed_layer:
                    module = get_module(q_model, op_name)
                    new_module = SQLinearWrapper(module, 1.0 / scale, input_minmax, alpha)
                    set_module(q_model, op_name, new_module)
                    logger.debug(f"Current SmoothQuant alpha of {op_name} is {alpha}")

        smoothquant_op_info = {"sq_linear": {}, "qdq_linear": []}
        stats_result["SQLinearWrapper"] = {"INT8(QDQ)": 0, "BF16": 0, "FP32": 0}
        for name, module in q_model.named_modules():
            if isinstance(module, SQLinearWrapper):
                smoothquant_op_info["sq_linear"][name] = module.input_scale
                if name not in fallback_op_name_list:
                    smoothquant_scale_info[name] = {
                        "input_scale_for_mul": module.input_scale,
                        "quant_scale": module.scale,
                        "quant_zero_point": module.zero_point,
                        "quant_dtype": module.dtype,
                    }
                    smoothquant_op_info["qdq_linear"].append(name + ".sq_linear")
                    new_module = QDQLinear(module.sq_linear, module.scale, module.zero_point, module.dtype)
                    set_module(q_model, name + ".sq_linear", new_module)
                    stats_result["SQLinearWrapper"]["INT8(QDQ)"] += 1
                else:
                    stats_result["SQLinearWrapper"]["FP32"] += 1

        tune_cfg["recipe_cfgs"]["smoothquant_op_info"] = smoothquant_op_info
        model._model = q_model
        model.q_config = copy.deepcopy(tune_cfg)
        field_names = ["Op Type", "Total", "INT8", "BF16", "FP32"]
        output_data = [
            [
                op_type,
                sum(stats_result[op_type].values()),
                stats_result[op_type]["INT8(QDQ)"],
                stats_result[op_type]["BF16"],
                stats_result[op_type]["FP32"],
            ]
            for op_type in stats_result.keys()
        ]
        Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()

        return model


unify_op_type_mapping = {
    "ConvReLU2d": "Conv2d",
    "ConvReLU3d": "Conv3d",
    "LinearReLU": "Linear",
    "ConvBn2d": "Conv2d",
    "ConvBnReLU2d": "Conv2d",
}


@adaptor_registry
class PyTorchAdaptor(TemplateAdaptor):
    """Adaptor of PyTorch framework, all PyTorch API is in this class.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """

    def __init__(self, framework_specific_info):
        super(PyTorchAdaptor, self).__init__(framework_specific_info)
        """
        # Map for swapping float module to quantized ones,
        # and this dictionary will change with different PoTorch versions
        DEFAULT_MODULE_MAPPING = {
            nn.Linear: nnq.Linear,
            nn.ReLU: nnq.ReLU,
            nn.ReLU6: nnq.ReLU6,
            nn.Conv2d: nnq.Conv2d,
            nn.Conv3d: nnq.Conv3d,
            QuantStub: nnq.Quantize,
            DeQuantStub: nnq.DeQuantize,
            # Wrapper Modules:
            nnq.FloatFunctional: nnq.QFunctional,
            # Intrinsic modules:
            nni.ConvReLU2d: nniq.ConvReLU2d,
            nni.ConvReLU3d: nniq.ConvReLU3d,
            nni.LinearReLU: nniq.LinearReLU,
            nniqat.ConvReLU2d: nniq.ConvReLU2d,
            nniqat.LinearReLU: nniq.LinearReLU,
            nniqat.ConvBn2d: nnq.Conv2d,
            nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
            # QAT modules:
            nnqat.Linear: nnq.Linear,
            nnqat.Conv2d: nnq.Conv2d,
        }
        """

        self.tune_cfg = None
        if self.device == "cpu":
            query_config_file = "pytorch_cpu.yaml"
        elif self.device == "gpu":
            query_config_file = "pytorch_gpu.yaml"
        else:  # pragma: no cover
            assert False, "Unsupported this device {}".format(self.device)
        self.query_handler = PyTorchQuery(local_config_file=os.path.join(os.path.dirname(__file__), query_config_file))

        self.white_list = get_torch_white_list(self.approach)

        # for tensorboard
        self.dump_times = 0

        self.optype_statistics = None

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization config.
            model (object): model need to do quantization.
            dataloader (object): calibration dataset.
            q_func (objext, optional): training function for quantization aware training mode.

        Returns:
            (object): quantized model
        """
        assert isinstance(model._model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        if self.performance_only:
            q_model = model
        else:
            try:
                q_model = copy.deepcopy(model)
            except Exception as e:  # pragma: no cover
                logger.warning("Fail to deep copy the model due to {}, inplace is used now.".format(repr(e)))
                q_model = model

        # For smoothquant optimized model
        recipe_cfgs = tune_cfg.get("recipe_cfgs", None)
        if (
            recipe_cfgs
            and recipe_cfgs.get("smooth_quant", False)
            and not recipe_cfgs["smooth_quant_args"]["folding"]
            and self.approach != "post_training_dynamic_quant"
        ):
            return self.qdq_quantize(q_model, tune_cfg)

        if recipe_cfgs and recipe_cfgs.get("smooth_quant", False) and recipe_cfgs["smooth_quant_args"]["folding"]:
            self._apply_pre_optimization(q_model, tune_cfg)

        # For tensorboard display
        self.tune_cfg = tune_cfg
        self.tune_cfg["approach"] = self.approach
        self.tune_cfg["reduce_range"] = REDUCE_RANGE
        self.tune_cfg["framework"] = "pytorch"
        op_cfgs = _cfg_to_qconfig(tune_cfg, self.approach)
        self.tune_cfg["bf16_ops_list"] = op_cfgs["bf16_ops_list"]
        del op_cfgs["bf16_ops_list"]
        gc.collect()

        if self.version.release < Version("2.0.0").release:
            from torch.quantization.quantize import add_observer_
        else:
            from torch.quantization.quantize import _add_observer_ as add_observer_

        if self.approach == "quant_aware_training":
            q_model._model.train()
        else:
            q_model._model.eval()
        if self.version.release < Version("1.7.0").release or self.approach != "quant_aware_training":
            _propagate_qconfig(q_model._model, op_cfgs, approach=self.approach)
            # sanity check common API misusage
            if not any(hasattr(m, "qconfig") and m.qconfig for m in q_model._model.modules()):
                logger.warn(
                    "None of the submodule got qconfig applied. Make sure you "
                    "passed correct configuration through `qconfig_dict` or "
                    "by assigning the `.qconfig` attribute directly on submodules."
                )

        if self.approach in ["post_training_static_quant", "post_training_auto_quant"]:
            add_observer_(q_model._model)
            if q_func is None:
                iterations = tune_cfg.get("calib_iteration", 1)
                self.model_calibration(
                    q_model._model, dataloader, iterations, calib_sampling_size=tune_cfg.get("calib_sampling_size", 1)
                )
            else:
                q_func(q_model._model)
        elif self.approach == "quant_aware_training":
            if self.version.release >= Version("1.7.0").release:
                _propagate_qconfig(q_model._model, op_cfgs, is_qat_convert=True)
                torch.quantization.convert(q_model._model, mapping=self.q_mapping, inplace=True, remove_qconfig=False)
                _propagate_qconfig(q_model._model, op_cfgs)
                add_observer_(q_model._model, self.white_list, set(self.q_mapping.values()))
            else:  # pragma: no cover
                add_observer_(q_model._model)
                torch.quantization.convert(q_model._model, self.q_mapping, inplace=True)
            # q_func can be created by neural_compressor internal or passed by user. It's critical to
            # distinguish how q_func is passed since neural_compressor built-in functions accept neural_compressor
            # model and user defined func should accept framework model.
            q_model._model = q_func(q_model if getattr(q_func, "builtin", None) else q_model._model)
            assert q_model._model is not None, "Please return a trained model in train function!"
            q_model._model.eval()

        if self.approach == "quant_aware_training":
            torch.quantization.convert(q_model._model, inplace=True)
        else:
            torch.quantization.convert(q_model._model, mapping=self.q_mapping, inplace=True)

        if (
            len(self.tune_cfg["bf16_ops_list"]) > 0
            and (self.version.release >= Version("1.11.0").release)
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
        ):  # pragma: no cover
            q_model._model = torch_utils.bf16_convert.Convert(q_model._model, self.tune_cfg)

        self.fused_dict = self.get_fused_list(q_model.model)
        q_model.q_config = copy.deepcopy(self.tune_cfg)
        if self.approach != "post_training_dynamic_quant":
            self._get_scale_zeropoint(q_model._model, q_model.q_config)
        q_model.is_quantized = True

        self._dump_model_op_stats(q_model._model, q_model.q_config)
        torch_utils.util.get_embedding_contiguous(q_model._model)
        return q_model

    def evaluate(
        self,
        model,
        dataloader,
        postprocess=None,
        metrics=None,
        measurer=None,
        iteration=-1,
        tensorboard=False,
        fp32_baseline=False,
    ):
        """Execute the evaluate process on the specified model.

        Args:
            model (object): model to run evaluation.
            dataloader (object): evaluation dataset.
            postprocess (object, optional): process function after evaluation.
            metrics (list, optional): list of metric function.
            measurer (object, optional): measurer function.
            iteration (int, optional): number of iterations to evaluate.
            tensorboard (bool, optional): dump output tensor to tensorboard summary files.
            fp32_baseline (boolean, optional): only for compare_label=False pipeline

        Returns:
            (object): accuracy
        """
        self.is_baseline = fp32_baseline
        if tensorboard:
            model = self._pre_eval_hook(model)

        model_ = model._model
        assert isinstance(model_, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        model_.eval()
        if self.device == "cpu":
            model_.to("cpu")
        elif self.device == "gpu":
            if self.is_baseline:
                model_.to("dpcpp")

        if metrics:
            self.fp32_preds_as_label = any(
                [hasattr(metric, "compare_label") and not metric.compare_label for metric in metrics]
            )
        acc = self.model_eval(model_, dataloader, postprocess, metrics, measurer, iteration)

        if tensorboard:
            self._post_eval_hook(model, accuracy=acc)
        return acc if not isinstance(acc, list) or len(acc) > 1 else acc[0]

    def _pre_hook_for_qat(self, dataloader=None):
        # self.model._model is needed here.
        self.model._model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.FakeQuantize.with_args(
                dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=REDUCE_RANGE
            ),
            weight=torch.quantization.default_weight_fake_quant,
        )
        self.non_quant_dict = self.get_non_quant_modules(self.model.kwargs)
        quantizable_ops = []
        self._get_quantizable_ops_recursively(self.model._model, "", quantizable_ops)
        bf16_ops = []
        if (
            self.version.release >= Version("1.11.0").release
            and self.use_bf16
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
        ):  # pragma: no cover
            self.bf16_ops = self.query_handler.get_op_types_by_precision("bf16")
            self._get_bf16_ops_recursively(self.model._model, "", bf16_ops)
        bf16_ops_list = [(op) for op in bf16_ops if op not in quantizable_ops]
        self.model.model.training = True
        torch.quantization.prepare_qat(self.model._model, inplace=True)

        # This is a flag for reloading
        self.model.q_config = {
            "is_oneshot": True,
            "framework": "pytorch",
            "reduce_range": REDUCE_RANGE,
            "approach": "quant_aware_training",
            "bf16_ops_list": bf16_ops_list,
        }

    def _post_hook_for_qat(self):
        torch.quantization.convert(self.model._model, inplace=True)
        if (
            self.model.q_config is not None
            and len(self.model.q_config["bf16_ops_list"]) > 0
            and self.version.release >= Version("1.11.0").release
            and self.use_bf16
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
        ):  # pragma: no cover
            self.model._model = torch_utils.bf16_convert.Convert(self.model._model, self.model.q_config)

    def _pre_hook_for_hvd(self, dataloader=None):
        # TODO: lazy init here
        hvd.init()
        hvd.broadcast_parameters(self.model._model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        self.optimizer = hvd.DistributedOptimizer(self.optimizer, named_parameters=self.model._model.named_parameters())

    def train(self, model, dataloader, optimizer_tuple, criterion_tuple, hooks, **kwargs):
        """Execute the train process on the specified model.

        Args:
            model (object): model to run evaluation.
            dataloader (object): training dataset.
            optimizer (tuple): It is a tuple of (cls, parameters) for optimizer.
            criterion (tuple): It is a tuple of (cls, parameters) for criterion.
            kwargs (dict, optional): other parameters.

        Returns:
            None
        """
        model_ = model._model
        device = "cuda:0" if self.device != "GPU" and torch.cuda.is_available() else self.device
        # self.model is set to neural_compressor model here to hold the inplace change in FWK model.
        self.model = model
        optimizer = optimizer_tuple[0](model_.parameters(), **optimizer_tuple[1])
        self.optimizer = optimizer
        criterion = criterion_tuple[0](**criterion_tuple[1])
        start_epochs = kwargs["kwargs"]["start_epoch"]
        end_epochs = kwargs["kwargs"]["end_epoch"]
        iters = kwargs["kwargs"]["iteration"]
        if hooks is not None:
            on_train_begin = hooks["on_train_begin"]
            on_train_end = hooks["on_train_end"]
            on_epoch_begin = hooks["on_epoch_begin"]
            on_epoch_end = hooks["on_epoch_end"]
            on_step_begin = hooks["on_step_begin"]
            on_step_end = hooks["on_step_end"]
            on_after_compute_loss = hooks["on_after_compute_loss"]
            on_before_optimizer_step = hooks["on_before_optimizer_step"]
        if hooks is not None:
            on_train_begin()
        for nepoch in range(start_epochs, end_epochs):
            model_.to(device)
            model_.train()
            cnt = 0
            if hooks is not None:
                on_epoch_begin(nepoch)
            if getattr(dataloader, "distributed", False) or isinstance(
                dataloader.sampler, torch.utils.data.distributed.DistributedSampler
            ):
                dataloader.sampler.set_epoch(nepoch)
            for image, target in dataloader:
                # TODO: to support adjust lr with epoch
                target = target.to(device)
                if hooks is not None:
                    on_step_begin(cnt)
                print(".", end="", flush=True)
                cnt += 1
                output = pytorch_forward_wrapper(model_, image)
                loss = criterion(output, target)
                if hooks is not None:
                    loss = on_after_compute_loss(image, output, loss)
                self.optimizer.zero_grad()
                loss.backward()
                if hooks is not None:
                    on_before_optimizer_step()
                self.optimizer.step()
                if hooks is not None:
                    on_step_end()
                if cnt >= iters:
                    break
            if hooks is not None:
                on_epoch_end()

        if device != self.device:  # pragma: no cover
            model_.to(self.device)

        if hooks is not None:
            on_train_end()

        return model_

    def _dump_model_op_stats(self, model, tune_cfg):
        """This is a function to dump quantizable ops of model to user.

        Args:
            model (object): input model
            tune_cfg (dict): quantization config
        Returns:
            None
        """
        res = {}
        ignore_log = False
        modules = dict(model.named_modules())
        # fetch quantizable ops supported in Neural Compressor from tune_cfg
        for key in tune_cfg["op"]:
            op_name = key[0]
            op_type = str(type(modules[op_name])).rstrip("'>").split(".")[-1]
            if op_type == "BF16ModuleWrapper":  # pragma: no cover
                op_type = str(type(modules[op_name].module)).rstrip("'>").split(".")[-1]
            if op_type == "DequantQuantWrapper":
                op_type = str(type(modules[op_name].module)).rstrip("'>").split(".")[-1]
            if "Functional" in op_type:
                op_type = op_name.split(".")[-1]
            if op_type not in res.keys():
                res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
            value = tune_cfg["op"][key]
            # Special cases: QuantStub, Embedding
            if ("weight" in value and value["weight"]["dtype"] == "fp32") or (
                "weight" not in value and value["activation"]["dtype"] == "fp32"
            ):
                res[op_type]["FP32"] += 1
            elif value["activation"]["dtype"] == "bf16":  # pragma: no cover
                res[op_type]["BF16"] += 1
            else:
                res[op_type]["INT8"] += 1
        # fetch other quantizable ops supported in PyTorch from model
        for name, child in modules.items():
            op_type = str(type(child)).rstrip("'>").split(".")[-1]
            if tune_cfg["approach"] != "post_training_dynamic_quant":
                if op_type == "DeQuantize":
                    if op_type not in res.keys():
                        res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
                    res[op_type]["INT8"] += 1
                if op_type in self.non_quant_dict["skipped_module_classes"]:
                    ignore_log = True
                    if op_type not in res.keys():
                        res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
                    res[op_type]["FP32"] += 1
        # show results to users
        if ignore_log:
            logger.info(
                "Ignore LayerNorm, InstanceNorm3d and Embedding quantizable ops" " due to accuracy issue in PyTorch."
            )

        field_names = ["Op Type", "Total", "INT8", "BF16", "FP32"]
        output_data = [
            [op_type, sum(res[op_type].values()), res[op_type]["INT8"], res[op_type]["BF16"], res[op_type]["FP32"]]
            for op_type in res.keys()
        ]

        Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()
        self.optype_statistics = field_names, output_data

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """
        module_dict = dict(model.named_modules())
        for op_name, child in model.named_modules():
            if self.is_fused_module(child):
                for name, _ in child.named_children():
                    module_prefix = op_name + "." + name
                    if module_prefix in module_dict:
                        module_dict.pop(module_prefix)  # remove sub-modules of fused modules
        for op_name, child in module_dict.items():
            # there is accuracy issue in quantized LayerNorm op in pytorch <1.8.1,
            # so remove it here
            if (
                op_name in self.non_quant_dict["skipped_module_names"]
                or str(child.__class__.__name__) in self.non_quant_dict["skipped_module_classes"]
            ):
                continue
            if (
                type(child) in self.white_list
                and type(child) != torch.nn.Sequential
                and type(child) != torch.quantization.stubs.DeQuantStub
            ):
                quantizable_ops.append(
                    (
                        op_name,
                        (
                            unify_op_type_mapping[str(child.__class__.__name__)]
                            if str(child.__class__.__name__) in unify_op_type_mapping
                            else str(child.__class__.__name__)
                        ),
                    )
                )

    def _get_scale_zeropoint(self, model, tune_cfg):
        """Get activation scale and zero_point for converted model.

        Args:
            model (dir): Int8 model converted from fp32 model.
                        scale and zero_point is set with calibration for each module
            tune_cfg (object): This file saves scale and zero_point of \
                            output activation of each quantized module.

        Returns:
            None
        """
        modules = dict(model.named_modules())
        for key, value in tune_cfg["op"].items():
            if hasattr(modules[key[0]], "scale"):
                value["activation"]["scale"] = float(modules[key[0]].scale)
            if hasattr(modules[key[0]], "zero_point"):
                value["activation"]["zero_point"] = int(modules[key[0]].zero_point)

    def is_fused_child(self, op_name):
        """This is a helper function for `_post_eval_hook`

        Args:
            op_name (string): op name

        Returns:
            (bool): if this op is fused
        """
        for key in self.fused_dict:
            if op_name in self.fused_dict[key]:
                return True
        return False

    def _post_eval_hook(self, model, **args):
        """The function is used to do some post process after complete evaluation.
           Here, it used to dump quantizable op's output tensor.

        Args:
            model (object): input model

        Returns:
            None
        """
        from torch.utils.tensorboard import SummaryWriter

        if self.version.release >= Version("2.0.0").release:
            from torch.quantization.quantize import _get_observer_dict as get_observer_dict
        else:
            from torch.quantization import get_observer_dict

        model = model._model

        if args is not None and "accuracy" in args:
            accuracy = args["accuracy"]
        else:
            accuracy = ""

        if self.dump_times == 0:
            writer = SummaryWriter("runs/eval/baseline" + "_acc" + str(accuracy), model)
        else:
            writer = SummaryWriter("runs/eval/tune_" + str(self.dump_times) + "_acc" + str(accuracy), model)

        if self.dump_times == 0:
            for input, _ in self.q_dataloader:
                if isinstance(input, dict) or isinstance(input, UserDict):
                    if self.device == "gpu":
                        for inp in input.keys():
                            input[inp] = input[inp].to("dpcpp")
                elif isinstance(input, list) or isinstance(input, tuple):
                    if self.device == "gpu":
                        input = [inp.to("dpcpp") for inp in input]
                else:
                    if self.device == "gpu":
                        input = input.to("dpcpp")
                writer.add_graph(model, input)
                break

        summary = OrderedDict()
        observer_dict = {}
        get_observer_dict(model, observer_dict)
        for key in observer_dict:
            if isinstance(observer_dict[key], torch.nn.modules.linear.Identity):
                continue
            op_name = key.replace(".activation_post_process", "")
            summary[op_name + ".output"] = observer_dict[key].get_tensor_value()
            for iter in summary[op_name + ".output"]:
                # Only collect last fused child output
                op = op_name
                if op_name in self.fused_dict:
                    op = self.fused_dict[op_name][0]
                else:
                    for key in self.fused_dict:
                        if op_name in self.fused_dict[key]:
                            op = op_name

                if summary[op_name + ".output"][iter].is_quantized:
                    writer.add_histogram(op + "/Output/int8", torch.dequantize(summary[op_name + ".output"][iter]))
                else:
                    writer.add_histogram(op + "/Output/fp32", summary[op_name + ".output"][iter])

        state_dict = model.state_dict()
        for key in state_dict:
            if not isinstance(state_dict[key], torch.Tensor):
                continue
            op = key[: key.rfind(".")]
            if self.is_fused_child(op) is True:
                # fused child tensorboard tag will be merge
                weight = key[key.rfind(".") + 1 :]
                op = op[: op.rfind(".")] + "/" + weight
            else:
                weight = key[key.rfind(".") + 1 :]
                op = key[: key.rfind(".")] + "/" + weight

            # To merge ._packed_params
            op = op.replace("._packed_params", "")

            if state_dict[key].is_quantized:
                writer.add_histogram(op + "/int8", torch.dequantize(state_dict[key]))
            else:
                writer.add_histogram(op + "/fp32", state_dict[key])

        writer.close()
        self.dump_times = self.dump_times + 1

        return summary

    @dump_elapsed_time("Pass save quantized model")
    def save(self, model, path=None):
        pass

    def set_tensor(self, model, tensor_dict):
        state_dict = model._model.state_dict()
        tensor_name = None
        for key in tensor_dict.keys():
            end = key.rfind(".")
            op_name = key[:end]
            state_op_name = None
            weight_bias = key[end + 1 :]
            for op in self.fused_dict:
                if op_name in self.fused_dict[op]:
                    if model.is_quantized:
                        state_op_name = op
                    else:
                        state_op_name = self.fused_dict[op][0]
                # elif op_name in self.fused_dict[op]:
                # state_op_name = op
            if state_op_name is None:
                state_op_name = op_name
            for state_dict_key in state_dict.keys():
                state_key_end = state_dict_key.rfind(".")
                state_key = state_dict_key[:state_key_end].replace("._packed_params", "")
                if weight_bias in state_dict_key and state_op_name == state_key:
                    tensor_name = state_dict_key
            assert tensor_name is not None, key + " is not in the state dict"
            tensor = torch.from_numpy(tensor_dict[key])
            dtype = state_dict[tensor_name].dtype
            if state_dict[tensor_name].is_quantized:
                if "channel" in str(state_dict[tensor_name].qscheme()):
                    scales = state_dict[tensor_name].q_per_channel_scales()
                    zero_points = state_dict[tensor_name].q_per_channel_zero_points()
                    axis = state_dict[tensor_name].q_per_channel_axis()
                    state_dict[tensor_name] = torch.quantize_per_channel(tensor, scales, zero_points, axis, dtype=dtype)
                elif "tensor" in str(state_dict[tensor_name].qscheme()):
                    scales = state_dict[tensor_name].q_scale()
                    zero_points = state_dict[tensor_name].q_zero_point()
                    state_dict[tensor_name] = torch.quantize_per_tensor(tensor, scales, zero_points, dtype)
            else:
                state_dict[tensor_name] = tensor
        model._model.load_state_dict(state_dict)

    @dump_elapsed_time("Pass query framework capability")
    def query_fw_capability(self, model):
        """This is a helper function to get all quantizable ops from model.

        Args:
            model (object): input model which is Neural Compressor model

        Returns:
            q_capability (dictionary): tuning capability for each op from model.
        """
        self.pre_optimized_model = model
        self.non_quant_dict = self.get_non_quant_modules(model.kwargs)
        return self._get_quantizable_ops(model.model)

    def get_non_quant_modules(self, model_kwargs):
        """This is a helper function to get all non_quant_modules from customer and default.

        Args:
            model_kwargs (dictionary): keyword args from Neural Compressor model

        Returns:
            custom_non_quant_dict (dictionary): non_quant_modules for model.
        """
        if model_kwargs is None:
            model_kwargs = {}
        skipped_module_names = model_kwargs.get("non_quant_module_name", [])
        skipped_module_classes = model_kwargs.get("non_quant_module_class", [])
        custom_non_quant_dict = {
            "skipped_module_names": skipped_module_names,
            "skipped_module_classes": skipped_module_classes,
        }
        # Ignore LayerNorm, InstanceNorm3d and Embedding quantizable ops,
        # due to huge accuracy regression in PyTorch.
        additional_skipped_module_classes = ["LayerNorm", "InstanceNorm3d", "Embedding", "Dropout"]
        if self.approach == "post_training_dynamic_quant":
            additional_skipped_module_classes.remove("Embedding")
        custom_non_quant_dict["skipped_module_classes"] += additional_skipped_module_classes
        return custom_non_quant_dict


unify_op_type_mapping_ipex = {
    "Convolution_Relu": "Conv2d",
    "Convolution_Sum_Relu": "Conv2d",
    "Convolution_BatchNorm": "Conv2d",
    "<class 'torch.nn.modules.conv.Conv1d'>": "Conv1d",
    "<class 'torch.nn.modules.conv.Conv2d'>": "Conv2d",
    "<class 'torch.nn.modules.conv.Conv3d'>": "Conv3d",
    "<class 'torch.nn.modules.activation.ReLU'>": "ReLU",
    "<method 'add' of 'torch._C._TensorBase' objects>": "add",  # for IPEX < 2.2
    "<method 'add' of 'torch._C.TensorBase' objects>": "add",  # for IPEX >= 2.2
    "<class 'torch.nn.modules.pooling.AdaptiveAvgPool2d'>": "AdaptiveAvgPool2d",
    "Linear_Relu": "Linear",
    "<class 'torch.nn.modules.linear.Linear'>": "Linear",
    "<class 'torch.nn.modules.pooling.MaxPool2d'>": "MaxPool2d",
    "re": {"<built-in method matmul of type object at": "matmul"},
}


@adaptor_registry
class PyTorch_IPEXAdaptor(TemplateAdaptor):
    """Adaptor of PyTorch framework with Intel PyTorch Extension,
       all PyTorch IPEX API is in this class.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """

    def __init__(self, framework_specific_info):
        super(PyTorch_IPEXAdaptor, self).__init__(framework_specific_info)
        self.version = get_ipex_version()
        query_config_file = "pytorch_ipex.yaml"
        self.query_handler = PyTorchQuery(
            device=self.device, local_config_file=os.path.join(os.path.dirname(__file__), query_config_file)
        )
        self.cfgs = None
        self.fuse_ops = None
        self.op_infos_from_cfgs = None
        self.output_tensor_id_op_name = None
        self.ipex_config_path = os.path.join(self.workspace_path, "ipex_config_tmp.json")
        self.sq_minmax_init = True if framework_specific_info.get("model_init_algo", "kl") == "minmax" else False

        try:
            os.remove(self.ipex_config_path)
        except:
            logger.warning("Fail to remove {}.".format(self.ipex_config_path))

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization config.
            model (object): model need to do quantization, it is Neural Compressor model.
            dataloader (object): calibration dataset.
            q_func (objext, optional): training function for quantization aware training mode.

        Returns:
            (dict): quantized model
        """
        # IPEX bug #1: deepcopied prepared model cannot do calibration, need model._model
        # q_model._model is useless, but we need to copy other attributes, and pass the converted
        # model to q_model. Also, sq will collect state_dict to origin_stat for recover
        if self.device == "xpu":
            model.to(self.device)
        if self.performance_only:
            q_model = model
        else:
            try:
                q_model = copy.deepcopy(model)
            except Exception as e:  # pragma: no cover
                logger.warning("Fail to deep copy the model due to {}, inplace is used now.".format(repr(e)))
                q_model = model

        assert self.approach in [
            "post_training_static_quant",
            "post_training_auto_quant",
        ], "IPEX in INC only supports approach is static or auto"
        assert not self.version.release < Version("1.10.0").release, "INC support IPEX version >= 1.10.0"

        # check smoothquant folding value
        recipe_cfgs = tune_cfg.get("recipe_cfgs", None)
        if "smooth_quant_args" in recipe_cfgs and "folding" in recipe_cfgs["smooth_quant_args"]:
            if recipe_cfgs["smooth_quant_args"]["folding"] is None:
                if self.version.release < Version("2.1").release:
                    folding = True
                else:
                    folding = False
            else:
                folding = recipe_cfgs["smooth_quant_args"]["folding"]
        # Update model parameter when smoothquant folding = False
        if (
            recipe_cfgs
            and recipe_cfgs.get("smooth_quant", False)
            and not folding
            and self.approach != "post_training_dynamic_quant"
        ):
            return self.qdq_quantize(model, q_model, tune_cfg, dataloader, q_func)
        # Update model parameter when smoothquant folding = True
        if recipe_cfgs and recipe_cfgs.get("smooth_quant", False) and folding:
            self._apply_pre_optimization(model, tune_cfg)

        assert (
            self.approach != "quant_aware_training"
        ), "Intel PyTorch Extension didn't support quantization aware training mode"
        assert not self.version.release < Version("1.10.0").release, "INC support IPEX version >= 1.10.0"

        qscheme = self._cfg_to_qconfig(tune_cfg)  # Update json file in self.ipex_config_path
        iterations = tune_cfg.get("calib_iteration", 1)
        model._model.eval()
        inplace = True if self.performance_only else False

        if self.version.release >= Version("1.12.0").release:
            # Check save_qconf_summary part is a workaround for IPEX bug.
            # Sometimes the prepared model from get_op_capablitiy loss this attribute
            if not hasattr(model._model, "save_qconf_summary") or not hasattr(model._model, "load_qconf_summary"):
                from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig

                if self.device == "xpu":
                    static_qconfig = QConfig(
                        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
                    )
                elif self.version.release >= Version("2.1").release:
                    static_qconfig = ipex.quantization.default_static_qconfig_mapping
                else:
                    static_qconfig = QConfig(
                        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                        weight=PerChannelMinMaxObserver.with_args(
                            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
                        ),
                    )
                if isinstance(self.example_inputs, dict):
                    model._model = ipex.quantization.prepare(
                        model._model, static_qconfig, example_kwarg_inputs=self.example_inputs, inplace=inplace
                    )
                else:
                    model._model = ipex.quantization.prepare(
                        model._model, static_qconfig, example_inputs=self.example_inputs, inplace=inplace
                    )
            model._model.load_qconf_summary(qconf_summary=self.ipex_config_path)
            if q_func is not None:
                q_func(model._model)
            else:
                self.model_calibration(
                    model._model, dataloader, iterations, None, tune_cfg.get("calib_sampling_size", 1)
                )
            model._model.save_qconf_summary(qconf_summary=self.ipex_config_path)
            self._ipex_post_quant_process(model, q_model, dataloader, inplace=inplace)
        else:
            # for IPEX version < 1.12
            ipex_conf = ipex.quantization.QuantConf(
                configure_file=self.ipex_config_path, qscheme=qscheme
            )  # pylint: disable=E1101
            self.model_calibration(
                q_model._model, dataloader, iterations, ipex_conf, tune_cfg.get("calib_sampling_size", 1)
            )
            ipex_conf.save(self.ipex_config_path)
            ipex_conf = ipex.quantization.QuantConf(self.ipex_config_path)  # pylint: disable=E1101
            q_model._model = ipex.quantization.convert(
                q_model._model, ipex_conf, self.example_inputs, inplace=True
            )  # pylint: disable=E1121

        # Recover model parameter when smoothquant folding = True, due to IPEX bug #1
        if (
            recipe_cfgs
            and recipe_cfgs.get("smooth_quant", False)
            and recipe_cfgs["smooth_quant_args"]["folding"]
            and not inplace
        ):
            self._apply_pre_optimization(model, tune_cfg, recover=True)

        with open(self.ipex_config_path, "r") as f:
            q_model.tune_cfg = json.load(f)
        q_model.ipex_config_path = self.ipex_config_path
        if self.version.release >= Version("1.12.0").release:
            self._dump_model_op_stats(tune_cfg)
        return q_model

    def _ipex_post_quant_process(self, model, q_model, dataloader, inplace=False):
        if (
            self.use_bf16
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
            and (self.version.release >= Version("1.11.0").release)
        ):
            with torch.no_grad():
                with torch.cpu.amp.autocast():
                    q_model._model = ipex.quantization.convert(model._model, inplace=inplace)
                    try:
                        if isinstance(self.example_inputs, dict):
                            q_model._model = torch.jit.trace(
                                q_model._model,
                                example_kwarg_inputs=self.example_inputs,
                            )
                        else:
                            q_model._model = torch.jit.trace(q_model._model, self.example_inputs)
                        q_model._model = torch.jit.freeze(q_model._model.eval())
                    except:
                        if isinstance(self.example_inputs, dict):
                            q_model._model = torch.jit.trace(
                                q_model._model,
                                example_kwarg_inputs=self.example_inputs,
                                strict=False,
                                check_trace=False,
                            )
                        else:
                            q_model._model = torch.jit.trace(q_model._model, self.example_inputs, strict=False)
                        q_model._model = torch.jit.freeze(q_model._model.eval())
        else:
            q_model._model = ipex.quantization.convert(model._model, inplace=inplace)
            with torch.no_grad():
                try:
                    if isinstance(self.example_inputs, dict):
                        q_model._model = torch.jit.trace(q_model._model, example_kwarg_inputs=self.example_inputs)
                    else:
                        q_model._model = torch.jit.trace(q_model._model, self.example_inputs)
                    q_model._model = torch.jit.freeze(q_model._model.eval())
                except:
                    if isinstance(self.example_inputs, dict):
                        q_model._model = torch.jit.trace(
                            q_model._model, example_kwarg_inputs=self.example_inputs, strict=False, check_trace=False
                        )
                    else:
                        q_model._model = torch.jit.trace(q_model._model, self.example_inputs, strict=False)
                    q_model._model = torch.jit.freeze(q_model._model.eval())
        # After freezing, run 1 time to warm up the profiling graph executor to insert prim::profile
        # At the 2nd run, the llga pass will be triggered and the model is turned into
        # an int8 model: prim::profile will be removed and will have LlgaFusionGroup in the graph
        self._simple_inference(q_model._model, dataloader, iterations=2)

    def _dump_model_op_stats(self, tune_cfg):
        """This is a function to dump quantizable ops of model to user.

        Args:
            tune_cfg (dict): quantization config
        Returns:
            None
        """
        res = dict()
        for k, v in tune_cfg["op"].items():
            op_type_list = k[-1].split("><")
            op_type = ""
            for op in op_type_list:
                if "class" in op:
                    op_type = (
                        op[op.rfind(".") + 1 : op.rfind("'")]
                        if op_type == ""
                        else op_type + "&" + op[op.rfind(".") + 1 : op.rfind("'")]
                    )
                elif "method" in op:
                    start = op.find("'") + 1
                    if start > 1:
                        op_type = (
                            op[start : op.find("'", start)]
                            if op_type == ""
                            else op_type + "&" + op[start : op.find("'", start)]
                        )
                    else:
                        start = op.find("method") + 7
                        op_type = (
                            op[start : op.find(" ", start)]
                            if op_type == ""
                            else op_type + "&" + op[start : op.find(" ", start)]
                        )
                else:
                    op_type = op if op_type == "" else op_type + "&" + op
            if op_type not in res.keys():
                res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
            if v["weight"]["dtype"] == "int8":
                res[op_type]["INT8"] += 1
            elif v["weight"]["dtype"] == "fp32":
                res[op_type]["FP32"] += 1

        output_data = [
            [op_type, sum(res[op_type].values()), res[op_type]["INT8"], res[op_type]["BF16"], res[op_type]["FP32"]]
            for op_type in res.keys()
        ]

        Statistics(
            output_data, header="Mixed Precision Statistics", field_names=["Op Type", "Total", "INT8", "BF16", "FP32"]
        ).print_stat()

    def _cfg_to_qconfig(self, tune_cfg, smooth_quant=False):
        """Convert tune configure to quantization config for each op.

        Args:
            tune_cfg (dict): dictionary of tune configure for each op
            ipex_config_path: configure file of Intel PyTorch Extension

        tune_cfg should be a format like below:
        {
          'calib_iteration': 10,
          'op': {
             ('op1', 'CONV2D'): {
               'activation':  {'dtype': 'uint8',
                               'algorithm': 'minmax',
                               'scheme':'sym',
                               'granularity': 'per_tensor'},
               'weight': {'dtype': 'int8',
                          'algorithm': 'kl',
                          'scheme':'asym',
                          'granularity': 'per_channel'}
             },
             ('op2', 'RELU): {
               'activation': {'dtype': 'int8',
               'scheme': 'asym',
               'granularity': 'per_tensor',
               'algorithm': 'minmax'}
             },
             ('op3', 'CONV2D'): {
               'activation':  {'dtype': 'fp32'},
               'weight': {'dtype': 'fp32'}
             },
             ...
          }
        }
        """
        assert self.cfgs is not None, "No configure for IPEX int8 model..."
        if self.version.release < Version("1.12.0").release:  # pragma: no cover
            for key in tune_cfg["op"]:
                try:
                    scheme = tune_cfg["op"][key]["activation"]["scheme"]
                except:
                    scheme = "asym"
                if scheme not in ["asym", "sym"]:
                    scheme = "asym"
                break
            for key in tune_cfg["op"]:
                value = tune_cfg["op"][key]
                pattern = self.get_pattern(key, self.fuse_ops)
                assert isinstance(value, dict)
                assert "activation" in value
                if value["activation"]["dtype"] == "fp32":
                    if "weight" in value:
                        assert value["weight"]["dtype"] == "fp32"
                    for op_cfg in self.cfgs:
                        if op_cfg["id"] == key[0]:
                            if key[1] in ["relu_", "add_"]:
                                continue
                            num_inputs = len(op_cfg["inputs_quantized"])
                            num_outputs = len(op_cfg["outputs_quantized"])
                            for i_num in range(num_inputs):
                                op_cfg["inputs_quantized"][i_num] = False
                            for o_num in range(num_outputs):
                                op_cfg["outputs_quantized"][o_num] = False
                            if pattern:
                                if pattern[1] in ["relu_", "add_"]:
                                    continue
                                tune_cfg["op"][pattern]["activation"]["dtype"] = "fp32"
                                if "weight" in tune_cfg["op"][pattern]:
                                    tune_cfg["op"][pattern]["weight"]["dtype"] = "fp32"
                else:
                    for op_cfg in self.cfgs:
                        if op_cfg["id"] == key[0]:
                            if key[1] in ["relu_", "add_"]:
                                continue
                            num_inputs = len(op_cfg["inputs_quantized"])
                            num_outputs = len(op_cfg["outputs_quantized"])
                            for i_num in range(num_inputs):
                                op_cfg["inputs_quantized"][i_num] = self.default_cfgs[key[0]]["inputs_quantized"][i_num]
                            for o_num in range(num_outputs):
                                op_cfg["outputs_quantized"][o_num] = self.default_cfgs[key[0]]["outputs_quantized"][
                                    o_num
                                ]
            with open(self.ipex_config_path, "w") as write_f:
                json.dump(self.cfgs, write_f)
            if scheme == "asym":
                return torch.per_tensor_affine
            else:
                return torch.per_tensor_symmetric
        else:
            op_infos = copy.deepcopy(self.op_infos_from_cfgs)
            self.cfgs = torch_utils.util.check_cfg_and_qconfig(
                tune_cfg["op"], self.cfgs, op_infos, self.output_tensor_id_op_name, smooth_quant
            )

            with open(self.ipex_config_path, "w") as write_f:
                json.dump(self.cfgs, write_f, indent=4)
            return None

    def get_pattern(self, fallback_op, fuse_ops):  # pragma: no cover
        for fuse_pattern in fuse_ops:
            if fuse_pattern[0] == fallback_op:
                if fuse_pattern[1] in ["relu_", "add_"]:
                    return None
                else:
                    return fuse_pattern[1]
        return None

    def evaluate(
        self,
        model,
        dataloader,
        postprocess=None,
        metrics=None,
        measurer=None,
        iteration=-1,
        tensorboard=False,
        fp32_baseline=False,
    ):
        """Execute the evaluate process on the specified model.

        Args:
            model (object): Neural Compressor model to run evaluation.
            dataloader (object): evaluation dataset.
            postprocess (object, optional): process function after evaluation.
            metrics (list, optional): list of metric function.
            measurer (object, optional): measurer function.
            iteration (int, optional): number of iterations to evaluate.
            tensorboard (bool, optional): dump output tensor to tensorboard summary
                                          files(IPEX unspport).
            fp32_baseline (boolean, optional): only for compare_label=False pipeline

        Returns:
            (dict): quantized model
        """

        assert not tensorboard, "Intel PyTorch Extension didn't tensor dump"
        self.is_baseline = fp32_baseline

        model_ = model._model
        model_.eval()

        if metrics:
            self.fp32_preds_as_label = any(
                [hasattr(metric, "compare_label") and not metric.compare_label for metric in metrics]
            )

        ipex_config = self.ipex_config_path if not self.benchmark else None
        if self.version.release < Version("1.12.0").release:
            conf = (
                ipex.quantization.QuantConf(configure_file=ipex_config)  # pylint: disable=E1101
                if not self.is_baseline
                else None
            )
        else:
            conf = None

        return self.model_eval(model_, dataloader, postprocess, metrics, measurer, iteration, conf)

    @dump_elapsed_time("Pass query framework capability")
    def query_fw_capability(self, model):
        """This is a helper function to get all quantizable ops from model.

        Args:
            model (object): input model which is Neural Compressor model

        Returns:
            q_capability (dictionary): tuning capability for each op from model.
        """
        self.pre_optimized_model = model
        return self._get_quantizable_ops(model.model)

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.
        Returns:
            None
        """

        # group ops by position for transform-based model
        from .torch_utils.pattern_detector import TransformerBasedModelBlockPatternDetector

        detector = TransformerBasedModelBlockPatternDetector(model)
        detect_result = detector.detect_block()
        attention_block = detect_result.get("attention_blocks", None)
        ffn_blocks = detect_result.get("ffn_blocks", None)
        logger.info(f"Attention Blocks: {len(attention_block)}")
        logger.info(f"FFN Blocks: {len(ffn_blocks)}")
        if not os.path.exists(self.ipex_config_path):
            assert isinstance(model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"

        if hasattr(model, "save_qconf_summary"):
            os.makedirs(os.path.dirname(self.ipex_config_path), exist_ok=True)
            model.save_qconf_summary(qconf_summary=self.ipex_config_path)
            if self.example_inputs is None:
                self.example_inputs = get_example_inputs(model, self.q_dataloader)
        else:
            model.eval()
            # to record the origin batch_size
            if isinstance(self.q_dataloader, BaseDataLoader):
                batch_size = self.q_dataloader.batch_size

            # create a quantization config file for intel pytorch extension model
            os.makedirs(os.path.dirname(self.ipex_config_path), exist_ok=True)
            if self.version.release < Version("1.12.0").release:
                assert self.q_func is None, (
                    "IPEX < 1.12.0 didn't support calibration function, " "Please use IPEX >= 1.12.0!"
                )
                ipex_conf = ipex.quantization.QuantConf(qscheme=torch.per_tensor_symmetric)  # pylint: disable=E1101
                self.model_calibration(
                    model,
                    self.q_dataloader,
                    conf=ipex_conf,
                )
                ipex_conf.save(self.ipex_config_path)
            else:
                if self.approach in ["post_training_static_quant", "post_training_auto_quant"]:
                    assert (
                        self.q_dataloader is not None or self.example_inputs is not None
                    ), "IPEX need q_dataloader or example_inputs to prepare the model"
                    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig

                    if self.device == "xpu":
                        static_qconfig = QConfig(
                            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric),
                        )
                    elif self.version.release >= Version("2.1").release:
                        # HistogramObserver will cause a performance issue.
                        # static_qconfig = ipex.quantization.default_static_qconfig_mapping
                        qconfig = QConfig(
                            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                            weight=PerChannelMinMaxObserver.with_args(
                                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
                            ),
                        )
                        from torch.ao.quantization import QConfigMapping

                        static_qconfig = QConfigMapping().set_global(qconfig)
                    else:
                        static_qconfig = QConfig(
                            activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                            weight=PerChannelMinMaxObserver.with_args(
                                dtype=torch.qint8, qscheme=torch.per_channel_symmetric
                            ),
                        )
                    # For smoothquant optimized model, need ipex version >= 2.1
                    if (
                        self.recipes
                        and self.recipes.get("smooth_quant", False)
                        and self.version.release >= Version("2.1").release
                    ):  # pragma: no cover
                        smooth_quant_args = self.recipes.get("smooth_quant_args", {})
                        folding = smooth_quant_args.get("folding", False)
                        if not folding:
                            from torch.ao.quantization.observer import MinMaxObserver

                            if self.version.release >= Version("2.1.1").release:
                                static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                                    alpha=0.5, act_observer=MinMaxObserver
                                )
                            else:
                                if self.sq_minmax_init:
                                    static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                                        alpha=0.5, act_observer=MinMaxObserver()
                                    )
                                    logger.warning(
                                        "The int8 model accuracy will be close to 0 with MinMaxobserver, "
                                        + "the suggested IPEX version is higher or equal than 2.1.100."
                                    )
                                else:
                                    static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5)
                    if self.example_inputs is None:
                        self.example_inputs = get_example_inputs(model, self.q_dataloader)
                    from neural_compressor.adaptor.torch_utils.util import move_input_device

                    self.example_inputs = move_input_device(self.example_inputs, device=self.device)
                    if isinstance(self.example_inputs, dict):
                        model = ipex.quantization.prepare(
                            model, static_qconfig, example_kwarg_inputs=self.example_inputs, inplace=True
                        )
                    else:
                        model = ipex.quantization.prepare(
                            model, static_qconfig, example_inputs=self.example_inputs, inplace=True
                        )

                if self.q_dataloader is not None or self.example_inputs is not None:
                    self._simple_inference(model, self.q_dataloader, iterations=1)
                else:
                    try:
                        self.q_func(model)
                    except Exception as e:
                        logger.error("Calibration with IPEX failed due to:{}".format(e))
                        assert False, "Please pass in example_inputs or calib_dataloader to bypass."

                model.save_qconf_summary(qconf_summary=self.ipex_config_path)
            if isinstance(self.q_dataloader, BaseDataLoader):
                self.q_dataloader.batch(batch_size)
                logger.info(
                    "Recovery `calibration.dataloader.batchsize` {} according \
                            to config.yaml".format(
                        batch_size
                    )
                )

        map_op_name_to_fqn = {}
        with open(self.ipex_config_path, "r") as f:
            self.cfgs = json.load(f)
            if self.version.release < Version("1.12.0").release:  # pragma: no cover
                self.default_cfgs = copy.deepcopy(self.cfgs)
                self.fuse_ops = self.get_fuse_ops(self.cfgs)
                for op_cfg in self.cfgs:
                    if op_cfg["name"] in unify_op_type_mapping_ipex:
                        quantizable_ops.append((op_cfg["id"], unify_op_type_mapping_ipex[op_cfg["name"]]))
                    else:
                        re_flag = False
                        for pattern, unify_op_type in unify_op_type_mapping_ipex["re"].items():
                            if re.match(pattern, op_cfg["name"]):
                                re_flag = True
                                quantizable_ops.append((op_cfg["id"], unify_op_type))
                                break
                        if not re_flag:
                            quantizable_ops.append((op_cfg["id"], op_cfg["name"]))
            else:
                (
                    ops_name,
                    op_infos_from_cfgs,
                    input_tensor_id_op_name,
                    output_tensor_id_op_name,
                ) = torch_utils.util.paser_cfgs(self.cfgs)
                quantizable_op_names = torch_utils.util.get_quantizable_ops_from_cfgs(
                    ops_name, op_infos_from_cfgs, input_tensor_id_op_name
                )
                for name in quantizable_op_names:
                    # name : list
                    if len(name) == 1:
                        module_key = name[0][0]
                        op_cfg_id = name[0][2]
                        ipex_op_type = self.cfgs[module_key]["q_op_infos"][op_cfg_id]["op_type"]
                        module_fqn = self.cfgs[module_key]["q_op_infos"][op_cfg_id].get("fqn", None)

                        if ipex_op_type in unify_op_type_mapping_ipex:
                            quantizable_ops.append((tuple(name), unify_op_type_mapping_ipex[ipex_op_type]))
                            map_op_name_to_fqn[(tuple(name), ipex_op_type)] = module_fqn
                        else:
                            re_flag = False
                            for pattern, unify_op_type in unify_op_type_mapping_ipex["re"].items():
                                if re.match(pattern, ipex_op_type):
                                    re_flag = True
                                    quantizable_ops.append((tuple(name), unify_op_type))
                                    map_op_name_to_fqn[(tuple(name), unify_op_type)] = module_fqn
                                    break
                            if not re_flag:
                                quantizable_ops.append((tuple(name), ipex_op_type))
                                map_op_name_to_fqn[(tuple(name), ipex_op_type)] = module_fqn
                    else:
                        op_type = ""
                        for op_name in name:
                            module_key = op_name[0]
                            op_cfg_id = op_name[2]
                            single_op_type = self.cfgs[module_key]["q_op_infos"][op_cfg_id]["op_type"]
                            if single_op_type in unify_op_type_mapping_ipex:
                                single_op_type = unify_op_type_mapping_ipex[single_op_type]
                            op_type += "&" + single_op_type if op_type else single_op_type
                        quantizable_ops.append((tuple(name), op_type))
                        _module_key = name[0][0]
                        _op_cfg_id = name[0][2]
                        module_fqn = self.cfgs[_module_key]["q_op_infos"][_op_cfg_id]["fqn"]
                        map_op_name_to_fqn[(tuple(name), op_type)] = module_fqn
                self.op_infos_from_cfgs = op_infos_from_cfgs
                self.output_tensor_id_op_name = output_tensor_id_op_name
        logger.debug("Map op name to fqn: ")
        logger.debug(map_op_name_to_fqn)
        logger.info("Attention Blocks : ")
        logger.info(attention_block)
        logger.info("FFN Blocks : ")
        logger.info(ffn_blocks)
        self.block_wise = ffn_blocks

    def get_fuse_ops(self, default_cfgs):  # pragma: no cover
        elt_wise = ["relu", "sigmoid", "gelu"]
        inplace_ops = ["relu_", "add_"]
        op_patterns = []
        num_ops = len(default_cfgs)
        for cur_id in range(num_ops):
            cur_op = default_cfgs[cur_id]["name"]
            if cur_op == "dropout":
                continue
            inputs = default_cfgs[cur_id]["inputs_flow"]
            num_input = len(inputs)
            pre_ops = {}
            for i_num in range(num_input):
                inp = inputs[i_num]
                for pre_id in range(cur_id):
                    pre_op = default_cfgs[pre_id]["name"]
                    pre_out = default_cfgs[pre_id]["outputs_flow"]
                    num_out = len(pre_out)
                    for o_num in range(num_out):
                        if pre_out[o_num] == inp:
                            if cur_op in inplace_ops and (pre_op in ["conv2d", "conv3d", "linear"]):
                                op_patterns.append([(pre_id, pre_op), (cur_id, cur_op)])
                            if cur_op in elt_wise and (pre_op in ["conv2d", "conv3d", "linear", "add"]):
                                op_patterns.append([(pre_id, pre_op), (cur_id, cur_op)])
                            if cur_op == "add":
                                pre_ops[i_num] = [pre_id, pre_op]
            if len(pre_ops) > 0:
                for key, value in pre_ops.items():
                    if (
                        value[1] in ["conv2d", "conv3d", "linear"]
                        and default_cfgs[cur_id]["inputs_quantized"][key] is False
                    ):
                        op_patterns.append([(value[0], value[1]), (cur_id, cur_op)])
        return op_patterns

    def qdq_quantize(self, model, q_model, tune_cfg, dataloader, q_func):
        assert not self.version.release < Version("2.1").release, "IPEX version >= 2.1 is required for SmoothQuant."
        inplace = True if self.performance_only else False

        # fetch SmoothQuant scale info from pre-optimized model
        smoothquant_scale_info = model.sq_scale_info

        # Check save_qconf_summary part is a workaround for IPEX bug.
        # Sometimes the prepared model from get_op_capablitiy loss this attribute
        if not hasattr(model._model, "save_qconf_summary") or not hasattr(model._model, "load_qconf_summary"):
            from torch.ao.quantization.observer import MinMaxObserver

            if self.version.release >= Version("2.1.1").release:
                static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                    alpha=0.5, act_observer=MinMaxObserver
                )
            else:
                if self.sq_minmax_init:
                    static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(
                        alpha=0.5, act_observer=MinMaxObserver()
                    )
                    logger.warning(
                        "The int8 model accuracy will be close to 0 with MinMaxobserver, "
                        + "the suggested IPEX version is higher or equal than 2.1.100+cpu."
                    )
                else:
                    static_qconfig = ipex.quantization.get_smooth_quant_qconfig_mapping(alpha=0.5)
            if isinstance(self.example_inputs, dict):
                model._model = ipex.quantization.prepare(
                    model._model, static_qconfig, example_kwarg_inputs=self.example_inputs, inplace=inplace
                )
            else:
                model._model = ipex.quantization.prepare(
                    model._model, static_qconfig, example_inputs=self.example_inputs, inplace=inplace
                )

        # The IPEX SmoothQuant observer can only use save/load_qconf_summary once.
        # The save_qconf_summary API will freeze the scale used in model and calibration won't work anymore.
        # The load_qconf_summary will overwrite the scales used in model but only work in the first call.
        # Here, we use INC collected scale for Linear and set normal observer instead of SQObserver \
        # to make sure calibration works for other ops, like add, bmm.
        from .torch_utils.util import update_sq_scale

        self._cfg_to_qconfig(tune_cfg, smooth_quant=True)
        update_sq_scale(self.ipex_config_path, smoothquant_scale_info)
        model._model.load_qconf_summary(qconf_summary=self.ipex_config_path)
        # real calibration for other operators
        try:
            # IPEX may raise an error on the second iteration.
            # OverflowError: cannot convert float infinity to integer
            if q_func is not None:
                q_func(model._model)
            else:
                iterations = tune_cfg.get("calib_iteration", 1)
                self.model_calibration(
                    model._model, dataloader, iterations, None, tune_cfg.get("calib_sampling_size", 1)
                )
        except:
            logger.warning(
                "The calibration failed when calibrating with ipex, "
                + "using scale info from SmoothQuant for Linear and "
                + "one iter calibration for other ops."
            )
        model._model.save_qconf_summary(qconf_summary=self.ipex_config_path)
        if self.version.release > Version("2.1.0").release:
            update_sq_scale(self.ipex_config_path, smoothquant_scale_info)
            model._model.load_qconf_summary(qconf_summary=self.ipex_config_path)
        self._ipex_post_quant_process(model, q_model, dataloader, inplace=inplace)

        with open(self.ipex_config_path, "r") as f:
            q_model.tune_cfg = json.load(f)
        q_model.ipex_config_path = self.ipex_config_path
        self._dump_model_op_stats(tune_cfg)
        return q_model

    @dump_elapsed_time("Pass save quantized model")
    def save(self, model, path=None):
        """The function is used by tune strategy class for set best configure in Neural Compressor model.

           Args:
               model (object): The Neural Compressor model which is best results.
               path (string): No used.

        Returns:
            None
        """

        pass

    def inspect_tensor(
        self, model, dataloader, op_list=None, iteration_list=None, inspect_type="activation", save_to_disk=False
    ):
        assert False, "Inspect_tensor didn't support IPEX backend now!"

    def _simple_inference(self, q_model, dataloader, iterations=1):
        """The function is used for ipex warm-up inference."""
        if self.example_inputs is not None:
            for _ in range(iterations):
                if isinstance(self.example_inputs, tuple) or isinstance(self.example_inputs, list):
                    q_model(*self.example_inputs)
                elif isinstance(self.example_inputs, dict):
                    q_model(**self.example_inputs)
                else:
                    q_model(self.example_inputs)
        else:
            self.calib_func(q_model, dataloader, iterations)


@adaptor_registry
class PyTorch_FXAdaptor(TemplateAdaptor):
    """Adaptor of PyTorch framework with FX graph mode, all PyTorch API is in this class.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """

    def __init__(self, framework_specific_info):
        super(PyTorch_FXAdaptor, self).__init__(framework_specific_info)
        assert (
            self.version.release >= Version("1.8.0").release
        ), "Please use PyTroch 1.8 or higher version with pytorch_fx backend!"
        if self.approach == "post_training_dynamic_quant":
            assert self.version.release >= Version("1.9.0").release, (
                "Please use PyTroch 1.9 or higher version for dynamic " "quantization with pytorch_fx backend!"
            )
        import torch.quantization as tq

        """
        # Map for swapping float module to quantized ones,
        # and this dictionary will change with different PoTorch versions
        DEFAULT_MODULE_MAPPING = {
            nn.Linear: nnq.Linear,
            nn.ReLU: nnq.ReLU,
            nn.ReLU6: nnq.ReLU6,
            nn.Conv2d: nnq.Conv2d,
            nn.Conv3d: nnq.Conv3d,
            QuantStub: nnq.Quantize,
            DeQuantStub: nnq.DeQuantize,
            # Wrapper Modules:
            nnq.FloatFunctional: nnq.QFunctional,
            # Intrinsic modules:
            nni.ConvReLU2d: nniq.ConvReLU2d,
            nni.ConvReLU3d: nniq.ConvReLU3d,
            nni.LinearReLU: nniq.LinearReLU,
            nniqat.ConvReLU2d: nniq.ConvReLU2d,
            nniqat.LinearReLU: nniq.LinearReLU,
            nniqat.ConvBn2d: nnq.Conv2d,
            nniqat.ConvBnReLU2d: nniq.ConvReLU2d,
            # QAT modules:
            nnqat.Linear: nnq.Linear,
            nnqat.Conv2d: nnq.Conv2d,
        }
        """

        self.tune_cfg = None
        if self.device == "cpu":
            query_config_file = "pytorch_cpu.yaml"
        else:  # pragma: no cover
            assert False, "Unsupported this device {}".format(self.device)
        self.query_handler = PyTorchQuery(local_config_file=os.path.join(os.path.dirname(__file__), query_config_file))

        if self.approach == "post_training_dynamic_quant":
            self.white_list = tq.quantization_mappings.get_default_dynamic_quant_module_mappings()
        elif self.approach == "post_training_static_quant":
            self.white_list = tq.quantization_mappings.get_default_static_quant_module_mappings()
        else:
            self.white_list = tq.quantization_mappings.get_default_qconfig_propagation_list()

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization config.
            model (object): model need to do quantization.
            dataloader (object): calibration dataset.
            q_func (objext, optional): training function for quantization aware training mode.

        Returns:
            (object): quantized model
        """

        assert isinstance(model._model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        if self.performance_only:
            q_model = model
        else:
            try:
                q_model = copy.deepcopy(model)
                q_model.fp32_model = model.fp32_model
            except Exception as e:  # pragma: no cover
                logger.warning("Fail to deep copy the model due to {}, inplace is used now.".format(repr(e)))
                q_model = model
        q_model._model.eval()

        # For smoothquant optimized model
        recipe_cfgs = tune_cfg.get("recipe_cfgs", None)
        if (
            recipe_cfgs
            and recipe_cfgs.get("smooth_quant", False)
            and not recipe_cfgs["smooth_quant_args"]["folding"]
            and self.approach != "post_training_dynamic_quant"
        ):
            return self.qdq_quantize(q_model, tune_cfg)
        if recipe_cfgs and recipe_cfgs.get("smooth_quant", False) and recipe_cfgs["smooth_quant_args"]["folding"]:
            self._apply_pre_optimization(q_model, tune_cfg)

        self.tune_cfg = tune_cfg
        self.tune_cfg["approach"] = self.approach
        self.tune_cfg["reduce_range"] = REDUCE_RANGE
        self.tune_cfg["framework"] = "pytorch_fx"

        # PyTorch 1.13 and above version, need example_inputs for fx trace, but it not really used,
        # so set it to None.
        self.example_inputs = None
        if self.default_qconfig is not None:
            default_qconfig = copy.deepcopy(self.default_qconfig)
            default_qconfig["activation"]["dtype"] = self.default_qconfig["activation"]["dtype"][0]
            default_qconfig["weight"]["dtype"] = self.default_qconfig["weight"]["dtype"][0]
            self.tune_cfg["op"][("default_qconfig", "")] = default_qconfig
        op_cfgs = _cfg_to_qconfig(self.tune_cfg, self.approach)
        self.tune_cfg["bf16_ops_list"] = op_cfgs["bf16_ops_list"]
        del op_cfgs["bf16_ops_list"]
        gc.collect()

        from torch.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx

        if q_model.kwargs is not None:
            self.prepare_custom_config_dict = q_model.kwargs.get("prepare_custom_config_dict", None)
            self.convert_custom_config_dict = q_model.kwargs.get("convert_custom_config_dict", None)
        else:
            self.prepare_custom_config_dict, self.convert_custom_config_dict = None, None
        self.fx_op_cfgs = _cfgs_to_fx_cfgs(op_cfgs, self.approach)

        # for layer-wise quant
        # recipe_cfgs = tune_cfg.get('recipe_cfgs', None)
        if (
            recipe_cfgs
            and recipe_cfgs.get("layer_wise_quant", False)
            and self.approach != "post_training_dynamic_quant"
        ):
            from .torch_utils.layer_wise_quant import LayerWiseQuant

            # model_path = recipe_cfgs["layer_wise_quant_args"].get("model_path", None)
            model_path = model._model.path
            smooth_quant = recipe_cfgs["layer_wise_quant_args"].get("smooth_quant", False)
            alpha = recipe_cfgs["layer_wise_quant_args"].get("smooth_quant_alpha", 0.5)
            # device = recipe_cfgs["layer_wise_quant_args"].get("decvice", "cpu")
            assert model_path is not None, "The model_path should not be None."
            device = self.device
            lw_quant = LayerWiseQuant(
                q_model._model,
                model_path,
                self.fx_op_cfgs,
                calib_data=dataloader,
                device=device,
                smooth_quant=smooth_quant,
                alpha=alpha,
            )
            q_model._model = lw_quant.quantize(clean_weight=False)
            tune_cfg["recipe_cfgs"]["lwq_layers"] = lw_quant.quantized_layers
            q_model.q_config = copy.deepcopy(tune_cfg)
            return q_model

        self.tune_cfg["fx_sub_module_list"] = self.sub_module_list

        # BF16 fallback
        if (
            len(self.tune_cfg["bf16_ops_list"]) > 0
            and self.version.release >= Version("1.11.0").release
            and self.use_bf16
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
        ):  # pragma: no cover
            q_model._model = torch_utils.bf16_convert.Convert(q_model._model, self.tune_cfg)

        if self.approach == "quant_aware_training":
            q_model._model.train()
            if self.sub_module_list is None:
                tmp_model = q_model._model
                if self.version > Version("1.12.1"):  # pragma: no cover
                    # pylint: disable=E1123
                    q_model._model = prepare_qat_fx(
                        q_model._model,
                        self.fx_op_cfgs,
                        example_inputs=self.example_inputs,
                        prepare_custom_config=self.prepare_custom_config_dict,
                    )
                else:
                    q_model._model = prepare_qat_fx(  # pylint: disable=E1120,E1123
                        q_model._model, self.fx_op_cfgs, prepare_custom_config_dict=self.prepare_custom_config_dict
                    )
            else:
                logger.info("Fx trace of the entire model failed. " + "We will conduct auto quantization")
                PyTorch_FXAdaptor.prepare_sub_graph(
                    self.sub_module_list,
                    self.fx_op_cfgs,
                    q_model._model,
                    prefix="",
                    is_qat=True,
                    example_inputs=self.example_inputs,
                    custom_config=self.prepare_custom_config_dict,
                )
            # q_func can be created by neural_compressor internal or passed by user. It's critical to
            # distinguish how q_func is passed since neural_compressor built-in functions accept
            # neural_compressor model and user defined func should accept framework model.
            q_model._model = q_func(q_model if getattr(q_func, "builtin", None) else q_model._model)
            assert q_model._model is not None, "Please return a trained model in train function!"
            q_model._model.eval()
        else:
            if self.sub_module_list is None:
                tmp_model = q_model._model
                if self.version.release >= Version("1.13.0").release:  # pragma: no cover
                    # pylint: disable=E1123
                    q_model._model = prepare_fx(
                        q_model._model,
                        self.fx_op_cfgs,
                        example_inputs=self.example_inputs,
                        prepare_custom_config=self.prepare_custom_config_dict,
                    )
                else:
                    q_model._model = prepare_fx(  # pylint: disable=E1120,E1123
                        q_model._model, self.fx_op_cfgs, prepare_custom_config_dict=self.prepare_custom_config_dict
                    )
            else:
                logger.info("Fx trace of the entire model failed, " + "We will conduct auto quantization")
                PyTorch_FXAdaptor.prepare_sub_graph(
                    self.sub_module_list,
                    self.fx_op_cfgs,
                    q_model._model,
                    prefix="",
                    example_inputs=self.example_inputs,
                    custom_config=self.prepare_custom_config_dict,
                )
            if self.approach in ["post_training_static_quant", "post_training_auto_quant"]:
                iterations = tune_cfg.get("calib_iteration", 1)
                if q_func is not None:
                    q_func(q_model._model)
                else:
                    self.model_calibration(
                        q_model._model,
                        dataloader,
                        iterations,
                        calib_sampling_size=tune_cfg.get("calib_sampling_size", 1),
                    )

        if self.sub_module_list is None:
            if self.version.release >= Version("1.13.0").release:  # pragma: no cover
                # pylint: disable=E1123
                q_model._model = convert_fx(q_model._model, convert_custom_config=self.convert_custom_config_dict)
            else:
                q_model._model = convert_fx(  # pylint: disable=E1123
                    q_model._model, convert_custom_config_dict=self.convert_custom_config_dict
                )
            torch_utils.util.append_attr(q_model._model, tmp_model)
            del tmp_model
            gc.collect()
        else:
            PyTorch_FXAdaptor.convert_sub_graph(
                self.sub_module_list, q_model._model, prefix="", custom_config=self.prepare_custom_config_dict
            )

        self.fused_dict = self.get_fused_list(q_model.model)
        q_model.is_quantized = True
        q_model.q_config = copy.deepcopy(self.tune_cfg)
        if self.approach != "post_training_dynamic_quant":
            self._get_scale_zeropoint(q_model._model, q_model.q_config)

        self._dump_model_op_stats(q_model._model, q_model.q_config, self.approach)
        torch_utils.util.get_embedding_contiguous(q_model._model)
        return q_model

    def evaluate(
        self,
        model,
        dataloader,
        postprocess=None,
        metrics=None,
        measurer=None,
        iteration=-1,
        tensorboard=False,
        fp32_baseline=False,
    ):
        """Execute the evaluate process on the specified model.

        Args:
            model (object): model to run evaluation.
            dataloader (object): evaluation dataset.
            postprocess (object, optional): process function after evaluation.
            metric (object, optional): metric function.
            measurer (object, optional): measurer function.
            iteration (int, optional): number of iterations to evaluate.
            tensorboard (bool, optional): dump output tensor to tensorboard summary files.
            fp32_baseline (boolean, optional): only for compare_label=False pipeline

        Returns:
            (object): accuracy
        """
        if tensorboard:  # pragma: no cover
            assert False, "PyTorch FX mode didn't support tensorboard flag now!"
        self.is_baseline = fp32_baseline

        model_ = model._model
        assert isinstance(model_, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        model_.eval()
        model_.to(self.device)

        if metrics:
            self.fp32_preds_as_label = any(
                [hasattr(metric, "compare_label") and not metric.compare_label for metric in metrics]
            )

        return self.model_eval(model_, dataloader, postprocess, metrics, measurer, iteration)

    def _pre_hook_for_qat(self, dataloader=None):
        q_cfgs = (
            torch.quantization.QConfig(
                activation=torch.quantization.FakeQuantize.with_args(
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_affine,
                    reduce_range=REDUCE_RANGE,
                    observer=torch.quantization.MovingAverageMinMaxObserver,
                ),
                weight=torch.quantization.default_weight_fake_quant,
            )
            if self.version.release < Version("1.10.0").release
            else torch.quantization.QConfig(
                activation=torch.quantization.FusedMovingAvgObsFakeQuantize.with_args(
                    dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=REDUCE_RANGE
                ),
                weight=torch.quantization.default_fused_per_channel_wt_fake_quant,
            )
        )
        quantizable_ops = []
        tmp_model = self.fuse_fx_model(self.model, is_qat=True)
        self._get_quantizable_ops_recursively(tmp_model, "", quantizable_ops)
        self._remove_fallback_ops_for_qat(quantizable_ops)
        bf16_ops = []
        if (
            self.version.release >= Version("1.11.0").release
            and self.use_bf16
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
        ):  # pragma: no cover
            self.bf16_ops = self.query_handler.get_op_types_by_precision("bf16")
            self._get_bf16_ops_recursively(tmp_model, "", bf16_ops)
        bf16_ops_list = [(op) for op in bf16_ops if op not in quantizable_ops]
        quantized_ops = OrderedDict()
        for op in quantizable_ops:
            if op[1] in ["Embedding", "EmbeddingBag", "LSTM", "GRU", "LSTMCell", "GRUCell", "RNNCell"]:
                quantized_ops[op[0]] = torch.quantization.default_dynamic_qconfig
            else:
                quantized_ops[op[0]] = q_cfgs
        # build op_config_dict to save module scale and zeropoint
        op_config_dict = {}
        for op in quantizable_ops:
            op_config_dict[op] = {"weight": {"dtype": "int8"}, "activation": {"dtype": "uint8"}}

        if self.version.release < Version("1.11.0").release:  # pragma: no cover
            quantized_ops["default_qconfig"] = None
        else:
            from torch.ao.quantization import default_embedding_qat_qconfig

            for op in quantizable_ops:
                if op[1] in ["Embedding", "EmbeddingBag"]:
                    quantized_ops[op[0]] = default_embedding_qat_qconfig
        from torch.quantization.quantize_fx import prepare_qat_fx

        fx_op_cfgs = _cfgs_to_fx_cfgs(quantized_ops, "quant_aware_training")
        self.model._model.train()

        # PyTorch 1.13 and above version, need example_inputs for fx trace, but it not really used,
        # so set it to None.
        self.example_inputs = None

        # For export API, deepcopy fp32_model
        try:
            self.model.fp32_model = copy.deepcopy(self.model.fp32_model)
        except Exception as e:  # pragma: no cover
            logger.warning("Fail to deep copy the model due to {}, inplace is used now.".format(repr(e)))

        if self.sub_module_list is None:
            if self.version.release >= Version("1.13.0").release:  # pragma: no cover
                # pylint: disable=E1123
                self.model._model = prepare_qat_fx(
                    self.model._model,
                    fx_op_cfgs,
                    example_inputs=self.example_inputs,
                    prepare_custom_config=(
                        self.model.kwargs.get("prepare_custom_config_dict", None)
                        if self.model.kwargs is not None
                        else None
                    ),
                )
            else:
                self.model._model = prepare_qat_fx(  # pylint: disable=E1120,E1123
                    self.model._model,
                    fx_op_cfgs,
                    prepare_custom_config_dict=(
                        self.model.kwargs.get("prepare_custom_config_dict", None)
                        if self.model.kwargs is not None
                        else None
                    ),
                )
        else:
            logger.info("Fx trace of the entire model failed. " + "We will conduct auto quantization")
            PyTorch_FXAdaptor.prepare_sub_graph(
                self.sub_module_list,
                fx_op_cfgs,
                self.model._model,
                prefix="",
                is_qat=True,
                example_inputs=self.example_inputs,
            )
        # This is a flag for reloading
        self.model.q_config = {
            "calib_sampling_size": 100,  # tmp arg for export API
            "is_oneshot": True,
            "framework": "pytorch_fx",
            "reduce_range": REDUCE_RANGE,
            "quantizable_ops": quantizable_ops,
            "bf16_ops_list": bf16_ops_list,
            "op": op_config_dict,
            "sub_module_list": self.sub_module_list,
            "approach": "quant_aware_training",
        }

    def _post_hook_for_qat(self):
        from torch.quantization.quantize_fx import convert_fx

        if self.sub_module_list is None:
            if self.version > Version("1.12.1"):  # pragma: no cover
                # pylint: disable=E1123
                self.model._model = convert_fx(
                    self.model._model,
                    convert_custom_config=(
                        self.model.kwargs.get("convert_custom_config_dict", None)
                        if self.model.kwargs is not None
                        else None
                    ),
                )
            else:
                self.model._model = convert_fx(  # pylint: disable=E1123
                    self.model._model,
                    convert_custom_config_dict=(
                        self.model.kwargs.get("convert_custom_config_dict", None)
                        if self.model.kwargs is not None
                        else None
                    ),
                )
        else:
            PyTorch_FXAdaptor.convert_sub_graph(self.sub_module_list, self.model._model, prefix="")

        if self.approach != "post_training_dynamic_quant":
            self._get_scale_zeropoint(self.model._model, self.model.q_config)
        if (
            len(self.model.q_config["bf16_ops_list"]) > 0
            and self.version.release >= Version("1.11.0").release
            and self.use_bf16
            and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
        ):  # pragma: no cover
            self.model._model = torch_utils.bf16_convert.Convert(self.model._model, self.model.q_config)
        self._dump_model_op_stats(self.model._model, self.model.q_config, self.approach)
        torch_utils.util.get_embedding_contiguous(self.model._model)

    def _get_fallback_ops_for_qat(self):
        # get fallback ops for quant aware training approach
        fallback_ops = {"op_wise": [], "optype_wise": []}
        if self.qat_optype_wise is not None:  # pragma: no cover
            for optype, optype_config in self.qat_optype_wise.items():
                if "weight" in optype_config and optype_config["weight"]["dtype"] == ["fp32"]:
                    fallback_ops["optype_wise"].append(optype)
        if self.qat_op_wise is not None:  # pragma: no cover
            for op, op_config in self.qat_op_wise.items():
                if "weight" in op_config and op_config["weight"]["dtype"] == ["fp32"]:
                    fallback_ops["op_wise"].append(op)
        return fallback_ops

    def _remove_fallback_ops_for_qat(self, quantizable_ops):
        # remove fallback ops from quantizable_ops for quant aware training approach
        fallback_ops = self._get_fallback_ops_for_qat()
        remove_ops = []
        for op_name, op_type in quantizable_ops:
            if op_name in fallback_ops["op_wise"] or op_type in fallback_ops["optype_wise"]:
                remove_ops.append((op_name, op_type))
        for op_name, op_type in remove_ops:
            quantizable_ops.remove((op_name, op_type))

    def train(self, model, dataloader, optimizer_tuple, criterion_tuple, hooks, **kwargs):
        """Execute the train process on the specified model.

        Args:
            model (object): model to run evaluation.
            dataloader (object): training dataset.
            optimizer (tuple): It is a tuple of (cls, parameters) for optimizer.
            criterion (tuple): It is a tuple of (cls, parameters) for criterion.
            kwargs (dict, optional): other parameters.

        Returns:
            None
        """
        device = "cuda:0" if self.device != "GPU" and torch.cuda.is_available() else self.device
        self.model = model
        optimizer = optimizer_tuple[0](model._model.parameters(), **optimizer_tuple[1])
        criterion = criterion_tuple[0](**criterion_tuple[1])
        # prepare hooks first to ensure model will be converted correctly
        if hooks is not None:  # pragma: no cover
            on_train_begin = hooks["on_train_begin"]
            on_train_end = hooks["on_train_end"]
            on_epoch_begin = hooks["on_epoch_begin"]
            on_epoch_end = hooks["on_epoch_end"]
            on_step_begin = hooks["on_step_begin"]
            on_step_end = hooks["on_step_end"]
            on_after_compute_loss = hooks["on_after_compute_loss"]
            on_before_optimizer_step = hooks["on_before_optimizer_step"]
        model._model.train()
        if hooks is not None:
            on_train_begin(dataloader)
        start_epochs = kwargs["kwargs"]["start_epoch"]
        end_epochs = kwargs["kwargs"]["end_epoch"]
        iters = kwargs["kwargs"]["iteration"]
        model._model.to(device)
        for nepoch in range(start_epochs, end_epochs):
            cnt = 0
            if hooks is not None:
                on_epoch_begin(nepoch)
            for input, target in dataloader:
                target = target.to(device)
                if hooks is not None:
                    on_step_begin(cnt)
                print(".", end="", flush=True)
                cnt += 1
                output = pytorch_forward_wrapper(model._model, input)
                loss = criterion(output, target)
                if hooks is not None:
                    loss = on_after_compute_loss(input, output, loss)
                optimizer.zero_grad()
                loss.backward()
                if hooks is not None:
                    loss = on_before_optimizer_step()
                optimizer.step()
                if hooks is not None:
                    on_step_end()
                if cnt >= iters:
                    break
            if hooks is not None:
                on_epoch_end()

        if device != self.device:  # pragma: no cover
            model._model.to(self.device)

        if hooks is not None:
            on_train_end()

        return model._model

    def _get_module_op_stats(self, model, tune_cfg, approach):
        """This is a function to get quantizable ops of model to user.

        Args:
            model (object): input model
            tune_cfg (dict): quantization config
            approach (str): quantization approach
        Returns:
            None
        """
        modules = dict(model.named_modules())
        res = dict()

        if approach == "post_training_dynamic_quant":
            # fetch int8 and fp32 ops set by Neural Compressor from tune_cfg
            for key in tune_cfg["op"]:
                op_type = key[1]
                # build initial dict
                if op_type not in res.keys():  # pragma: no cover
                    res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
                value = tune_cfg["op"][key]
                # Special cases: QuantStub, Embedding
                if ("weight" in value and value["weight"]["dtype"] == "fp32") or (
                    "weight" not in value and value["activation"]["dtype"] == "fp32"
                ):
                    res[op_type]["FP32"] += 1
                elif value["activation"]["dtype"] == "bf16":  # pragma: no cover
                    res[op_type]["BF16"] += 1
                else:
                    res[op_type]["INT8"] += 1
        else:
            quantized_mode = False
            for node in model.graph.nodes:
                if node.op == "call_module":
                    if node.target not in modules:  # pragma: no cover
                        continue
                    op_class = type(modules[node.target])
                    op_type = str(op_class.__name__)
                    if "quantized" in str(op_class) or (quantized_mode and "pooling" in str(op_class)):
                        if op_type not in res.keys():
                            res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
                        res[op_type]["INT8"] += 1
                    elif op_class in self.white_list:
                        if op_type not in res.keys():
                            res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
                        res[op_type]["FP32"] += 1
                    continue
                elif node.op == "call_function":
                    op_type = str(node.target.__name__)
                else:
                    op_type = node.target
                # skip input and output
                if "quantize_per" not in op_type and not quantized_mode:
                    continue
                # skip zero_pioint and scale
                if "zero_point" in op_type or "scale" in op_type:
                    continue
                # build initial dict
                if op_type not in res.keys():
                    res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}

                if "quantize_per" in op_type and not quantized_mode:
                    quantized_mode = True
                elif "dequantize" in op_type and quantized_mode:
                    quantized_mode = False
                res[op_type]["INT8"] += 1
        return res

    def _get_sub_module_op_stats(self, model, tune_cfg, approach, res, prefix=""):
        """This is a function to get quantizable ops of sub modules to user recursively.

        Args:
            model (object): input model
            tune_cfg (dict): quantization config
            approach (str): quantization approach
            res (dict) : contains result of quantizable ops
            prefix (string): prefix of op name
        Returns:
            None
        """
        for name, module in model.named_children():
            op_name = prefix + "." + name if prefix != "" else name
            if op_name in self.sub_module_list:
                module_res = self._get_module_op_stats(module, tune_cfg, approach)
                for key, value in module_res.items():
                    if key in res:
                        res[key] = {k: res[key][k] + v for k, v in value.items()}
                    else:
                        res[key] = value
            else:
                self._get_sub_module_op_stats(module, tune_cfg, approach, res, op_name)

    def _dump_model_op_stats(self, model, tune_cfg, approach):
        """This is a function to dump quantizable ops of model to user.

        Args:
            model (object): input model
            tune_cfg (dict): quantization config
            approach (str): quantization approach
        Returns:
            None
        """
        if self.sub_module_list is None or self.approach == "post_training_dynamic_quant":
            res = self._get_module_op_stats(model, tune_cfg, approach)
        else:
            res = dict()
            self._get_sub_module_op_stats(model, tune_cfg, approach, res)

            if (
                self.use_bf16
                and (self.version.release >= Version("1.11.0").release)
                and (CpuInfo().bf16 or os.getenv("FORCE_BF16") == "1")
            ):  # pragma: no cover
                bf16_ops_list = tune_cfg["bf16_ops_list"]
                if len(bf16_ops_list) > 0:
                    for bf16_op in bf16_ops_list:
                        op_type = bf16_op[1]
                        if op_type in res.keys():
                            res[op_type]["BF16"] += 1
                            if res[op_type]["FP32"] > 0:
                                res[op_type]["FP32"] -= 1
                        else:
                            res[op_type] = {"INT8": 0, "BF16": 1, "FP32": 0}

        output_data = [
            [op_type, sum(res[op_type].values()), res[op_type]["INT8"], res[op_type]["BF16"], res[op_type]["FP32"]]
            for op_type in res.keys()
        ]

        Statistics(
            output_data, header="Mixed Precision Statistics", field_names=["Op Type", "Total", "INT8", "BF16", "FP32"]
        ).print_stat()

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """
        from .torch_utils.pattern_detector import TransformerBasedModelBlockPatternDetector
        from .torch_utils.util import get_op_type_by_name

        detector = TransformerBasedModelBlockPatternDetector(model)
        detect_result = detector.detect_block()
        attention_block = detect_result.get("attention_blocks", None)
        ffn_blocks = detect_result.get("ffn_blocks", None)
        logger.info(f"Attention Blocks: {len(attention_block)}")
        logger.info(f"FFN Blocks: {len(ffn_blocks)}")
        module_dict = dict(model.named_modules())
        for op_name, child in model.named_modules():
            if self.is_fused_module(child):
                for name, _ in child.named_children():
                    module_prefix = op_name + "." + name
                    if module_prefix in module_dict:
                        module_dict.pop(module_prefix)  # remove sub-modules of fused modules
        q_ops_set = set()
        for op_name, child in module_dict.items():
            if (
                type(child) in self.white_list
                and type(child) != torch.nn.Sequential
                and type(child) != torch.quantization.stubs.DeQuantStub
            ):
                quantizable_ops.append(
                    (
                        op_name,
                        (
                            unify_op_type_mapping[str(child.__class__.__name__)]
                            if str(child.__class__.__name__) in unify_op_type_mapping
                            else str(child.__class__.__name__)
                        ),
                    )
                )
                q_ops_set.add(op_name)
        # discard the op does not belong to quantizable_ops
        block_wise = [
            [
                (name, get_op_type_by_name(name, quantizable_ops))
                for name in block
                if get_op_type_by_name(name, quantizable_ops) is not None
            ]
            for block in ffn_blocks
        ]
        self.block_wise = block_wise

    def _get_module_scale_zeropoint(self, model, tune_cfg, prefix=""):
        """Get activation scale and zero_point for converted module.

        Args:
            model (dir): Int8 model converted from fp32 model.
                         scale and zero_point is set with calibration for each module
            tune_cfg (object): This file saves scale and zero_point of
                               output activation of each quantized module.
            prefix (string): prefix of op name

        Returns:
            None
        """
        # get scale and zero_point of modules.
        modules = dict(model.named_modules())
        for key in tune_cfg["op"]:
            if prefix:
                sub_name = key[0].replace(prefix + ".", "", 1)
            else:
                sub_name = key[0]
            if sub_name in modules:
                value = tune_cfg["op"][key]
                assert isinstance(value, dict)
                if hasattr(modules[sub_name], "scale"):
                    value["activation"]["scale"] = float(modules[sub_name].scale)
                if hasattr(modules[sub_name], "zero_point"):
                    value["activation"]["zero_point"] = int(modules[sub_name].zero_point)
        # get scale and zero_point of getattr ops (like quantize ops).
        for node in model.graph.nodes:
            if node.op == "get_attr":
                if prefix:
                    sub_name = prefix + "--" + node.target
                else:
                    sub_name = node.target
                if not hasattr(model, node.target):
                    continue
                if "scale" in node.target:
                    tune_cfg["get_attr"][sub_name] = float(getattr(model, node.target))
                elif "zero_point" in node.target:
                    tune_cfg["get_attr"][sub_name] = int(getattr(model, node.target))
                else:
                    pass

    def _get_sub_module_scale_zeropoint(self, model, tune_cfg, prefix=""):
        """Get activation scale and zero_point for converted sub modules recursively.

        Args:
            model (dir): Int8 model converted from fp32 model.
                        scale and zero_point is set with calibration for each module
            tune_cfg (object): This file saves scale and zero_point of \
                            output activation of each quantized module.
            prefix (string): prefix of op name

        Returns:
            None
        """
        for name, module in model.named_children():
            op_name = prefix + "." + name if prefix != "" else name
            if op_name in self.sub_module_list:
                self._get_module_scale_zeropoint(module, tune_cfg, op_name)
            else:
                self._get_sub_module_scale_zeropoint(module, tune_cfg, op_name)

    def _get_scale_zeropoint(self, model, tune_cfg):
        """Get activation scale and zero_point for converted model.

        Args:
            model (dir): Int8 model converted from fp32 model.
                        scale and zero_point is set with calibration for each module
            tune_cfg (object): This file saves scale and zero_point of \
                            output activation of each quantized module.

        Returns:
            None
        """
        tune_cfg["get_attr"] = {}
        if self.sub_module_list is None:
            self._get_module_scale_zeropoint(model, tune_cfg)
        else:
            self._get_sub_module_scale_zeropoint(model, tune_cfg)

    @staticmethod
    def prepare_sub_graph(
        sub_module_list, fx_op_cfgs, model, prefix, is_qat=False, example_inputs=None, custom_config=None
    ):
        """Static method to prepare sub modules recursively.

        Args:
            sub_module_list (list): contains the name of traceable sub modules
            fx_op_cfgs (dict, QConfigMapping): the configuration for prepare_fx quantization.
            model (dir): input model which is PyTorch model.
            prefix (string): prefix of op name
            is_qat (bool): whether it is a qat quantization
            example_inputs (tensor / tuple of tensor): example inputs
            custom_config (dict): custom non traceable module dict

        Returns:
            model (dir): output model which is a prepared PyTorch model.
        """
        import torch.quantization.quantization_mappings as tqqm
        from torch.quantization.quantize_fx import prepare_fx, prepare_qat_fx

        version = get_torch_version()
        fx_white_list = tqqm.get_default_qconfig_propagation_list()
        for name, module in model.named_children():
            op_name = prefix + "." + name if prefix != "" else name
            # skip custom non traceable module in fine-grained FX
            if custom_config:
                if (
                    "non_traceable_module_name" in custom_config
                    and op_name in custom_config["non_traceable_module_name"]
                ) or (
                    "non_traceable_module_class" in custom_config
                    and isinstance(module, tuple(custom_config["non_traceable_module_class"]))
                ):
                    continue
            if op_name in sub_module_list:
                # remove prefix in fx_op_cfgs
                version = get_torch_version()
                if version > Version("1.12.1"):  # pragma: no cover
                    from torch.ao.quantization import QConfigMapping

                    fx_sub_op_cfgs = QConfigMapping()
                    fx_sub_op_cfgs.set_global(None)
                    fx_op_cfgs_dict = fx_op_cfgs.to_dict()
                else:
                    fx_sub_op_cfgs = dict()
                    fx_sub_op_cfgs[""] = None
                    fx_sub_op_cfgs["module_name"] = []
                    fx_op_cfgs_dict = fx_op_cfgs

                for k, v in fx_op_cfgs_dict["module_name"]:
                    if op_name in k:
                        sub_name = k.replace(op_name + ".", "", 1)
                        if version > Version("1.12.1"):  # pragma: no cover
                            # pylint: disable=no-member
                            fx_sub_op_cfgs.set_module_name(sub_name, v)
                        else:
                            fx_sub_op_cfgs["module_name"].append((sub_name, v))

                if type(module) in fx_white_list and type(module) != torch.nn.Sequential:
                    # Don't really need a quant/dequant, just move nn.Embedding \
                    # to lower level for fx detection.
                    tmp_module = torch.quantization.QuantWrapper(module)
                else:
                    tmp_module = module
                # pylint: disable=E1123
                # pragma: no cover
                if is_qat:
                    module_pre = (
                        prepare_qat_fx(tmp_module, fx_sub_op_cfgs)  # pylint: disable=E1120
                        if version <= Version("1.12.1")
                        else prepare_qat_fx(tmp_module, fx_sub_op_cfgs, example_inputs=example_inputs)
                    )
                # pylint: disable=E1123
                # pragma: no cover
                else:
                    module_pre = (
                        prepare_fx(tmp_module, fx_sub_op_cfgs)  # pylint: disable=E1120
                        if version <= Version("1.12.1")
                        else prepare_fx(tmp_module, fx_sub_op_cfgs, example_inputs=example_inputs)
                    )
                torch_utils.util.append_attr(module_pre, module, fx_white_list)
                setattr(model, name, module_pre)
            else:
                PyTorch_FXAdaptor.prepare_sub_graph(
                    sub_module_list, fx_op_cfgs, module, op_name, is_qat, example_inputs=example_inputs
                )

    @staticmethod
    def convert_sub_graph(sub_module_list, model, prefix, custom_config=None):
        """Static method to convert sub modules recursively.

        Args:
            sub_module_list (list): contains the name of traceable sub modules
            model (dir): input model which is prepared PyTorch model.
            prefix (string): prefix of op name
            custom_config (dict): custom non traceable module dict

        Returns:
            model (dir): output model which is a converted PyTorch int8 model.
        """
        from torch.quantization.quantize_fx import convert_fx

        for name, module in model.named_children():
            op_name = prefix + "." + name if prefix != "" else name
            # skip custom non traceable module in fine-grained FX
            if custom_config:
                if (
                    "non_traceable_module_name" in custom_config
                    and op_name in custom_config["non_traceable_module_name"]
                ) or (
                    "non_traceable_module_class" in custom_config
                    and isinstance(module, tuple(custom_config["non_traceable_module_class"]))
                ):
                    continue
            if op_name in sub_module_list:
                module_con = convert_fx(module)
                torch_utils.util.append_attr(module_con, module)
                setattr(model, name, module_con)
            else:
                PyTorch_FXAdaptor.convert_sub_graph(sub_module_list, module, op_name)

    @dump_elapsed_time("Pass query framework capability")
    def query_fw_capability(self, model):
        """This is a helper function to get all quantizable ops from model.

        Args:
            model (object): input model which is Neural Compressor model

        Returns:
            q_capability (dictionary): tuning capability for each op from model.
        """
        self.pre_optimized_model = model
        tmp_model = model._model
        tmp_model = self.fuse_fx_model(model, is_qat=(self.approach == "quant_aware_training"))
        return self._get_quantizable_ops(tmp_model)

    def fuse_fx_model(self, model, is_qat):
        """This is a helper function to get fused fx model for PyTorch_FXAdaptor.

        Args:
            model (object): input model which is Neural Compressor model.
            is_qat (bool): check quantization approach is qat or not.

        Returns:
            fused_model (GraphModule): fused GraphModule model from torch.fx.
        """
        try:
            tmp_model = copy.deepcopy(model._model)
        except Exception as e:  # pragma: no cover
            tmp_model = model._model
            logger.warning("Deepcopy failed: {}, inplace=True now!".format(repr(e)))

        tmp_model.train() if is_qat else tmp_model.eval()
        from torch.fx import GraphModule
        from torch.quantization.quantize_fx import QuantizationTracer, _fuse_fx

        if model.kwargs is not None:
            prepare_custom_config_dict = model.kwargs.get("prepare_custom_config_dict", {})
        else:
            prepare_custom_config_dict = {}
        skipped_module_names = prepare_custom_config_dict.get("non_traceable_module_name", [])
        skipped_module_classes = prepare_custom_config_dict.get("non_traceable_module_class", [])
        try:
            tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)
            graph_module = GraphModule(tmp_model, tracer.trace(tmp_model))
            if self.version.release >= Version("1.13.0").release:  # pragma: no cover
                # pylint: disable=E1124, E1123
                fused_model = _fuse_fx(graph_module, is_qat, fuse_custom_config=prepare_custom_config_dict)
            elif self.version.release >= Version("1.11.0").release:  # pragma: no cover
                # pylint: disable=E1124
                fused_model = _fuse_fx(  # pylint: disable=E1123
                    graph_module, is_qat, fuse_custom_config_dict=prepare_custom_config_dict
                )
            else:
                fused_model = _fuse_fx(graph_module, prepare_custom_config_dict)
        except:
            self.sub_module_list = []
            module_dict = dict(tmp_model.named_modules())
            self._fuse_sub_graph(tmp_model, module_dict, prefix="", is_qat=is_qat)
            fused_model = tmp_model
        return fused_model

    def _fuse_sub_graph(self, model, module_dict, prefix, is_qat):
        """This is a helper function to get fused fx sub modules recursively for PyTorch_FXAdaptor.

        Args:
            model (object): input model which is PyTorch model.
            module_dict (dict): module dict of input model.
            prefix (string): prefix of op name.
            is_qat (bool): check quantization approach is qat or not.

        Returns:
            fused_model (GraphModule): fused GraphModule model from torch.fx.
        """
        import torch.quantization.quantization_mappings as tqqm
        from torch.quantization.quantize_fx import _fuse_fx

        fx_white_list = tqqm.get_default_qconfig_propagation_list()
        for name, module in model.named_children():
            # FX QAT cannot fallback nn.Dropout from train mode to eval
            if type(module) == torch.nn.Dropout:  # pragma: no cover
                continue
            op_name = prefix + "." + name if prefix != "" else name
            if op_name not in module_dict:
                continue
            if type(module) in fx_white_list and type(module) != torch.nn.Sequential:
                module = torch.quantization.QuantWrapper(module)
            if self._check_dynamic_control(module):
                self._fuse_sub_graph(module, module_dict, op_name, is_qat=is_qat)
            else:
                try:
                    graph_module = torch.fx.symbolic_trace(module)
                    if self.version.release >= Version("1.11.0").release:  # pragma: no cover
                        fused_model = _fuse_fx(graph_module, is_qat)
                    else:
                        fused_model = _fuse_fx(graph_module)  # pylint: disable=E1120
                    setattr(model, name, fused_model)
                    self.sub_module_list.append(op_name)
                except:
                    self._fuse_sub_graph(module, module_dict, op_name, is_qat)

    @staticmethod
    def _check_dynamic_control(module):
        """This is a helper function to check dynamic control in forward function of module.

        Args:
            module (object): input module which is PyTorch Module.

        Returns:
            fused_model (GraphModule): fused GraphModule model from torch.fx.
        """
        import inspect

        try:
            lines = inspect.getsource(module.forward)
            # Proxy obj. will always be detectd as `not None`.
            # Other situations could be detected by prepare_fx function.
            pattern = "is( not)? None"
            anws = re.search(pattern, lines)
            if anws:
                return True
        except:  # pragma: no cover
            logger.info("Module has no forward function")
        return False

    def get_output_op_names(self, *args, **kwargs):
        return None

    def calculate_op_sensitivity(
        self, model, dataloader, tune_cfg, output_op_names, confidence_batches, fallback=True, requantize_cfgs=None
    ):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): INC model containing fp32 model
            dataloader (string): dataloader contains real data.
            tune_cfg (dict): dictionary of tune configure for each op.
            fallback (bool): switch method in fallback stage and re-quantize stage

        Returns:
            ops_lst (list): sorted op list by sensitivity
        """
        from .torch_utils.util import get_fallback_order

        ordered_ops = get_fallback_order(
            self, model.model, dataloader, tune_cfg, confidence_batches, fallback, requantize_cfgs
        )
        return ordered_ops


@adaptor_registry
class PyTorchWeightOnlyAdaptor(TemplateAdaptor):
    """Adaptor of PyTorch framework, all PyTorch API is in this class.

    Args:
        framework_specific_info (dict): dictionary of tuning configure from yaml file.
    """

    def __init__(self, framework_specific_info):
        super(PyTorchWeightOnlyAdaptor, self).__init__(framework_specific_info)
        self.tune_cfg = None
        if self.device == "cpu":
            query_config_file = "pytorch_cpu.yaml"
        else:  # pragma: no cover
            assert False, "Unsupported this device {}".format(self.device)
        self.query_handler = PyTorchQuery(local_config_file=os.path.join(os.path.dirname(__file__), query_config_file))

        self.white_list = [torch.nn.Linear]
        # Contains parameters for algorithms such as AWQ, GPTQ, etc.
        self.recipes = framework_specific_info["recipes"]
        self.optype_statistics = None

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, calib_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization config.
            model (object): model need to do quantization.
            dataloader (object): calibration dataset.
            calib_func (objext, optional): calibration function for ease-of-use.

        Returns:
            (object): quantized model
        """

        assert isinstance(model._model, torch.nn.Module), "The model passed in is not the instance of torch.nn.Module"
        if self.performance_only:
            q_model = model
        else:
            try:
                q_model = copy.deepcopy(model)
            except Exception as e:  # pragma: no cover
                logger.warning("Fail to deep copy the model due to {}, inplace is used now.".format(repr(e)))
                q_model = model

        # For tensorboard display
        self.tune_cfg = tune_cfg
        self.tune_cfg["approach"] = self.approach
        self.tune_cfg["framework"] = "pytorch"
        assert self.approach == "post_training_weight_only", "Please make sure the approach is weight_only"

        all_algo = set()
        for key, config in tune_cfg["op"].items():
            op_name, op_type = key
            if config["weight"]["dtype"] == "fp32":
                continue
            else:
                dtype = config["weight"]["dtype"]
                if dtype in ["nf4", "fp4", "fp4_e2m1_bnb", "fp4_e2m1"]:
                    config["weight"]["bits"] = 4
                    config["weight"]["scheme"] = "sym"
                elif dtype in ["int4"]:
                    config["weight"]["bits"] = 4
                elif dtype in ["int8"]:
                    config["weight"]["bits"] = 8
                algorithm = config["weight"]["algorithm"]
                all_algo.add(algorithm)
        if len(all_algo):
            logger.info(f"All algorithms to do: {all_algo}")
        if "GPTQ" in all_algo:
            q_model._model, gptq_config = self.gptq_quantize(q_model._model, tune_cfg, dataloader)
            q_model.gptq_config = gptq_config
        if "TEQ" in all_algo:
            q_model._model = self.teq_quantize(q_model._model, tune_cfg, dataloader, calib_func)
        if "AWQ" in all_algo:  # includes RTN in AWQ
            q_model._model = self.awq_quantize(q_model._model, tune_cfg, dataloader, calib_func)
        if "RTN" in all_algo:
            q_model._model = self.rtn_quantize(q_model._model, tune_cfg)
        if "AUTOROUND" in all_algo:
            q_model._model, autoround_config = self.autoround_quantize(q_model._model, tune_cfg, dataloader)
            q_model.autoround_config = autoround_config

        q_model.q_config = copy.deepcopy(self.tune_cfg)
        q_model.is_quantized = True
        self._dump_model_op_stats(q_model._model, q_model.q_config)
        return q_model

    def rtn_quantize(self, model, tune_cfg):
        logger.info("quantizing with the round-to-nearest algorithm")
        if "rtn_args" in self.recipes:
            enable_full_range = self.recipes["rtn_args"].get("enable_full_range", False)
            enable_mse_search = self.recipes["rtn_args"].get("enable_mse_search", False)
            group_dim = self.recipes["rtn_args"].get("group_dim", 1)
            return_int = self.recipes["rtn_args"].get("return_int", False)
        else:  # pragma: no cover
            enable_full_range = False
            enable_mse_search = False
            group_dim = 1
            return_int = False
        from .torch_utils.util import fetch_module, set_module
        from .torch_utils.weight_only import rtn_quantize

        # for layer_wise quant mode
        recipe_cfgs = tune_cfg.get("recipe_cfgs", None)
        if recipe_cfgs.get("layer_wise_quant", False):
            from .torch_utils.layer_wise_quant.utils import LWQ_WORKSPACE, _get_path, load_module

            os.makedirs(LWQ_WORKSPACE, exist_ok=True)
            # model_path = recipe_cfgs["layer_wise_quant_args"].get("model_path", None)
            model_path = model.path
            assert model_path, "model_path should not be None."
            model_path = _get_path(model_path)

        for key, config in tune_cfg["op"].items():
            op_name, op_type = key
            if config["weight"]["dtype"] == "fp32":
                continue
            else:
                dtype = config["weight"]["dtype"]
                num_bits = config["weight"]["bits"]
                scheme = config["weight"]["scheme"]
                group_size = config["weight"]["group_size"]
                algorithm = config["weight"]["algorithm"]
                if algorithm != "RTN":
                    continue
                m = fetch_module(model, op_name)
                # load weight if use layer-wise quant mode
                recipe_cfgs = tune_cfg.get("recipe_cfgs", None)
                if recipe_cfgs.get("layer_wise_quant", False):
                    # load weight
                    load_module(model, op_name, model_path, device=self.device)
                m = rtn_quantize(
                    m,
                    num_bits,
                    group_size,
                    scheme,
                    return_int=return_int,
                    data_type=dtype,
                    enable_full_range=enable_full_range,
                    enable_mse_search=enable_mse_search,
                    group_dim=group_dim,
                )
                if recipe_cfgs.get("layer_wise_quant", False):
                    # save and clean weight
                    from .torch_utils.layer_wise_quant.utils import clean_module_weight

                    torch.save(m.state_dict(), os.path.join(LWQ_WORKSPACE, f"{op_name}.pt"))
                    clean_module_weight(m)
                set_module(model, op_name, m)
        if recipe_cfgs.get("layer_wise_quant", False):
            # register hooks
            from .torch_utils.layer_wise_quant.utils import register_weight_hooks

            register_weight_hooks(model, model_path, device=self.device, clean_weight=True)
        return model

    def gptq_quantize(self, model, tune_cfg, dataloader):
        logger.info("quantizing with the GPTQ algorithm")
        from .torch_utils.weight_only import gptq_quantize

        # convert tune_cfg to gptq_quantize's weight config
        """please refer to weight_config which can be analyzed by user-define API function weight_only.gptq_quantize
        keys of weight_config can not only be specific name, but can also be a re formula
        weight_config = {
            "layer_name_1": {
                'wbits': 4,
                'group_size': 128,
                'sym': False,
                'percdamp': 0.01,
                'actorder': True
            },
            "layer_name_2": {
                'wbits': 4,
                'group_size': 128,
                'sym': False,
                'percdamp': 0.01,
                'actorder': True
            }
            ...
        }
        """
        # for layer_wise quant mode
        recipe_cfgs = tune_cfg.get("recipe_cfgs", None)
        model_path = None
        layer_wise = False
        if recipe_cfgs.get("layer_wise_quant", False):
            layer_wise = True
            from .torch_utils.layer_wise_quant.utils import LWQ_WORKSPACE, _get_path, register_weight_hooks

            os.makedirs(LWQ_WORKSPACE, exist_ok=True)
            # model_path = recipe_cfgs["layer_wise_quant_args"].get("model_path", None)
            model_path = model.path
            assert model_path, "model_path should not be None."
            model_path = _get_path(model_path)
            lwq_handles = register_weight_hooks(
                model, model_path, device=self.device, clean_weight=True, saved_path=LWQ_WORKSPACE
            )

        weight_config = {}
        for key, config in tune_cfg["op"].items():
            op_name, op_type = key
            if config["weight"]["dtype"] == "fp32":
                continue  # no need to be quantized
            else:
                weight_config[op_name] = {
                    "wbits": config["weight"]["bits"],
                    "group_size": config["weight"]["group_size"],
                    "sym": config["weight"]["scheme"] == "sym",
                    "percdamp": self.recipes["gptq_args"].get("percdamp", 0.01),
                    "act_order": self.recipes["gptq_args"].get("act_order", False),
                    "block_size": self.recipes["gptq_args"].get("block_size", True),
                    "static_groups": self.recipes["gptq_args"].get("static_groups", False),
                    "true_sequential": self.recipes["gptq_args"].get("true_sequential", False),
                    "lm_head": self.recipes["gptq_args"].get("lm_head", False),
                }
        nsamples = self.recipes["gptq_args"].get("nsamples", 128)
        use_max_length = self.recipes["gptq_args"].get("use_max_length", False)
        pad_max_length = self.recipes["gptq_args"].get("pad_max_length", 2048)
        if use_max_length and "pad_max_length" not in self.recipes["gptq_args"]:
            logger.warning(
                "You choose to use unified sequence length for calibration, \
            but you have not set length value. Default sequence length is 2048 and this might cause inference error!"
            )
        # tune_cfg => weight_config
        model, quantization_perm = gptq_quantize(
            model,
            weight_config,
            dataloader,
            nsamples,
            use_max_length,
            pad_max_length,
            self.device,
            layer_wise,
            model_path,
        )
        return model, quantization_perm

    def teq_quantize(self, model, tune_cfg, dataloader, calib_func):
        logger.info("quantizing with the TEQ algorithm")
        from .torch_utils.weight_only import teq_quantize

        # get example inputs if not provided.
        if self.example_inputs is None:  # pragma: no cover
            if dataloader is None:
                assert False, "Please provide dataloader or example_inputs for TEQ algorithm."
            try:
                for idx, (x, label) in enumerate(dataloader):
                    self.example_inputs = x.to(model.device)
                    break
            except:
                for idx, x in enumerate(dataloader):
                    self.example_inputs = x.to(model.device)
                    break

        folding = True
        if "teq_args" in self.recipes:  # pragma: no cover
            folding = self.recipes["teq_args"].get("folding", True)

        supported_layers = ["Linear"]
        if folding:  # pragma: no cover
            from neural_compressor.adaptor.torch_utils.waq import GraphTrace

            tg = GraphTrace()
            absorb_to_layer, _ = tg.get_absorb_to_layer(model, self.example_inputs, supported_layers)
            if absorb_to_layer is None or absorb_to_layer == {}:
                logger.warning("No absorb layer is detected, skip TEQ algorithm")
                return model
        else:  # pragma: no cover
            absorb_to_layer = {}
            for name, module in model.named_modules():
                for op_type in supported_layers:
                    if op_type == str(module.__class__.__name__):
                        absorb_to_layer[name] = [name]

        # got flipped dict from absorb_to_layer dict
        flipped_dict = {}
        for k, v in absorb_to_layer.items():
            for m in v:
                flipped_dict[m] = {"absorb_layer": k}

        # check tune_cfg to skip layers without TEQ config
        weight_config = {}
        skipped_op_name_set = set()
        for key, config in tune_cfg["op"].items():
            op_name, op_type = key
            if config["weight"]["dtype"] == "fp32":  # pragma: no cover
                if op_name in flipped_dict:
                    absorb_to_layer.pop(flipped_dict[op_name]["absorb_layer"])
                continue
            else:
                weight_config[op_name] = {}
                weight_config[op_name]["bits"] = config["weight"]["bits"]
                weight_config[op_name]["group_size"] = config["weight"]["group_size"]
                weight_config[op_name]["scheme"] = config["weight"]["scheme"]
                if op_name in flipped_dict:
                    algorithm = config["weight"]["algorithm"]
                    if algorithm != "TEQ":
                        absorb_to_layer.pop(weight_config[op_name]["absorb_layer"])
                else:
                    skipped_op_name_set.add(op_name)
        if skipped_op_name_set:  # pragma: no cover
            logger.info("{} is skipped by TEQ algorithm".format(skipped_op_name_set))

        # collect TEQ config from tune_cfg for quantization.
        if len(absorb_to_layer) == 0:  # pragma: no cover
            logger.warning("No absorb layer needs TEQ algorithm, skip it")
        else:  # pragma: no cover
            logger.debug("**absorb layer**: **absorbed layers**")
        for k, v in absorb_to_layer.items():
            logger.debug(f"{k}: {v}")

        logger.info("Absorbed layers with the same absorb layer use the same config")

        extra_config = {"folding": folding}

        model = teq_quantize(
            model,
            weight_config,
            absorb_to_layer,
            extra_config,
            dataloader,
            example_inputs=self.example_inputs,
            calib_func=calib_func,
        )
        return model

    def awq_quantize(self, model, tune_cfg, dataloader, calib_func):
        logger.info("quantizing with the AWQ algorithm")
        from .torch_utils.weight_only import awq_quantize

        # get example inputs if not provided.
        if self.example_inputs is None:
            from neural_compressor.adaptor.torch_utils.util import get_example_input

            assert dataloader is not None, "datalaoder or example_inputs is required."
            self.example_inputs = get_example_input(dataloader)

        # build weight_config
        weight_config = {}
        for key, config in tune_cfg["op"].items():
            op_name, op_type = key
            if config["weight"]["dtype"] == "fp32":
                weight_config[op_name] = {
                    "bits": -1,  # skip quantization
                    "group_size": 128,
                    "scheme": "asym",
                    "algorithm": "RTN",
                }
            else:
                weight_config[op_name] = config["weight"]

        if "awq_args" in self.recipes:
            enable_auto_scale = self.recipes["awq_args"].get("enable_auto_scale", True)
            enable_mse_search = self.recipes["awq_args"].get("enable_mse_search", True)
            folding = self.recipes["awq_args"].get("folding", False)
        else:
            enable_auto_scale, enable_mse_search, folding = True, True, False
        if "rtn_args" in self.recipes:
            enable_full_range = self.recipes["rtn_args"].get("enable_full_range", False)
            return_int = self.recipes["rtn_args"].get("return_int", False)
        else:
            enable_full_range, return_int = False, False
        calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)
        model = awq_quantize(
            model,
            bits=-1,  # no quantize for op not in weight_config
            example_inputs=self.example_inputs,
            weight_config=weight_config,
            dataloader=dataloader,
            n_samples=calib_sampling_size,
            enable_auto_scale=enable_auto_scale,
            enable_mse_search=enable_mse_search,
            calib_func=calib_func,
            folding=folding,
            return_int=return_int,
            enable_full_range=enable_full_range,
        )
        return model

    def autoround_quantize(self, model, tune_cfg, dataloader):
        logger.info("quantizing with the AutoRound algorithm")
        from .torch_utils.weight_only import autoround_quantize

        # build weight_config
        """
            weight_config={
                        'layer1':##layer_name
                        {
                            'data_type': 'int',
                            'bits': 4,
                            'group_size': 32,
                            'scheme': "asym", ## or sym
                        }
                        ...
                    }
        """
        weight_config = {}
        for key, config in tune_cfg["op"].items():
            if config["weight"]["dtype"] == "fp32":
                continue
            op_name, op_type = key
            weight_config[op_name] = {}
            weight_config[op_name]["data_type"] = config["weight"]["dtype"]
            weight_config[op_name]["bits"] = config["weight"]["bits"]
            weight_config[op_name]["group_size"] = config["weight"]["group_size"]
            weight_config[op_name]["sym"] = config["weight"]["scheme"] == "sym"

        # auto round recipes

        enable_full_range = self.recipes["autoround_args"].get("enable_full_range", False)
        batch_size = self.recipes["autoround_args"].get("batch_size", 8)
        lr_scheduler = self.recipes["autoround_args"].get("lr_scheduler", None)
        dataset = self.recipes["autoround_args"].get("dataset", "NeelNanda/pile-10k")
        enable_quanted_input = self.recipes["autoround_args"].get("enable_quanted_input", True)
        enable_minmax_tuning = self.recipes["autoround_args"].get("enable_minmax_tuning", True)
        lr = self.recipes["autoround_args"].get("lr", None)
        minmax_lr = self.recipes["autoround_args"].get("minmax_lr", None)
        low_gpu_mem_usage = self.recipes["autoround_args"].get("low_gpu_mem_usage", False)
        iters = self.recipes["autoround_args"].get("iters", 200)
        seqlen = self.recipes["autoround_args"].get("seqlen", 2048)
        nsamples = self.recipes["autoround_args"].get("nsamples", 128)
        sampler = self.recipes["autoround_args"].get("sampler", "rand")
        seed = self.recipes["autoround_args"].get("seed", 42)
        nblocks = self.recipes["autoround_args"].get("nblocks", 1)
        gradient_accumulate_steps = self.recipes["autoround_args"].get("gradient_accumulate_steps", 1)
        not_use_best_mse = self.recipes["autoround_args"].get("not_use_best_mse", False)
        dynamic_max_gap = self.recipes["autoround_args"].get("dynamic_max_gap", -1)
        data_type = self.recipes["autoround_args"].get("data_type", "int")  ##only support data_type
        scale_dtype = self.recipes["autoround_args"].get("scale_dtype", "fp16")
        amp = self.recipes["autoround_args"].get("amp", True)
        device = self.recipes["autoround_args"].get("device", None)
        bits = self.recipes["autoround_args"].get("bits", 4)
        group_size = self.recipes["autoround_args"].get("group_size", 128)
        sym = self.recipes["autoround_args"].get("scheme", "asym") == "sym"
        act_bits = self.recipes["autoround_args"].get("act_bits", 32)
        act_group_size = self.recipes["autoround_args"].get("act_group_size", None)
        act_sym = self.recipes["autoround_args"].get("act_sym", None)
        act_dynamic = self.recipes["autoround_args"].get("act_dynamic", True)
        to_quant_block_names = self.recipes["autoround_args"].get("to_quant_block_names", None)
        use_layer_wise = self.recipes["autoround_args"].get("use_layer_wise", False)

        if dataloader is not None:
            dataset = dataloader
        model, autoround_config = autoround_quantize(
            model=model,
            bits=bits,
            group_size=group_size,
            sym=sym,
            weight_config=weight_config,
            enable_full_range=enable_full_range,
            batch_size=batch_size,
            amp=amp,
            device=device,
            lr_scheduler=lr_scheduler,
            dataset=dataset,
            enable_quanted_input=enable_quanted_input,
            enable_minmax_tuning=enable_minmax_tuning,
            lr=lr,
            minmax_lr=minmax_lr,
            low_gpu_mem_usage=low_gpu_mem_usage,
            iters=iters,
            seqlen=seqlen,
            nsamples=nsamples,
            sampler=sampler,
            seed=seed,
            nblocks=nblocks,
            gradient_accumulate_steps=gradient_accumulate_steps,
            not_use_best_mse=not_use_best_mse,
            dynamic_max_gap=dynamic_max_gap,
            data_type=data_type,
            scale_dtype=scale_dtype,
            to_quant_block_names=to_quant_block_names,
            act_bits=act_bits,
            act_group_size=act_group_size,
            act_sym=act_sym,
            act_dynamic=act_dynamic,
            use_layer_wise=use_layer_wise,
        )
        return model, autoround_config

    def _dump_model_op_stats(self, model, tune_cfg):
        """This is a function to dump quantizable ops of model to user.

        Args:
            model (object): input model
            tune_cfg (dict): quantization config
        Returns:
            None
        """
        res = {}
        # collect all dtype info and build empty results with existing op_type
        dtype_set = set()
        for op, config in tune_cfg["op"].items():
            op_type = op[1]
            if not config["weight"]["dtype"] == "fp32":
                num_bits = config["weight"]["bits"]
                group_size = config["weight"]["group_size"]
                dtype_str = "A32W{}G{}".format(num_bits, group_size)
                dtype_set.add(dtype_str)
        dtype_set.add("FP32")
        dtype_list = list(dtype_set)
        dtype_list.sort()
        for op, config in tune_cfg["op"].items():
            op_type = op[1]
            if op_type not in res.keys():
                res[op_type] = {dtype: 0 for dtype in dtype_list}

        # fill in results with op_type and dtype
        for op, config in tune_cfg["op"].items():
            if config["weight"]["dtype"] == "fp32":
                res[op_type]["FP32"] += 1
            else:
                num_bits = config["weight"]["bits"]
                group_size = config["weight"]["group_size"]
                dtype_str = "A32W{}G{}".format(num_bits, group_size)
                res[op_type][dtype_str] += 1

        # update stats format for dump.
        field_names = ["Op Type", "Total"]
        field_names.extend(dtype_list)
        output_data = []
        for op_type in res.keys():
            field_results = [op_type, sum(res[op_type].values())]
            field_results.extend([res[op_type][dtype] for dtype in dtype_list])
            output_data.append(field_results)

        Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()
        self.optype_statistics = field_names, output_data

    def _get_quantizable_ops_recursively(self, model, prefix, quantizable_ops):
        """This is a helper function for `query_fw_capability`,
           and it will get all quantizable ops from model.

        Args:
            model (object): input model
            prefix (string): prefix of op name
            quantizable_ops (list): list of quantizable ops from model include op name and type.

        Returns:
            None
        """

        module_dict = dict(model.named_modules())
        for op_name, child in module_dict.items():
            if isinstance(child, tuple(self.white_list)):
                quantizable_ops.append((op_name, str(child.__class__.__name__)))

    @dump_elapsed_time("Pass query framework capability")
    def query_fw_capability(self, model):
        """This is a helper function to get all quantizable ops from model.

        Args:
            model (object): input model which is Neural Compressor model

        Returns:
            q_capability (dictionary): tuning capability for each op from model.
        """
        self.pre_optimized_model = model
        return self._get_quantizable_ops(model.model)


class PyTorchQuery(QueryBackendCapability):
    def __init__(self, device="cpu", local_config_file=None):
        super().__init__()
        self.version = get_torch_version()
        self.cfg = local_config_file
        self.device = device
        self.cur_config = None
        self._one_shot_query()

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime.
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        # default_config = None
        for sub_data in data:
            if sub_data["version"]["name"] == "default":
                return sub_data
            sub_data_version = Version(sub_data["version"]["name"])
            if self.version >= sub_data_version:
                return sub_data

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e:  # pragma: no cover
                logger.info("Fail to parse {} due to {}".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError(
                    "Please check if the format of {} follows " "Neural Compressor yaml scheme.".format(self.cfg)
                )
        if self.device == "xpu":
            self.cur_config = self.cur_config[self.device]
        elif "cpu" in self.cur_config:
            self.cur_config = self.cur_config["cpu"]

    def get_quantization_capability(self, datatype="int8"):
        """Get the supported op types' quantization capability.

        Args:
            datatype: the data type. Defaults to 'int8'.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        assert (
            datatype in self.get_quant_datatypes()
        ), f"The target data type should be one of {self.get_quant_datatypes()}"
        return self.cur_config[datatype]

    def get_quant_datatypes(self):
        """Got low-precision data types for quantization.

        Collects all data types for quantization, such as int8, int4.
        """
        # TODO to handle other data types such FP8, FP8E4M3
        datatype_lst = []
        for key in self.cur_config:
            if key.startswith("int") or key == "weight_only_integer":
                datatype_lst.append(key)
        return datatype_lst

    def get_op_types(self):
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return self.cur_config

    def get_op_types_by_precision(self, precision):
        """Get op types per precision
        Args:
            precision (string): precision name
        Returns:
            [string list]: A list composed of op type.
        """
        return self.cur_config[precision]
