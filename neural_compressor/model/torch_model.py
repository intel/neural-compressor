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
"""Class for PyTorch model."""

import copy
import inspect
import os
import sys
from collections import OrderedDict, UserDict

from neural_compressor import config as cfg
from neural_compressor.model.base_model import BaseModel
from neural_compressor.utils import logger
from neural_compressor.utils.utility import LazyImport, compute_sparsity

torch = LazyImport("torch")
yaml = LazyImport("yaml")
json = LazyImport("json")
np = LazyImport("numpy")
onnx = LazyImport("onnx")
ort = LazyImport("onnxruntime")
ortq = LazyImport("onnxruntime.quantization")


class PyTorchBaseModel(torch.nn.Module, BaseModel):
    """Build PyTorch base model."""

    def __init__(self, model, **kwargs):
        """Initialize a PyTorch model.

        Args:
            model (torch.nn.model): torch.nn.model instance.
        """
        torch.nn.Module.__init__(self)
        self._model = model
        assert isinstance(model, torch.nn.Module), "model should be pytorch nn.Module."
        self._model_path = None if not isinstance(model, str) else model
        self.handles = []
        self.tune_cfg = None
        self.q_config = None
        self._workspace_path = ""
        self.is_quantized = False
        self.fp32_model = model
        self.kwargs = kwargs if kwargs else None

    def __repr__(self):
        """Describe a PyTorchBaseModel as a string."""
        # rewrite this func to avoid printing fp32_model
        from torch.nn.modules.module import _addindent

        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            if key == "fp32_model":
                continue
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines
        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def forward(self, *args, **kwargs):
        """Pytorch model forward func."""
        return self._model(*args, **kwargs)

    @property
    def model(self):
        """Getter to model."""
        return self._model

    @model.setter
    def model(self, model):
        """Setter to model."""
        self._model = model

    @property
    def model_path(self):
        """Return model path."""
        return self._model_path

    @model_path.setter
    def model_path(self, path):
        """Set model path."""
        self._model_path = path

    @property
    def fp32_model(self):
        """Getter to model."""
        return self._fp32_model

    @fp32_model.setter
    def fp32_model(self, fp32_model):
        """Setter to model."""
        self._fp32_model = fp32_model

    def register_forward_pre_hook(self):
        """Register forward pre hook."""
        self.handles.append(self._model.register_forward_pre_hook(self.generate_forward_pre_hook()))

    def remove_hooks(self):
        """Remove hooks."""
        for handle in self.handles:
            handle.remove()

    def generate_forward_pre_hook(self):
        """Generate forward pre hook."""
        # skip input argument 'self' in forward
        self.input_args = OrderedDict().fromkeys(inspect.getfullargspec(self._model.forward).args[1:], None)

        # a wrapper is needed to insert self into the actual hook
        def actual_forward_pre_hook(module, input):
            args, _, _, values = inspect.getargvalues(inspect.stack()[1].frame)
            # intersection update kw arguments
            self.input_args.update(values["kwargs"])
            # update arguments
            if "input" in values:
                for single_input, single_arg in zip(
                    values["input"], list(self.input_args.keys())[: len(values["input"])]
                ):
                    self.input_args[single_arg] = single_input
            elif "args" in values:
                for single_input, single_arg in zip(
                    values["args"], list(self.input_args.keys())[: len(values["args"])]
                ):
                    self.input_args[single_arg] = single_input
            else:
                assert False, "there is no input field was found!"

        return actual_forward_pre_hook

    def framework(self):
        """Return framework."""
        return "pytorch"

    def get_all_weight_names(self):
        """Get weight names."""
        names = []
        for name, param in self._model.named_parameters():
            names.append(name)
        return names

    def get_weight(self, tensor_name):
        """Get weight value."""
        state_dict = self._model.state_dict()
        for name, tensor in state_dict.items():
            if tensor_name == name:
                return tensor.cpu()

    def update_weights(self, tensor_name, new_tensor):
        """Update weight value.

        Args:
            tensor_name (string): weight name.
            new_tensor (ndarray): weight value.
        """
        # TODO: copy tensor option to new tensor is better
        device = next(self._model.parameters()).device
        new_tensor = torch.tensor(new_tensor).float().to(device)
        module_index = ".".join(tensor_name.split(".")[:-1])
        module = dict(self._model.named_modules())[module_index]
        getattr(module, tensor_name.split(".")[-1]).data = new_tensor.data

    def update_gradient(self, grad_name, new_grad):
        """Update grad value.

        Args:
            grad_name (str): grad name.
            new_grad (ndarray): grad value.
        """
        device = next(self._model.parameters()).device
        new_grad = torch.tensor(new_grad).float().to(device)
        params = [p for n, p in self._model.named_parameters() if n == grad_name]
        assert len(params) == 1, "lpot can only update grad of one tensor at one time"
        param = params[0]
        param.grad.copy_(new_grad)

    def prune_weights_(self, tensor_name, mask):
        """Prune weight in place according to tensor_name with mask.

        Args:
            tensor_name (str): weight name.
            mask (tensor): pruning mask.
        """
        state_dict = self._model.state_dict()
        for name in state_dict:
            if name == tensor_name:
                state_dict[name].masked_fill_(mask.to(state_dict[name].device), 0.0)

    def get_inputs(self, input_name=None):
        """Get inputs of model.

        Args:
            input_name (str, optional): name of input tensor. Defaults to None.

        Returns:
            tensor: input tensor
        """
        return self.input_args[input_name].cpu()

    def get_gradient(self, input_tensor):
        """Get gradients of specific tensor.

        Args:
            input_tensor (string or tensor): weight name or a tensor.

        Returns:
            ndarray: gradient tensor array
        """
        if isinstance(input_tensor, str):
            for name, tensor in self._model.named_parameters():
                if name == input_tensor:
                    assert tensor.grad is not None, "Please call backward() before get_gradient"
                    return np.array(tensor.grad.cpu())
        elif isinstance(input_tensor, torch.Tensor):
            assert input_tensor.grad is not None, "Please call backward() before get_gradient"
            return np.array(input_tensor.grad.cpu())
        else:  # pragma: no cover
            logger.error("Expect str or torch.Tensor in get_gradient, " "but get {}.".format(type(input_tensor)))

    def report_sparsity(self):
        """Get sparsity of the model.

        Returns:
            df (DataFrame): DataFrame of sparsity of each weight.
            total_sparsity (float): total sparsity of model.
        """
        if isinstance(self._model, torch.jit._script.RecursiveScriptModule):
            logger.info("INC IPEX don't support compute sparsity for model in TorchScript format now.")
            return [0.0]
        import pandas as pd

        df = pd.DataFrame(columns=["Name", "Shape", "NNZ (dense)", "NNZ (sparse)", "Sparsity(%)"])
        pd.set_option("display.precision", 2)
        # TODO: need to specify modules(Conv2d, Linear, etc.) instead of dims
        param_dims = [2, 4]
        params_size = 0
        sparse_params_size = 0
        model_params = dict(self._model.state_dict())
        for name, param in model_params.items():
            # '_packed_params._packed_params' and dtype is specific for quantized module
            if "_packed_params._packed_params" in name and isinstance(param, tuple):
                param = param[0]
            if hasattr(param, "dtype") and param.dtype in [torch.qint8, torch.quint8]:
                param = param.dequantize()
            if (
                hasattr(param, "dim")
                and param.dim() in param_dims
                and any(type in name for type in ["weight", "bias", "_packed_params"])
            ):
                param_size, sparse_param_size, dense_param_size = compute_sparsity(param.detach().cpu().numpy())
                density = dense_param_size / param_size
                params_size += param_size
                sparse_params_size += sparse_param_size
                df.loc[len(df.index)] = [
                    name,
                    list(param.shape),
                    dense_param_size,
                    sparse_param_size,
                    (1 - density) * 100,
                ]

        total_sparsity = sparse_params_size / params_size * 100

        df.loc[len(df.index)] = [
            "Total sparsity:",
            "-",
            params_size,
            sparse_params_size,
            total_sparsity,
        ]
        return df, total_sparsity


class PyTorchModel(PyTorchBaseModel):
    """Build PyTorchModel object."""

    def __init__(self, model, **kwargs):
        """Initialize PyTorchModel object."""
        super(PyTorchModel, self).__init__(model, **kwargs)

    @property
    def workspace_path(self):
        """Return workspace path."""
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path):
        """Set workspace path."""
        from neural_compressor.utils.pytorch import load

        workspace_path = path
        weights_file = os.path.join(os.path.abspath(os.path.expanduser(workspace_path)), "best_model.pt")
        assert os.path.exists(weights_file), "weight file %s didn't exist" % weights_file
        self._model = load(weights_file, self._model)

    def save(self, root=None):
        """Save configure file and weights."""
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        try:
            stat_dict = self._model.state_dict()
            if self.q_config:
                if self.q_config["approach"] == "post_training_weight_only":
                    from ..adaptor.torch_utils.util import collect_weight_info

                    weight_config_path = os.path.join(root, "qconfig.json")
                    weight_config = collect_weight_info(self.model, self.q_config)
                    with open(weight_config_path, "w") as f:
                        json.dump(weight_config, f, indent=4)
                    if hasattr(self, "gptq_config") and self.gptq_config:
                        gptq_config_path = os.path.join(root, "gptq_config.json")
                        with open(gptq_config_path, "w") as f:
                            json.dump(self.gptq_config, f, indent=4)
                    # for layer_wise quant mode
                    if self.q_config["recipe_cfgs"].get("layer_wise_quant", False):
                        from ..adaptor.torch_utils.layer_wise_quant.utils import (
                            LWQ_WORKSPACE,
                            _get_path,
                            get_named_children,
                            load_value,
                            set_module_tensor_to_device,
                        )

                        modules = get_named_children(self._model)
                        for name, module in modules:
                            state_dict = None
                            if os.path.exists(os.path.join(LWQ_WORKSPACE, f"{name}.pt")):
                                state_dict = torch.load(os.path.join(LWQ_WORKSPACE, f"{name}.pt"))
                            model_path = _get_path(
                                # self.q_config["recipe_cfgs"]["layer_wise_quant_args"].get("model_path")
                                self._model.path
                            )
                            for n, p in module.named_parameters():
                                param_name = name + "." + n
                                if state_dict:
                                    value = state_dict[n]
                                else:
                                    value = load_value(self._model, param_name, model_path)
                                # set_module_tensor_to_device(self._model, param_name, "cpu", value)
                                torch.save(value, os.path.join(root, f"{param_name}.pt"))
                        # stat_dict = self._model.state_dict()
                        return
                else:
                    stat_dict["best_configure"] = self.q_config
            torch.save(stat_dict, os.path.join(root, "best_model.pt"))
            logger.info("Save config file and weights of quantized model to {}.".format(root))
        except IOError as e:  # pragma: no cover
            logger.error("Fail to save configure file and weights due to {}.".format(e))

    def quantized_state_dict(self):
        """Load quantized state dict."""
        try:
            stat_dict = self._model.state_dict()
            stat_dict["best_configure"] = self.q_config
        except IOError as e:  # pragma: no cover
            logger.error("Fail to dump configure and weights due to {}.".format(e))
        return stat_dict

    def load_quantized_state_dict(self, stat_dict):
        """Load quantized state with given dict."""
        from ..utils.pytorch import load

        self.q_config = stat_dict["best_configure"]
        self._model = load(stat_dict, self._model)

    @property
    def graph_info(self):
        """Return graph info."""
        from ..adaptor.pytorch import get_ops_recursively

        op_map = {}
        get_ops_recursively(self._model, "", op_map)
        return op_map

    def export(
        self,
        save_path: str,
        conf,
    ):
        """Export PyTorch model to ONNX model."""
        from packaging.version import Version

        from ..adaptor.pytorch import get_torch_version

        version = get_torch_version()
        if version.release < Version("1.12.0").release:  # pragma: no cover
            assert False, (
                "PyTorch to ONNX export function requires a minimum torch version of {}, "
                "but the torch version found is {}".format(Version("1.12.0"), version)
            )

        from neural_compressor.utils.export import torch_to_fp32_onnx, torch_to_int8_onnx

        if conf.dtype == "int8":
            torch_to_int8_onnx(
                self.fp32_model,
                self.model,
                save_path,
                conf.example_inputs,
                self.q_config,
                opset_version=conf.opset_version,
                dynamic_axes=conf.dynamic_axes,
                input_names=conf.input_names,
                output_names=conf.output_names,
                quant_format=conf.quant_format,
                weight_type=conf.kwargs.get("weight_type", "S8"),
                verbose=True,
            )
        elif conf.dtype == "fp32":
            torch_to_fp32_onnx(
                self.model,
                save_path,
                conf.example_inputs,
                opset_version=conf.opset_version,
                dynamic_axes=conf.dynamic_axes,
                input_names=conf.input_names,
                output_names=conf.output_names,
                do_constant_folding=True,
                verbose=True,
            )
        else:  # pragma: no cover
            assert False, "Not allowed dtype: {}, please use 'fp32' or 'int8'.".format(conf.dtype)

    def export_compressed_model(
        self,
        qweight_config_path=None,
        enable_full_range=False,
        compression_dtype=torch.int32,
        compression_dim=1,
        scale_dtype=torch.float32,
        gptq_config_path=None,
        device="cpu",
        use_optimum_format=True,
    ):
        """Convert Linear to WeightOnlyLinear for low memory inference.

        Args:
            qweight_config_path (str, optional): Path of qconfig.json. Defaults to None.
            enable_full_range (bool, optional): Whether to leverage the full compression range
                                             under symmetric quantization. Defaults to False.
            compression_dtype (torch.Tensor, optional): The target dtype after comoression.
                                                        Defaults to torch.int32.
            compression_dim (int, optional): Select from [0, 1], 0 is output channel,
                                                1 is input channel. Defaults to 1.
            scale_dtype (torch.Tensor, optional): Use float32 or float16.
                                                    Defaults to torch.float32.
            gptq_config_path (str, optional): Path of gptq_config.json. Defaults to None.
            device (str, optional): choose device for compression. Defaults to cpu.
            use_optimum_format (bool, optional): use the popular huggingface compression format.
                1: compression_dim: weight = 1, zeros = 0 and both are transposed.
                2: zeros -= 1 before compression. Why we need it?
                3: g_idx: use same number for one group instead of recording the channel order.
                4. parameter name changed, such as 'packed_weight' -> 'qweight'.
                5. zeros is always needed even for sym.
        """
        from ..adaptor.torch_utils.model_wrapper import WeightOnlyLinear
        from ..adaptor.torch_utils.util import collect_weight_info, fetch_module, set_module
        from ..adaptor.torch_utils.weight_only import quant_weight_w_scale, rtn_quantize

        if qweight_config_path is not None:
            with open(qweight_config_path, "r") as f:
                weight_config = json.load(f)
        else:
            weight_config = collect_weight_info(self.model, self.q_config)
        if gptq_config_path is not None:
            with open(gptq_config_path, "r") as f:
                gptq_config = json.load(f)
        else:
            gptq_config = self.gptq_config if hasattr(self, "gptq_config") else {}

        autoround_config = self.autoround_config if hasattr(self, "autoround_config") else {}
        # check available device, priority: ["xpu", "cuda", "cpu"]
        availiable_device = []
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            availiable_device.append("xpu")
        if torch.cuda.is_available():
            availiable_device.append("cuda")
        availiable_device.append("cpu")
        orig_device = device
        if device not in availiable_device and "cuda" not in device:  # cuda in cuda:0
            logger.info(f"{device} is not detected in current environment, please check.")
            device = availiable_device[0]
            logger.info(f"The compression device has been changed to {device}.")
        if gptq_config:
            for k, v in weight_config.items():
                logger.debug(f"Compressing {k} on device {device}")
                if v["dtype"] == "fp32":
                    continue
                else:
                    dtype = v["dtype"]
                    num_bits = v["bits"]
                    group_size = v["group_size"]
                    scheme = v["scheme"]
                m = fetch_module(self.model, k)
                if k not in gptq_config:
                    new_module = rtn_quantize(
                        m,
                        num_bits,
                        group_size,
                        scheme,
                        data_type=dtype,
                        return_int=True,
                        enable_full_range=enable_full_range,
                        compression_dtype=compression_dtype,
                        compression_dim=compression_dim,
                        scale_dtype=scale_dtype,
                        device=device,
                        use_optimum_format=use_optimum_format,
                    )
                    set_module(self.model, k, new_module)
                    continue
                gptq_conf = gptq_config[k]
                if "perm" in gptq_conf:
                    gptq_perm = torch.tensor(gptq_conf["perm"])
                    fp32_weight = m.weight.data[:, gptq_perm]
                else:
                    fp32_weight = m.weight.data
                    gptq_perm = None
                gptq_scale = torch.tensor(gptq_conf["scale"], dtype=torch.float32)
                gptq_zp = None if scheme == "sym" else torch.tensor(gptq_conf["zero"], dtype=torch.int32)
                int_weight = quant_weight_w_scale(fp32_weight, gptq_scale, gptq_zp, group_size)
                int_weight = int_weight.type(torch.int32)
                if "perm" in gptq_conf:
                    invperm = torch.argsort(gptq_perm)
                    int_weight = int_weight[:, invperm]
                new_module = WeightOnlyLinear(
                    m.in_features,
                    m.out_features,
                    num_bits,
                    group_size,
                    dtype=dtype,
                    zp=gptq_zp is not None,
                    bias=m.bias is not None,
                    g_idx=gptq_perm is not None,
                    compression_dtype=compression_dtype,
                    compression_dim=compression_dim,
                    scale_dtype=scale_dtype,
                    device=device,
                    use_optimum_format=use_optimum_format,
                )
                new_module.pack(int_weight, gptq_scale, gptq_zp, m.bias, gptq_perm)
                set_module(self.model, k, new_module)
        elif autoround_config:
            if orig_device == "xpu":
                for k, v in weight_config.items():
                    logger.debug(f"Compressing {k} on device {device}")
                    if v["dtype"] == "fp32":
                        continue
                    else:
                        dtype = v["dtype"]
                        num_bits = v["bits"]
                        group_size = v["group_size"]
                        scheme = v["scheme"]
                    m = fetch_module(self.model, k)
                    autoround_conf = autoround_config[k]
                    fp32_weight = m.weight.data
                    autoround_scale = torch.tensor(autoround_conf["scale"], dtype=torch.float32)
                    autoround_zp = None if scheme == "sym" else torch.tensor(autoround_conf["zero"], dtype=torch.int32)
                    int_weight = quant_weight_w_scale(fp32_weight, autoround_scale, autoround_zp, group_size)
                    int_weight = int_weight.type(torch.int32)
                    new_module = WeightOnlyLinear(
                        m.in_features,
                        m.out_features,
                        num_bits,
                        group_size,
                        dtype=dtype,
                        zp=autoround_zp is not None,
                        bias=m.bias is not None,
                        g_idx=None,
                        compression_dtype=compression_dtype,
                        compression_dim=compression_dim,
                        scale_dtype=scale_dtype,
                        device=device,
                        use_optimum_format=use_optimum_format,
                    )
                    new_module.pack(int_weight, autoround_scale, autoround_zp, m.bias, None)
                    set_module(self.model, k, new_module)
            else:
                from auto_round.export.export_to_itrex.export import pack_model  # pylint: disable=E0401

                self.model = pack_model(
                    self.model,
                    layer_config=autoround_config,
                    enable_full_range=enable_full_range,
                    compression_dtype=compression_dtype,
                    compression_dim=compression_dim,
                    device=device,
                    use_optimum_format=use_optimum_format,
                    inplace=True,
                )
        else:
            for k, v in weight_config.items():
                logger.debug(f"Compressing {k} on device {device}")
                if v["dtype"] == "fp32":
                    continue
                else:
                    dtype = v["dtype"]
                    num_bits = v["bits"]
                    group_size = v["group_size"]
                    scheme = v["scheme"]
                mod = fetch_module(self.model, k)
                mod = rtn_quantize(
                    mod,
                    num_bits,
                    group_size,
                    scheme,
                    data_type=dtype,
                    return_int=True,
                    enable_full_range=enable_full_range,
                    compression_dtype=compression_dtype,
                    compression_dim=compression_dim,
                    scale_dtype=scale_dtype,
                    device=device,
                    use_optimum_format=use_optimum_format,
                )
                set_module(self.model, k, mod)
        return self.model


class PyTorchFXModel(PyTorchModel):
    """Build PyTorchFXModel object."""

    def __init__(self, model, **kwargs):
        """Initialize PyTorchFXModel object."""
        super(PyTorchFXModel, self).__init__(model, **kwargs)


class IPEXModel(PyTorchBaseModel):  # pragma: no cover
    """Build IPEXModel object."""

    def __init__(self, model, **kwargs):
        """Initialize IPEXModel object."""
        super(IPEXModel, self).__init__(model, **kwargs)
        self.ipex_config_path = None

    @property
    def _graph_info(self):
        pass

    @property
    def workspace_path(self):
        """Return workspace path."""
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path):
        """Set workspace path."""
        self._workspace_path = path
        tune_cfg_file = os.path.join(os.path.abspath(os.path.expanduser(path)), "best_configure.json")
        assert os.path.exists(tune_cfg_file), "tune configure file %s didn't exist" % tune_cfg_file

        with open(tune_cfg_file, "r") as f:
            self.tune_cfg = json.load(f)

    def save(self, root=None):
        """Save PyTorch IPEX model."""
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        os.makedirs(root, exist_ok=True)
        try:
            with open(os.path.join(root, "best_configure.json"), "w") as f:
                json.dump(self.tune_cfg, f, indent=4)
            logger.info("Save config file of quantized model to {}.".format(root))
        except IOError as e:
            logger.error("Fail to save configure file and weights due to {}.".format(e))

        if isinstance(self.model, torch.jit._script.RecursiveScriptModule):
            self.model.save(os.path.join(root, "best_model.pt"))
