# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WOQ save and load."""
# pylint:disable=import-error

import copy
import json
import os
import re
import tempfile

import torch

from neural_compressor.common.utils import AWQ, TEQ, save_config_mapping
from neural_compressor.torch.utils import (
    HPU_SAFE_WEIGHTS_NAME,
    HPU_WEIGHT_NAME,
    QCONFIG_NAME,
    WEIGHT_NAME,
    SaveLoadFormat,
    logger,
    set_module,
)

from .modules import HPUWeightOnlyLinear, INCWeightOnlyLinear, MulLinear
from .utility import convert_dtype_str2torch

format_woqlinear_mapping = {
    SaveLoadFormat.HUGGINGFACE: INCWeightOnlyLinear,
    SaveLoadFormat.DEFAULT: INCWeightOnlyLinear,
}
device_woqlinear_mapping = {"cpu": INCWeightOnlyLinear, "hpu": HPUWeightOnlyLinear}


def save(model, output_dir="./saved_results", format=SaveLoadFormat.DEFAULT, **kwargs):
    """Save the quantized model and config to the output path.

    Args:
        model (torch.nn.module): raw fp32 model or prepared model.
        output_dir (str, optional): output path to save.
        format (str, optional): The format in which to save the model. Options include "default" and "huggingface". Defaults to "default".
        kwargs: Additional arguments for specific formats. For example:
            - safe_serialization (bool): Whether to use safe serialization when saving (only applicable for 'huggingface' format). Defaults to True.
            - tokenizer (Tokenizer, optional): The tokenizer to be saved along with the model (only applicable for 'huggingface' format).
            - max_shard_size (str, optional): The maximum size for each shard (only applicable for 'huggingface' format). Defaults to "5GB".
    """
    os.makedirs(output_dir, exist_ok=True)
    if format == SaveLoadFormat.HUGGINGFACE:  # pragma: no cover
        config = model.config
        config_file = "quantize_config.json"
        quantization_config = config.quantization_config if hasattr(config, "quantization_config") else None
        if "backend" in quantization_config and "auto_round" in quantization_config["backend"]:
            safe_serialization = kwargs.get("safe_serialization", True)
            tokenizer = kwargs.get("tokenizer", None)
            max_shard_size = kwargs.get("max_shard_size", "5GB")
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
            del model.save
            model.save_pretrained(output_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
            with open(os.path.join(output_dir, config_file), "w", encoding="utf-8") as f:
                json.dump(quantization_config, f, indent=2)
            return

    qmodel_weight_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), QCONFIG_NAME)
    # saving process
    save_config_mapping(model.qconfig, qconfig_file_path)

    # MethodType 'save' not in state_dict
    del model.save
    torch.save(model.state_dict(), qmodel_weight_file_path)

    logger.info("Save quantized model weight to {}.".format(qmodel_weight_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))


def load(model_name_or_path, original_model=None, format=SaveLoadFormat.DEFAULT, device="cpu", **kwargs):
    """Load quantized weight-only quantization model.

    1. Load INC weight-only quantized model in local.
        from neural_compressor.torch.quantization import load
        load(model_name_or_path="saved_results", original_model=fp32_model, format="default", device="cpu")

    2. Load HuggingFace weight-only quantized model, including GPTQ models and
       upstreamed INC quantized models in HF model hub.
        from neural_compressor.torch.quantization import load
        load(model_name_or_path=model_name_or_path, format="huggingface", device="cpu")

    Args:
        model_name_or_path (str):  torch checkpoint directory or hugginface model_name_or_path.
            If 'format' is set to 'huggingface', it means the huggingface model_name_or_path.
            If 'format' is set to 'default', it means the 'checkpoint_dir'.
            Parameter should not be None. it coworks with 'original_model' parameter to load INC
            weight-only quantized model in local.
        original_model (torch.nn.module, optional): original model before quantization.
            Needed if 'format' is set to 'default'. Defaults to None.
        format (str, optional): 'defult' for loading INC weight-only quantized model.
            'huggingface' for loading huggingface WOQ causal language model. Defaults to "default".
        kwargs (remaining dictionary of keyword arguments, optional):
            remaining dictionary of keyword arguments for loading huggingface models.
            will be passed to the huggingface model's `__init__` method, such as 'trust_remote_code', 'revision'.

    Returns:
        torch.nn.Module: quantized model
    """
    model_loader = WOQModelLoader(model_name_or_path, original_model, format, device, **kwargs)
    model = model_loader.load_woq_model()
    return model


class WOQModelLoader:
    """WOQ Model Loader."""

    def __init__(self, model_name_or_path, original_model=None, format=SaveLoadFormat.DEFAULT, device="cpu", **kwargs):
        """Init the WOQModelLoader object."""
        self.model_name_or_path = model_name_or_path
        self.original_model = original_model
        self.format = format
        self.device = device
        self.kwargs = kwargs
        self.quantization_config = {}
        self.loaded_state_dict = {}
        self.loaded_state_dict_keys = []
        self._should_save_hpu_format_tensor = False
        self._model_local_dir = None  # local directory where model files are saved

    def load_woq_model(self):
        """Load quantized weight-only quantization model.

        Raises:
            ValueError: `format` in load function can only be 'huggingface' or 'default'.

        Returns:
            torch.nn.Module: quantized model
        """
        if self.format == SaveLoadFormat.HUGGINGFACE:
            assert self.model_name_or_path is not None, "'model_name_or_path' can't be None."

            model = self.load_hf_format_woq_model()
            logger.info("Loading HuggingFace weight-only quantization model successfully.")
        elif self.format == SaveLoadFormat.DEFAULT:
            assert os.path.exists(self.model_name_or_path), f"'{self.model_name_or_path}' path doesn't exist."
            assert (
                self.original_model is not None
            ), "Can't get original model. Please pass `original_model` to load function."

            model = self.load_inc_format_woq_model()
            logger.info("Loading weight-only quantization model successfully.")
        else:
            raise ValueError(f"`format` in load function can only be 'huggingface' or 'default', but get {self.format}")

        return model

    def load_inc_format_woq_model(self):
        """Load WOQ model saved in INC file format."""
        self._model_local_dir = self.model_name_or_path

        qmodel_weight_file_path = os.path.join(
            os.path.abspath(os.path.expanduser(self.model_name_or_path)), WEIGHT_NAME
        )
        # if hpu format tensor can be used directly, then update qmodel_weight_file_path to the hpu format tensor file
        if self._use_hpu_module():
            qmodel_weight_file_path = os.path.join(
                os.path.abspath(os.path.expanduser(self.model_name_or_path)), HPU_WEIGHT_NAME
            )
        assert os.path.exists(qmodel_weight_file_path), (
            "Cannot load model weight from path {}. "
            "Please make sure '{}' file is saved in your '{}' directory ".format(
                qmodel_weight_file_path, WEIGHT_NAME, self.model_name_or_path
            )
        )
        logger.info(f"Find weight file {qmodel_weight_file_path}")

        qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(self.model_name_or_path)), QCONFIG_NAME)
        assert os.path.exists(qconfig_file_path), (
            "Cannot load model quantization config from path {}. "
            "Please make sure '{}' file is saved in your '{}' directory".format(
                qconfig_file_path, QCONFIG_NAME, self.model_name_or_path
            )
        )

        # get loaded state_dict
        self.loaded_state_dict = torch.load(qmodel_weight_file_path)
        self.loaded_state_dict_keys = list(set(self.loaded_state_dict.keys()))

        # get qconfig
        with open(qconfig_file_path, "r") as file:
            self.quantization_config = json.load(file)

        # build weight-only quantization model with WeightOnlyLinear module
        model = self._build_woq_model()

        # load remaining pretrained weight to weight-only quantization model
        is_meta_device = hasattr(self.original_model, "device") and self.original_model.device.type == "meta"
        algo_name = next(iter(self.quantization_config[next(iter(self.quantization_config))].keys()))
        if is_meta_device or algo_name in [AWQ, TEQ]:
            # AWQ and TEQ will update some weight except WOQLinear to handle additional input_scale
            model.load_state_dict(self.loaded_state_dict, assign=True, strict=False)

        # save hpu format tensor to local directory
        if self._should_save_hpu_format_tensor:
            self._save_hpu_format_tensor(model)

        model.eval()
        return model

    def load_hf_format_woq_model(self):
        """Load HuggingFace weight-only quantized model.

        Returns:
            torch.nn.Module: quantized model
        """
        # check required package
        from neural_compressor.torch.utils import is_package_available

        if not is_package_available("transformers"):
            raise ImportError("Loading huggingface model requires transformers: `pip install transformers`")
        if not is_package_available("accelerate"):
            raise ImportError("Loading huggingface model requires accelerate: `pip install accelerate`")

        # get model class and config
        model_class, config = self._get_model_class_and_config()
        self.quantization_config = config.quantization_config if hasattr(config, "quantization_config") else None
        if (
            "backend" in self.quantization_config and "auto_round" in self.quantization_config["backend"]
        ):  # # pragma: no cover
            # load autoround format quantized model
            from auto_round import AutoRoundConfig

            hf_kargs = {}
            pretrain_args = ["trust_remote_code", "_attn_implementation", "device_map", "torch_dtype"]
            for item in pretrain_args:
                arg_value = self.kwargs.get(item, None)
                if arg_value is not None:
                    hf_kargs[item] = arg_value
            model = model_class.from_pretrained(self.model_name_or_path, **hf_kargs)
            return model
        # get loaded state_dict
        self.loaded_state_dict = self._get_loaded_state_dict(config)
        self.loaded_state_dict_keys = list(set(self.loaded_state_dict.keys()))

        # initiate the huggingface model (FP32 empty model)
        self.original_model = self._init_hf_model(model_class, config)

        # build weight-only quantization model with WeightOnlyLinear module
        # and load quantized weight to WeightOnlyLinear modules
        model = self._build_woq_model()

        # clear loaded_state_dict
        self.loaded_state_dict = {}

        # load remaining pretrained weight to weight-only quantization model
        model = self._load_remaining_pretrained_weight(model)

        # save hpu format tensor to local directory
        if self._should_save_hpu_format_tensor:
            self._save_hpu_format_tensor(model)

        model.eval()
        return model

    def _is_hqq_model(self):
        for name, module in self.original_model.named_modules():
            pattern = rf"(\(.*{re.escape(name)}.*{re.escape(type(module).__name__)}.*\))"
            for q_config_key, q_config_value in self.quantization_config.items():
                if re.search(pattern, q_config_key):
                    if isinstance(q_config_value, dict) and [algo for algo in q_config_value.keys()][0] == "hqq":
                        return True

    def _build_hqq_model(self):
        """Replace quantized Linear with HQQLinear."""
        from neural_compressor.torch.algorithms.weight_only.hqq.core import HQQLinear
        from neural_compressor.torch.utils import set_module

        for name, module in self.original_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                loaded_state_dict_keys_set = set(self.loaded_state_dict_keys)
                if name + ".val" not in loaded_state_dict_keys_set:
                    continue
                new_module = HQQLinear(
                    in_features=module.in_features, out_features=module.out_features, bias=module.bias is not None
                )
                self._load_data_to_new_module_hqq(new_module, name)
                set_module(self.original_model, name, new_module)
        woq_model = self.original_model
        return woq_model

    def _load_data_to_new_module_hqq(self, new_module, module_name):
        new_module_state_dict = {}
        for key in self.loaded_state_dict:
            if key.startswith(module_name):
                new_key = key[len(module_name) + 1 :]  # Remove module_name and the following dot
                new_module_state_dict[new_key] = self.loaded_state_dict[key]
                self.loaded_state_dict_keys.remove(key)
        new_module.load_state_dict(new_module_state_dict, strict=False)

    def _build_woq_model(self):
        """Build weight-only quantization model."""
        if self._is_hqq_model():
            return self._build_hqq_model()

        self._update_format_woqlinear_mapping()

        for name, module in self.original_model.named_modules():
            # replace `torch.nn.Linear` to `WeightOnlyLinear` in self.original_model and load its quantized data
            if isinstance(module, torch.nn.Linear):
                # module without qweight means it is not quantized, then skip it
                if (
                    name + ".qweight" not in self.loaded_state_dict_keys
                    and name + ".linear.qweight" not in self.loaded_state_dict_keys
                ):
                    continue

                module_quantization_config, _is_autoround = self._get_module_quantization_config(name, module)
                self._replace_woqlinear_modules(name, module, module_quantization_config, _is_autoround)

        woq_model = self.original_model
        return woq_model

    def _update_format_woqlinear_mapping(self):
        """Update format mapping module to HPUWeightOnlyLinear if tensor is hpu format."""
        if self._use_hpu_module():
            format_woqlinear_mapping.update({self.format: HPUWeightOnlyLinear})

        logger.debug(
            f"Build weight-only quantization model according to format and device mapping. \n"
            f"Format mapping is {format_woqlinear_mapping}. \n"
            f"Device mapping is {device_woqlinear_mapping}."
        )

    def _get_module_quantization_config(self, module_name, module):
        """Gt quantization config of current module.

        1. INC weight-only quantization model, quantization_config will be structured in module level like:
            {(module1_name, module1_type): {"rtn": {"bits": 4, ...}}, ...}
        2. HF weight-only quantization model, quantization_config will be structured in model level like:
            {'bits': 4, ...}
        """
        module_quantization_config = self.quantization_config
        pattern = rf"(\(.*{re.escape(module_name)}.*{re.escape(type(module).__name__)}.*\))"
        _is_autoround = False
        # for loop is used to find quantization config of the target module in INC weight-only quantization model
        for q_config_key, q_config_value in self.quantization_config.items():
            if re.search(pattern, q_config_key):
                # pattern will map (module_name, moduele_type)
                if isinstance(q_config_value, dict) and [algo for algo in q_config_value.keys()][0] == "autoround":
                    _is_autoround = True
                module_quantization_config = [config for config in q_config_value.values()][0]
        return module_quantization_config, _is_autoround

    def _replace_woqlinear_modules(self, name, linear_module, module_quantization_config, _is_autoround):
        """Replace torch.nn.Linear modules with WeightOnlyLinear and load its quantized data."""
        # insert MulLinear module for AWQ/TEQ algorithm
        if name + ".linear.qweight" in self.loaded_state_dict_keys:
            new_module = MulLinear(linear_module)
            set_module(self.original_model, name, new_module)
            name += ".linear"

        # get format mapping module class
        WeightOnlyLinearClass = format_woqlinear_mapping[self.format]

        # update initialization kwargs for woq linear module
        module_kwargs = {}

        # base initialization kwargs
        module_kwargs["in_features"] = linear_module.in_features
        module_kwargs["out_features"] = linear_module.out_features
        module_kwargs["dtype"] = module_quantization_config.get("dtype", "int")
        module_kwargs["bits"] = module_quantization_config.get("bits", 4)
        module_kwargs["group_size"] = module_quantization_config.get("group_size", 32)

        # spceific initialization kwargs
        module_kwargs["g_idx"] = True if name + ".g_idx" in self.loaded_state_dict_keys else False
        module_kwargs["zp"] = True if name + ".qzeros" in self.loaded_state_dict_keys else False
        module_kwargs["use_optimum_format"] = True
        module_kwargs["bias"] = linear_module.bias is not None
        if _is_autoround:
            module_kwargs["scale_dtype"] = convert_dtype_str2torch(
                module_quantization_config.get("scale_dtype", "fp16")
            )

        # initialize the new WeightOnlyLinearClass
        new_module = WeightOnlyLinearClass(**module_kwargs)

        # load quantized data of current module
        self._load_data_to_new_module(new_module, name)

        # update mapped woqlinear module if needed
        new_module = self._update_mapped_woqlinear_modules(name, new_module, module_kwargs)

        set_module(self.original_model, name, new_module)

    def _load_data_to_new_module(self, new_module, module_name):
        new_module_state_dict = {}
        for key in [".qweight", ".scales", ".qzeros", ".bias", ".g_idx"]:
            full_name = module_name + key
            if full_name in self.loaded_state_dict:
                new_module_state_dict[key[1:]] = self.loaded_state_dict.pop(full_name)
                self.loaded_state_dict_keys.remove(full_name)
        new_module.load_state_dict(new_module_state_dict, strict=False)  # bias is not needed.

    def _update_mapped_woqlinear_modules(self, name, format_woqlinear_module, module_kwargs):
        """Checks whether the format mapping module needs to be updated to the device mapping module."""
        # format mapping module class
        OldWeightOnlyLinearClass = format_woqlinear_mapping[self.format]

        # deivice mapping module class
        NewWeightOnlyLinearClass = device_woqlinear_mapping[self.device]

        # if format mapping module doesn't match device mapping module, then replace to device mapping module
        if OldWeightOnlyLinearClass != NewWeightOnlyLinearClass:
            logger.debug(
                f"Replacing {name}'s type from "
                f"'{OldWeightOnlyLinearClass.__name__}' "
                f"to '{NewWeightOnlyLinearClass.__name__}'"
            )

            # initialize the new WeightOnlyLinearClass
            device_mapping_module = NewWeightOnlyLinearClass(**module_kwargs)

            # unpack format mapping module and re-pack to device mapping module
            params_dict = format_woqlinear_module.unpack().to(self.device)
            device_mapping_module.pack(**params_dict)

            # if the new module is HPUWeightOnlyLinear, save hpu format tensor for next loading
            if NewWeightOnlyLinearClass == HPUWeightOnlyLinear and not self._should_save_hpu_format_tensor:
                self._should_save_hpu_format_tensor = True

            return device_mapping_module
        else:
            return format_woqlinear_module

    def _get_model_class_and_config(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
        from transformers.models.auto.auto_factory import _get_model_class

        # Autofactory
        kwargs_orig = copy.deepcopy(self.kwargs)
        trust_remote_code = self.kwargs.pop("trust_remote_code", None)
        kwarg_attn_imp = self.kwargs.pop("attn_implementation", None)

        config = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=trust_remote_code)
        # quantization_config = config.quantization_config

        if kwarg_attn_imp is not None and config._attn_implementation != kwarg_attn_imp:  # pragma: no cover
            config._attn_implementation = kwarg_attn_imp

        has_remote_code = hasattr(config, "auto_map") and AutoModelForCausalLM.__name__ in config.auto_map

        has_local_code = type(config) in AutoModelForCausalLM._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code,
            self.model_name_or_path,
            has_local_code,
            has_remote_code,
        )

        model_class = self.kwargs.get("model_class", None)
        if model_class:
            return model_class, config

        if has_remote_code and trust_remote_code:  # pragma: no cover
            class_ref = config.auto_map[AutoModelForCausalLM.__name__]
            model_class = get_class_from_dynamic_module(class_ref, self.model_name_or_path, **kwargs_orig)
            if os.path.isdir(self.model_name_or_path):
                model_class.register_for_auto_class(AutoModelForCausalLM.__name__)
            else:
                AutoModelForCausalLM.register(config.__class__, model_class, exist_ok=True)
        elif type(config) in AutoModelForCausalLM._model_mapping.keys():
            model_class = _get_model_class(config, AutoModelForCausalLM._model_mapping)
        else:
            logger.info("Couldn't find model class.")
        return model_class, config

    def _get_loaded_state_dict(self, config):
        from transformers.configuration_utils import PretrainedConfig
        from transformers.modeling_utils import get_checkpoint_shard_files, load_state_dict
        from transformers.utils import cached_file, extract_commit_hash, is_safetensors_available

        subfolder = self.kwargs.pop("subfolder", "")
        variant = self.kwargs.pop("variant", None)
        cache_dir = self.kwargs.pop("cache_dir", None)
        force_download = self.kwargs.pop("force_download", False)
        proxies = self.kwargs.pop("proxies", None)
        resume_download = self.kwargs.pop("resume_download", False)
        local_files_only = self.kwargs.pop("local_files_only", False)
        offload_folder = self.kwargs.pop("offload_folder", None)
        offload_state_dict = self.kwargs.pop("offload_state_dict", False)
        use_auth_token = self.kwargs.pop("use_auth_token", None)
        token = self.kwargs.pop("token", None)
        from_pipeline = self.kwargs.pop("_from_pipeline", None)
        from_auto_class = self.kwargs.pop("_from_auto", False)
        revision = self.kwargs.pop("revision", "main")
        commit_hash = self.kwargs.pop("_commit_hash", None)
        use_hpu_safetensors = self.kwargs.pop("use_hpu_safetensors", None)
        use_safetensors = self.kwargs.pop("use_safetensors", None)

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

        if use_hpu_safetensors is None and not is_safetensors_available():
            use_hpu_safetensors = False

        if use_auth_token is not None:  # pragma: no cover
            logger.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. "
                "Please use `token` instead."
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        user_agent = {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": from_auto_class,
        }
        if from_pipeline is not None:  # pragma: no cover
            user_agent["using_pipeline"] = from_pipeline

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):  # pragma: no cover
                # We make a call to the config file first (which may be absent)
                # to get the commit hash as soon as possible.
                resolved_config_file = cached_file(
                    self.model_name_or_path,
                    "config.json",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        is_sharded = False
        sharded_metadata = None

        self.model_name_or_path = str(self.model_name_or_path)

        # get resolved weight archive file
        kwargs = {
            "use_safetensors": use_safetensors,
            "variant": variant,
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "token": token,
            "user_agent": user_agent,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_gated_repo": False,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
        }
        resolved_archive_file, is_sharded = self._get_resolved_archive_file(**kwargs)

        self._model_local_dir = os.path.abspath(os.path.expanduser(os.path.dirname(resolved_archive_file)))
        # if hpu format tensor can be used directly, then update resolved_archive_file to the hpu format tensor file
        if self._use_hpu_module():
            resolved_archive_file = os.path.join(self._model_local_dir, HPU_SAFE_WEIGHTS_NAME)

        logger.info(f"Find weight file {resolved_archive_file}")

        if is_sharded:  # pragma: no cover
            # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
            resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                self.model_name_or_path,
                resolved_archive_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )
            self.kwargs["sharded_metadata"] = sharded_metadata

        # Time to load the checkpoint
        state_dict = None
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]
        for shard_file in resolved_archive_file:
            if state_dict is None:
                state_dict = load_state_dict(shard_file)
            else:
                state_dict.update(load_state_dict(shard_file))

        # set kwargs for next functions to use
        self.kwargs["is_sharded"] = is_sharded
        self.kwargs["offload_folder"] = offload_folder
        self.kwargs["offload_state_dict"] = offload_state_dict
        self.kwargs["resolved_archive_file"] = resolved_archive_file

        return state_dict

    def _get_resolved_archive_file(self, **kwargs):
        """Get weight archive file of model."""
        from transformers.modeling_utils import _add_variant
        from transformers.utils import (
            SAFE_WEIGHTS_INDEX_NAME,
            SAFE_WEIGHTS_NAME,
            WEIGHTS_INDEX_NAME,
            WEIGHTS_NAME,
            cached_file,
            download_url,
            has_file,
            is_remote_url,
        )

        use_safetensors = kwargs.pop("use_safetensors")
        variant = kwargs.pop("variant")
        subfolder = kwargs.get("subfolder")

        resolved_archive_file = None
        is_sharded = False
        is_local = os.path.isdir(self.model_name_or_path)
        if is_local:  # pragma: no cover
            # self.model_name_or_path is a local directory
            if os.path.isfile(
                os.path.join(
                    self.model_name_or_path,
                    subfolder,
                    _add_variant(WEIGHTS_NAME, variant),
                )
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    self.model_name_or_path,
                    subfolder,
                    _add_variant(WEIGHTS_NAME, variant),
                )
            elif os.path.isfile(
                os.path.join(
                    self.model_name_or_path,
                    subfolder,
                    _add_variant(WEIGHTS_INDEX_NAME, variant),
                )
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    self.model_name_or_path,
                    subfolder,
                    _add_variant(WEIGHTS_INDEX_NAME, variant),
                )
                is_sharded = True
            elif os.path.isfile(
                os.path.join(
                    self.model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_NAME, variant),
                )
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    self.model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_NAME, variant),
                )
            elif os.path.isfile(
                os.path.join(
                    self.model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                )
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    self.model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                )
                is_sharded = True
        elif os.path.isfile(os.path.join(subfolder, self.model_name_or_path)):  # pragma: no cover
            archive_file = self.model_name_or_path
            is_local = True
        elif is_remote_url(self.model_name_or_path):  # pragma: no cover
            # self.model_name_or_path is a url
            filename = self.model_name_or_path
            resolved_archive_file = download_url(self.model_name_or_path)
        else:
            # self.model_name_or_path is a model_id in huggingface
            if use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)
            try:
                # Load from URL or cache if already cached
                cached_file_kwargs = kwargs
                resolved_archive_file = cached_file(self.model_name_or_path, filename, **cached_file_kwargs)

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(
                    SAFE_WEIGHTS_NAME, variant
                ):  # pragma: no cover
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        self.model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        raise EnvironmentError(
                            f"{self.model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or "
                            f"{_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                            "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                            "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                        )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(self.model_name_or_path, filename, **cached_file_kwargs)
                if resolved_archive_file is None and filename == _add_variant(
                    WEIGHTS_NAME, variant
                ):  # pragma: no cover
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        self.model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True

                if resolved_archive_file is None:  # pragma: no cover
                    # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                    # message.
                    has_file_kwargs = {
                        "revision": cached_file_kwargs.get("revision"),
                        "proxies": cached_file_kwargs.get("proxies"),
                        "token": cached_file_kwargs.get("token"),
                    }
                    if variant is not None and has_file(self.model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{self.model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"{self.model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)}."
                        )
            except EnvironmentError:  # pragma: no cover
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception as e:  # pragma: no cover
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{self.model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{self.model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}."
                ) from e

        if is_local:
            resolved_archive_file = archive_file

        return resolved_archive_file, is_sharded

    def _init_hf_model(self, model_class, config):
        from accelerate.big_modeling import init_empty_weights
        from transformers.modeling_utils import no_init_weights
        from transformers.utils import ContextManagers

        _fast_init = self.kwargs.pop("_fast_init", True)
        torch_dtype = self.kwargs.pop("torch_dtype", "auto")
        is_sharded = self.kwargs.pop("is_sharded", False)
        sharded_metadata = self.kwargs.pop("sharded_metadata", None)
        offload_folder = self.kwargs.pop("offload_folder", None)
        offload_state_dict = self.kwargs.pop("offload_state_dict", False)
        resolved_archive_file = self.kwargs.pop("resolved_archive_file", None)

        # set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict,
        #    by checking its first weights entry that is of a floating type
        #    - we assume all floating dtype weights are of the same dtype
        dtype_orig = None
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":  # pragma: no cover
                    if (
                        hasattr(config, "torch_dtype")
                        and config.torch_dtype is not None
                        and config.torch_dtype != "auto"
                    ):
                        torch_dtype = config.torch_dtype
                    else:
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        else:
                            torch_dtype = torch.float32
                else:  # pragma: no cover
                    assert False, f'`torch_dtype` can be either `torch.dtype` or `"auto"`, but received {torch_dtype}'

            dtype_orig = model_class._set_default_torch_dtype(torch_dtype)

        init_contexts = [no_init_weights(_enable=_fast_init)]
        init_contexts.append(init_empty_weights())

        with ContextManagers(init_contexts):
            model = model_class(config, **self.kwargs)

        # set kwargs for next functions to use
        self.kwargs["resolved_archive_file"] = resolved_archive_file
        self.kwargs["torch_dtype"] = torch_dtype
        self.kwargs["dtype_orig"] = dtype_orig
        self.kwargs["offload_folder"] = offload_folder
        self.kwargs["offload_state_dict"] = offload_state_dict

        return model

    def _load_remaining_pretrained_weight(self, model):
        """Load remaining pretrained weight.

        In _build_woq_model function, linear will be replaced to weight-only quantization linear
        and its quantized weight will be loaded. Remaining pretrained weight (like layernorm weight,
        embedding weight or other unquantized linear weight) will be loaded in this function.
        """
        from transformers.modeling_utils import _load_state_dict_into_meta_model, load_state_dict

        resolved_archive_file = self.kwargs.pop("resolved_archive_file", None)
        torch_dtype = self.kwargs.pop("torch_dtype", torch.float32)
        dtype_orig = self.kwargs.pop("dtype_orig", None)
        offload_folder = self.kwargs.pop("offload_folder", None)
        offload_state_dict = self.kwargs.pop("offload_state_dict", False)

        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]
        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)

            params_dict = {
                "model": model,
                "state_dict": state_dict,
                "start_prefix": "",
                "expected_keys": self.loaded_state_dict_keys,
                "device_map": {"": self.device},
                "offload_folder": offload_folder,
                "state_dict_folder": tempfile.mkdtemp() if offload_state_dict else None,
                "state_dict_index": {} if offload_state_dict else None,
                "dtype": torch_dtype,
                "keep_in_fp32_modules": [],
            }

            import transformers

            if transformers.__version__ < "4.45.0":
                params_dict["loaded_state_dict_keys"] = self.loaded_state_dict_keys

            _load_state_dict_into_meta_model(**params_dict)

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model

    def _save_hpu_format_tensor(self, model):  # pragma: no cover
        from safetensors.torch import save_file

        if not os.path.exists(self._model_local_dir):
            logger.warning(f"{self._model_local_dir} doesn't exist, can't save hpu format safetensors")

        if self.format == SaveLoadFormat.HUGGINGFACE:
            filename = os.path.join(self._model_local_dir, HPU_SAFE_WEIGHTS_NAME)
            save_file({k: v for k, v in model.state_dict().items()}, filename=filename, metadata={"format": "pt"})
            logger.debug(f"Save hpu format tensor to {filename}")
        elif self.format == SaveLoadFormat.DEFAULT:
            qmodel_weight_file_path = os.path.join(self._model_local_dir, HPU_WEIGHT_NAME)
            torch.save({k: v for k, v in model.state_dict().items()}, qmodel_weight_file_path)
            logger.debug(f"Save hpu format tensor to {qmodel_weight_file_path}")

    def _use_hpu_module(self):  # pragma: no cover
        """Check whether hpu weight-only quantization linear module can be used.

        return True when:
        1. device is 'hpu'
        2. model has hpu format tensor in local cache directory:
            - has 'hpu_model.safetensors' file with huggingface format
            - or has 'quantized_hpu_weight.pt' file with default format
           or 'format' flag in config.json file is 'habana' (flag name needs discussion, not implemented yet)
        """
        if self.device == "hpu" and os.path.exists(self._model_local_dir):
            if self.format == SaveLoadFormat.HUGGINGFACE:
                if os.path.exists(os.path.join(self._model_local_dir, HPU_SAFE_WEIGHTS_NAME)):
                    # update resolved_archive_file
                    self.kwargs["resolved_archive_file"] = os.path.join(self._model_local_dir, HPU_SAFE_WEIGHTS_NAME)
                    return True
            elif self.format == SaveLoadFormat.DEFAULT:
                if os.path.exists(os.path.join(self._model_local_dir, HPU_WEIGHT_NAME)):
                    return True
        return False
