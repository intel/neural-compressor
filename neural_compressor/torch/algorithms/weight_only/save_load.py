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

import torch

from neural_compressor.common.utils import load_config_mapping, save_config_mapping
from neural_compressor.torch.utils import QCONFIG_NAME, WEIGHT_NAME, LoadFormat, logger


def save(model, output_dir="./saved_results"):
    """Save the quantized model and config to the output path.

    Args:
        model (torch.nn.module): raw fp32 model or prepared model.
        output_dir (str, optional): output path to save.
    """
    os.makedirs(output_dir, exist_ok=True)
    qmodel_weight_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), WEIGHT_NAME)
    qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), QCONFIG_NAME)
    # saving process
    save_config_mapping(model.qconfig, qconfig_file_path)

    if hasattr(model, "gptq_config") and model.gptq_config:
        gptq_config_path = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), "gptq_config.json")
        with open(gptq_config_path, "w") as f:
            json.dump(model.gptq_config, f, indent=4)

    # MethodType 'save' not in state_dict
    del model.save
    torch.save(model.state_dict(), qmodel_weight_file_path)

    logger.info("Save quantized model to {}.".format(qmodel_weight_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))


def load(model_name_or_path, original_model=None, format=LoadFormat.DEFAULT, device="cpu", **kwargs):
    """Load quantized weight-only quantization model.

    1. Load INC weight-only quantized model in local.
    2. Load HuggingFace weight-only quantized model,
       including GPTQ/AWQ models and upstreamed INC quantized models in HF model hub.

    Args:
        model_name_or_path (str):  torch checkpoint directory or hugginface model_name_or_path.
            If 'format' is set to 'huggingface', it means the huggingface model_name_or_path.
            If 'format' is set to 'default', it means the 'checkpoint_dir'.
            Parameter should not be None. it coworks with 'original_model' parameter to load INC
            weight-only quantized model in local.
        original_model (torch.nn.module, optional): original model before quantization.
            Needed if 'format' is set to 'default' and not TorchScript model.Defaults to None.
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

    def __init__(self, model_name_or_path, original_model=None, format=LoadFormat.DEFAULT, device="cpu", **kwargs):
        """Init the WOQModelLoader object."""
        # TODO: When loading WOQ model, use different WeightOnlyLinear module according to device.
        self.model_name_or_path = model_name_or_path
        self.original_model = original_model
        self.format = format
        self.device = device
        self.kwargs = kwargs
        self.quantization_config = {}
        self.loaded_state_dict_keys = {}

    def load_woq_model(self):
        """Load quantized weight-only quantization model.

        Raises:
            ValueError: `format` in load function can only be 'huggingface' or 'default'.

        Returns:
            torch.nn.Module: quantized model
        """
        if self.format == LoadFormat.HUGGINGFACE:
            model = self.load_hf_format_woq_model()
            logger.info("Loading HuggingFace weight-only quantization model successfully.")
        elif self.format == LoadFormat.DEFAULT:
            qmodel_weight_file_path = os.path.join(
                os.path.abspath(os.path.expanduser(self.model_name_or_path)), WEIGHT_NAME
            )
            assert os.path.exists(qmodel_weight_file_path), (
                "Cannot load model weight from path {}. "
                "Please make sure '{}' file is saved in your '{}' directory ".format(
                    qmodel_weight_file_path, WEIGHT_NAME, self.model_name_or_path
                )
            )

            qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(self.model_name_or_path)), QCONFIG_NAME)
            assert os.path.exists(qconfig_file_path), (
                "Cannot load model quantization config from path {}. "
                "Please make sure '{}' file is saved in your '{}' directory".format(
                    qconfig_file_path, QCONFIG_NAME, self.model_name_or_path
                )
            )

            assert (
                self.original_model is not None
            ), "Can't get original model. Please pass `original_model` to load function."

            model = self.load_inc_format_woq_model(qmodel_weight_file_path, qconfig_file_path)
            logger.info("Loading weight-only quantization model successfully.")
        else:
            raise ValueError(f"`format` in load function can only be 'huggingface' or 'default', but get {self.format}")

        return model

    def load_inc_format_woq_model(self, qmodel_weight_file_path, qconfig_file_path):
        """Load INC weight-only quantized model in local.

        Args:
            qmodel_weight_file_path (str): path to the quantized model.
            qconfig_file_path (str): path to the quant config.

        Returns:
            torch.nn.Module: quantized model
        """
        qweights = torch.load(qmodel_weight_file_path)
        self.loaded_state_dict_keys = qweights.keys()

        with open(qconfig_file_path, "r") as file:
            self.quantization_config = json.load(file)
        model = self._build_woq_model()
        model.load_state_dict(qweights, assign=True)
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

        # get model_class and config
        model_class, config = self._get_model_class_and_config()
        self.quantization_config = config.quantization_config

        # get loaded_state_dict_keys
        self.loaded_state_dict_keys = self._get_loaded_state_dict_keys(config)

        # initiate the huggingface model
        self.original_model = self._init_hf_model(model_class, config)

        # build weight-only quantization model with WeightOnlyLinear module
        model = self._build_woq_model()

        # load quantized weight to woq model
        model = self._load_pretrained_weight(model, model_class)

        return model

    def _is_hqq_model(self):
        for name, module in self.original_model.named_modules():
            pattern = rf"(\(.*{re.escape(name)}.*{re.escape(type(module).__name__)}.*\))"
            for q_config_key, q_config_value in self.quantization_config.items():
                if re.search(pattern, q_config_key):
                    if isinstance(q_config_value, dict) and [algo for algo in q_config_value.keys()][0] == "hqq":
                        return True

    def _build_woq_model(self):
        """Build weight-only quantization model."""
        if self._is_hqq_model():
            return self._build_hqq_model()

        from neural_compressor.torch.utils import set_module

        from .modules import MulLinear

        for name, module in self.original_model.named_modules():
            _is_autoround = False
            # get quantization config of module
            module_quantization_config = self.quantization_config
            # pattern will map (module_name, moduele_type)
            pattern = rf"(\(.*{re.escape(name)}.*{re.escape(type(module).__name__)}.*\))"
            for q_config_key, q_config_value in self.quantization_config.items():
                if re.search(pattern, q_config_key):
                    if isinstance(q_config_value, dict) and [algo for algo in q_config_value.keys()][0] == "autoround":
                        _is_autoround = True
                    module_quantization_config = [config for config in q_config_value.values()][0]

            if isinstance(module, torch.nn.Linear):
                # module without qweight means it is not quantized, then skip it
                loaded_state_dict_keys_set = set(self.loaded_state_dict_keys)
                if (
                    name + ".qweight" not in loaded_state_dict_keys_set
                    and name + ".linear.qweight" not in loaded_state_dict_keys_set
                ):
                    continue

                # insert MulLinear module
                if name + ".linear.qweight" in loaded_state_dict_keys_set:
                    new_module = MulLinear(module)
                    set_module(self.original_model, name, new_module)
                    name += ".linear"

                # replace `torch.nn.Linear` with `WeightOnlyLinear`
                zp = True if name + ".qzeros" in loaded_state_dict_keys_set else False
                g_idx = True if name + ".g_idx" in loaded_state_dict_keys_set else False

                kwargs = {}
                if _is_autoround:
                    from auto_round.export.export_to_itrex.model_wrapper import (
                        WeightOnlyLinear as AutoRoundWeightOnlyLinear,
                    )

                    from .utility import convert_dtype_str2torch

                    WeightOnlyLinearClass = AutoRoundWeightOnlyLinear
                    kwargs["groupsize"] = module_quantization_config.get("group_size", 32)
                    kwargs["scale_dtype"] = convert_dtype_str2torch(
                        module_quantization_config.get("scale_dtype", "fp16")
                    )
                else:
                    from .modules import WeightOnlyLinear as INCWeightOnlyLinear

                    WeightOnlyLinearClass = INCWeightOnlyLinear
                    kwargs["group_size"] = module_quantization_config.get("group_size", 32)
                    kwargs["g_idx"] = g_idx

                new_module = WeightOnlyLinearClass(
                    module.in_features,
                    module.out_features,
                    dtype=module_quantization_config.get("dtype", "int"),
                    bits=module_quantization_config.get("bits", 4),
                    zp=zp,
                    bias=module.bias is not None,
                    use_optimum_format=True,
                    **kwargs,
                )
                set_module(self.original_model, name, new_module)
        woq_model = self.original_model
        return woq_model

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
                set_module(self.original_model, name, new_module)
        woq_model = self.original_model
        return woq_model

    def _get_model_class_and_config(self):
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
        from transformers.models.auto.auto_factory import _get_model_class

        # Autofactory
        kwargs_orig = copy.deepcopy(self.kwargs)
        trust_remote_code = self.kwargs.pop("trust_remote_code", None)
        kwarg_attn_imp = self.kwargs.pop("attn_implementation", None)

        config = AutoConfig.from_pretrained(self.model_name_or_path)
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

        if has_remote_code and trust_remote_code:  # pragma: no cover
            class_ref = config.auto_map[AutoModelForCausalLM.__name__]
            model_class = get_class_from_dynamic_module(class_ref, self.model_name_or_path, **kwargs_orig)
            if os.path.isdir(self.model_name_or_path):
                model_class.register_for_auto_class(AutoModelForCausalLM.__name__)
            else:
                AutoModelForCausalLM.register(config.__class__, model_class, exist_ok=True)
        elif type(config) in AutoModelForCausalLM._model_mapping.keys():
            model_class = _get_model_class(config, AutoModelForCausalLM._model_mapping)

        return model_class, config

    def _get_loaded_state_dict_keys(self, config):
        from transformers.configuration_utils import PretrainedConfig
        from transformers.modeling_utils import _add_variant, get_checkpoint_shard_files, load_state_dict
        from transformers.utils import (
            SAFE_WEIGHTS_INDEX_NAME,
            SAFE_WEIGHTS_NAME,
            WEIGHTS_INDEX_NAME,
            WEIGHTS_NAME,
            cached_file,
            download_url,
            extract_commit_hash,
            has_file,
            is_remote_url,
            is_safetensors_available,
        )

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
        use_safetensors = self.kwargs.pop("use_safetensors", None)

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False

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

        if self.model_name_or_path is not None:  # pragma: no cover
            self.model_name_or_path = str(self.model_name_or_path)
            is_local = os.path.isdir(self.model_name_or_path)
            if is_local:
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
            elif os.path.isfile(os.path.join(subfolder, self.model_name_or_path)):
                archive_file = self.model_name_or_path
                is_local = True
            elif is_remote_url(self.model_name_or_path):
                filename = self.model_name_or_path
                resolved_archive_file = download_url(self.model_name_or_path)
            else:
                if use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(WEIGHTS_NAME, variant)
                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = {
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
                    resolved_archive_file = cached_file(self.model_name_or_path, filename, **cached_file_kwargs)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
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
                    if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            self.model_name_or_path,
                            _add_variant(WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True

                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        has_file_kwargs = {
                            "revision": revision,
                            "proxies": proxies,
                            "token": token,
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
                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{self.model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{self.model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        else:  # pragma: no cover
            resolved_archive_file = None

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

        if is_sharded:  # pragma: no cover
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            # Time to load the checkpoint
            state_dict = load_state_dict(resolved_archive_file)
            loaded_state_dict_keys = list(state_dict.keys())

        # set kwargs for next functions to use
        self.kwargs["is_sharded"] = is_sharded
        self.kwargs["offload_folder"] = offload_folder
        self.kwargs["offload_state_dict"] = offload_state_dict
        self.kwargs["resolved_archive_file"] = resolved_archive_file

        return loaded_state_dict_keys

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
                if torch_dtype == "auto":
                    if (
                        hasattr(config, "torch_dtype")
                        and config.torch_dtype is not None
                        and config.torch_dtype != "auto"
                    ):
                        torch_dtype = config.torch_dtype
                    else:  # pragma: no cover
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
        self.kwargs["sharded_metadata"] = sharded_metadata
        self.kwargs["torch_dtype"] = torch_dtype
        self.kwargs["dtype_orig"] = dtype_orig
        self.kwargs["_fast_init"] = _fast_init
        self.kwargs["offload_folder"] = offload_folder
        self.kwargs["offload_state_dict"] = offload_state_dict

        return model

    def _load_pretrained_weight(self, model, model_class):
        resolved_archive_file = self.kwargs.pop("resolved_archive_file", None)
        sharded_metadata = self.kwargs.pop("sharded_metadata", None)
        torch_dtype = self.kwargs.pop("torch_dtype", torch.float32)
        dtype_orig = self.kwargs.pop("dtype_orig", None)
        _fast_init = self.kwargs.pop("_fast_init", True)
        offload_folder = self.kwargs.pop("offload_folder", None)
        offload_state_dict = self.kwargs.pop("offload_state_dict", False)

        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        (
            model,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = model_class._load_pretrained_model(
            model,
            None,
            self.loaded_state_dict_keys,
            resolved_archive_file,
            self.model_name_or_path,
            sharded_metadata=sharded_metadata,
            _fast_init=_fast_init,
            low_cpu_mem_usage=True,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            dtype=torch_dtype,
            keep_in_fp32_modules=[],
        )

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model
