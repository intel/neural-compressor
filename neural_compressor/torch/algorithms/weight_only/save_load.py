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

# pylint:disable=import-error

import json
import os

import torch

from neural_compressor.common.utils import load_config_mapping, save_config_mapping
from neural_compressor.torch.utils import QCONFIG_NAME, WEIGHT_NAME, logger


def save(model, output_dir="./saved_results"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
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

    logger.info("Save quantized model weight to {}.".format(qmodel_weight_file_path))
    logger.info("Save configuration of quantized model to {}.".format(qconfig_file_path))


def load(model_name_or_path, model=None, format="default", *hf_model_args, **hf_model_kwargs):
    if format == "huggingface":
        model = _load_hf_woq_model(model_name_or_path, *hf_model_args, **hf_model_kwargs)
        logger.info("Quantized huggingface model loading successful.")
        return model
    elif format == "default":
        qmodel_weight_file_path = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)), WEIGHT_NAME)
        assert os.path.exists(qmodel_weight_file_path), "Cannot load model weight from path {}".format(
            qmodel_weight_file_path
        )

        qconfig_file_path = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)), QCONFIG_NAME)
        assert os.path.exists(qconfig_file_path), "Cannot load model quantization config from path {}".format(
            qconfig_file_path
        )

        assert model is not None, "Can't get origin model. Please pass `model` to load function."

        model = _load_inc_woq_model(qmodel_weight_file_path, qconfig_file_path, model)
        logger.info("Quantized model loading successful.")
        return model
    else:
        raise ValueError("`format` in load function can only be 'huggingface' or 'default', but get {}".format(format))


def _build_woq_model(model, quantization_config, loaded_state_dict_keys):
    """Build weight-only quantization model."""
    from neural_compressor.torch.utils import set_module

    from .modules import MulLinear, WeightOnlyLinear

    for name, module in model.named_modules():
        # get quantization config of module
        module_name_type = str((name, type(module).__name__))
        module_quantization_config = quantization_config
        if module_name_type in quantization_config:
            module_quantization_config = quantization_config[module_name_type]

        if isinstance(module, torch.nn.Linear):
            # module without qweight means it is not quantized, then skip it
            loaded_state_dict_keys_set = set(loaded_state_dict_keys)
            if (
                name + ".qweight" not in loaded_state_dict_keys_set
                and name + ".linear.qweight" not in loaded_state_dict_keys_set
            ):
                continue

            # insert MulLinear module
            if name + ".linear.qweight" in loaded_state_dict_keys_set:
                new_module = MulLinear(module)
                set_module(model, name, new_module)
                name += ".linear"

            # replace `torch.nn.Linear` with `WeightOnlyLinear`
            zp = True if name + ".qzeros" in loaded_state_dict_keys else False
            g_idx = True if name + ".g_idx" in loaded_state_dict_keys else False
            new_module = WeightOnlyLinear(
                module.in_features,
                module.out_features,
                bits=module_quantization_config.get("bits", 4),
                group_size=module_quantization_config.get("group_size", 32),
                dtype="int",
                zp=zp,
                bias=module.bias is not None,
                g_idx=g_idx,
                use_optimum_format=True,
            )
            set_module(model, name, new_module)

    return model


def _load_inc_woq_model(qmodel_weight_file_path, qconfig_file_path, origin_model):
    qweights = torch.load(qmodel_weight_file_path)

    quantization_config = {}
    with open(qconfig_file_path, "r") as file:
        quantization_config = json.load(file)

    model = _build_woq_model(origin_model, quantization_config, qweights.keys())
    model.load_state_dict(qweights, assign=True)
    model.eval()
    return model


def _load_hf_woq_model(pretrained_model_name_or_path, *model_args, **kwargs):
    # check required package
    try:
        import transformers
    except ImportError:
        logger.error("`transformers` package is required for loading hugginface weight-only quantization model.")

    try:
        import accelerate
    except ImportError:
        logger.error("`accelerate` package is required for loading hugginface weight-only quantization model.")

    # below codes are refer to load_low_bit function in
    # https://github.com/intel/intel-extension-for-transformers/blob/v1.4.2/intel_extension_for_transformers/transformers/modeling/modeling_auto.py#L1464
    import copy

    from accelerate.big_modeling import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.configuration_utils import PretrainedConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
    from transformers.modeling_utils import _add_variant, get_checkpoint_shard_files, load_state_dict, no_init_weights
    from transformers.models.auto.auto_factory import _get_model_class
    from transformers.utils import (
        SAFE_WEIGHTS_INDEX_NAME,
        SAFE_WEIGHTS_NAME,
        WEIGHTS_INDEX_NAME,
        WEIGHTS_NAME,
        ContextManagers,
        cached_file,
        download_url,
        extract_commit_hash,
        has_file,
        is_remote_url,
        is_safetensors_available,
    )

    # Autofactory
    kwargs_orig = copy.deepcopy(kwargs)
    trust_remote_code = kwargs.pop("trust_remote_code", None)
    subfolder = kwargs.pop("subfolder", "")
    variant = kwargs.pop("variant", None)
    offload_folder = kwargs.pop("offload_folder", None)
    offload_state_dict = kwargs.pop("offload_state_dict", False)
    torch_dtype = kwargs.pop("torch_dtype", "auto")
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    resume_download = kwargs.pop("resume_download", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    token = kwargs.pop("token", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    revision = kwargs.pop("revision", "main")
    commit_hash = kwargs.pop("_commit_hash", None)
    _fast_init = kwargs.pop("_fast_init", True)
    use_safetensors = kwargs.pop("use_safetensors", None)
    kwarg_attn_imp = kwargs.pop("attn_implementation", None)

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    quantization_config = config.quantization_config

    if use_safetensors is None and not is_safetensors_available():
        use_safetensors = False

    if use_auth_token is not None:
        logger.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. "
            "Please use `token` instead."
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    user_agent = {
        "file_type": "model",
        "framework": "pytorch",
        "from_auto_class": from_auto_class,
    }
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if kwarg_attn_imp is not None and config._attn_implementation != kwarg_attn_imp:
        config._attn_implementation = kwarg_attn_imp

    if commit_hash is None:
        if not isinstance(config, PretrainedConfig):
            # We make a call to the config file first (which may be absent)
            # to get the commit hash as soon as possible.
            resolved_config_file = cached_file(
                pretrained_model_name_or_path,
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

    has_remote_code = hasattr(config, "auto_map") and AutoModelForCausalLM.__name__ in config.auto_map

    has_local_code = type(config) in AutoModelForCausalLM._model_mapping.keys()
    trust_remote_code = resolve_trust_remote_code(
        trust_remote_code,
        pretrained_model_name_or_path,
        has_local_code,
        has_remote_code,
    )

    if has_remote_code and trust_remote_code:
        class_ref = config.auto_map[AutoModelForCausalLM.__name__]
        model_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs_orig)
        if os.path.isdir(pretrained_model_name_or_path):
            model_class.register_for_auto_class(AutoModelForCausalLM.__name__)
        else:
            AutoModelForCausalLM.register(config.__class__, model_class, exist_ok=True)
    elif type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_class = _get_model_class(config, AutoModelForCausalLM._model_mapping)

    init_contexts = [no_init_weights(_enable=_fast_init)]
    init_contexts.append(init_empty_weights())

    with ContextManagers(init_contexts):
        model = model_class(config, *model_args, **kwargs)

    is_sharded = False
    sharded_metadata = None

    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            if os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(WEIGHTS_NAME, variant),
                )
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(WEIGHTS_NAME, variant),
                )
            elif os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(WEIGHTS_INDEX_NAME, variant),
                )
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(WEIGHTS_INDEX_NAME, variant),
                )
                is_sharded = True
            elif os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_NAME, variant),
                )
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_NAME, variant),
                )
            elif os.path.isfile(
                os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                )
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path,
                    subfolder,
                    _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                )
                is_sharded = True
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            filename = pretrained_model_name_or_path
            resolved_archive_file = download_url(pretrained_model_name_or_path)
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
                resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or "
                            f"{_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                            "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                            "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                        )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
                        pretrained_model_name_or_path,
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
                    if variant is not None and has_file(pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)}."
                        )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)}."
                ) from e

        if is_local:
            logger.info(f"loading weights file {archive_file}")
            resolved_archive_file = archive_file
        else:
            logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
    else:
        resolved_archive_file = None

    if is_sharded:
        # rsolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
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

    # set dtype to instantiate the model under:
    # 1. If torch_dtype is not None, we use that dtype
    # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict,
    #    by checking its first weights entry that is of a floating type
    #    - we assume all floating dtype weights are of the same dtype
    dtype_orig = None
    if torch_dtype is not None:
        if isinstance(torch_dtype, str):
            if torch_dtype == "auto":
                if hasattr(config, "torch_dtype") and config.torch_dtype is not None and config.torch_dtype != "auto":
                    torch_dtype = config.torch_dtype
                else:
                    if is_sharded and "dtype" in sharded_metadata:
                        torch_dtype = sharded_metadata["dtype"]
                    else:
                        torch_dtype = torch.float32
            else:
                assert False, f'`torch_dtype` can be either `torch.dtype` or `"auto"`, but received {torch_dtype}'

        dtype_orig = model_class._set_default_torch_dtype(torch_dtype)

    if is_sharded:
        loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
    else:
        # Time to load the checkpoint
        state_dict = load_state_dict(resolved_archive_file)
        loaded_state_dict_keys = list(state_dict.keys())

    model = _build_woq_model(model, quantization_config, loaded_state_dict_keys)

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
        loaded_state_dict_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
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
