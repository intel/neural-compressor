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
"""Huggingface Loader: provides access to Huggingface pretrained models."""

import copy
import os
import sys

import torch
import transformers
from transformers import AutoConfig

from neural_compressor.utils import logger

WEIGHTS_NAME = "pytorch_model.bin"


class OptimizedModel:
    """The class provides a method from_pretrained to access Huggingface models."""

    def __init__(self, *args, **kwargs):  # pragma: no cover
        """Init method (Not used)."""
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated using the"
            f"`{self.__class__.__name__}.from_pretrained(model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> torch.nn.Module:
        """Instantiate a quantized pytorch model from a given Intel Neural Compressor (INC) configuration file.

        Args:
            model_name_or_path (:obj:`str`):
                Repository name in the Hugging Face Hub or path to a local directory hosting the model.
            cache_dir (:obj:`str`, `optional`):
                Path to a directory in which a downloaded configuration should be cached if the standard cache should
                not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            revision(:obj:`str`, `optional`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.

        Returns:
            q_model: Quantized model.
        """
        from neural_compressor.utils.pytorch import load

        config = kwargs.pop("config", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)

        if config is None:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                revision=revision,
                **kwargs,
            )

        model_class = eval(f"transformers.{config.architectures[0]}")
        if config.torch_dtype is not torch.int8:
            model = model_class.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                revision=revision,
                **kwargs,
            )
            return model
        else:
            logger.info("the quantization optimized model is loading.")
            keys_to_ignore_on_load_unexpected = copy.deepcopy(
                getattr(model_class, "_keys_to_ignore_on_load_unexpected", None)
            )
            keys_to_ignore_on_load_missing = copy.deepcopy(
                getattr(model_class, "_keys_to_ignore_on_load_missing", None)
            )

            # Avoid unnecessary warnings resulting from quantized model initialization
            quantized_keys_to_ignore_on_load = [
                r"zero_point",
                r"scale",
                r"packed_params",
                r"constant",
                r"module",
                r"best_configure",
            ]
            if keys_to_ignore_on_load_unexpected is None:
                model_class._keys_to_ignore_on_load_unexpected = quantized_keys_to_ignore_on_load
            else:
                model_class._keys_to_ignore_on_load_unexpected.extend(quantized_keys_to_ignore_on_load)
            missing_keys_to_ignore_on_load = [r"weight", r"bias"]
            if keys_to_ignore_on_load_missing is None:
                model_class._keys_to_ignore_on_load_missing = missing_keys_to_ignore_on_load
            else:  # pragma: no cover
                model_class._keys_to_ignore_on_load_missing.extend(missing_keys_to_ignore_on_load)

            if not os.path.isdir(model_name_or_path) and not os.path.isfile(model_name_or_path):  # pragma: no cover
                from transformers.utils import cached_file

                try:
                    # Load from URL or cache if already cached
                    resolved_weights_file = cached_file(
                        model_name_or_path,
                        filename=WEIGHTS_NAME,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        use_auth_token=use_auth_token,
                    )
                except EnvironmentError as err:  # pragma: no cover
                    logger.error(err)
                    msg = (
                        f"Can't load weights for '{model_name_or_path}'. Make sure that:\n\n"
                        f"- '{model_name_or_path}' is a correct model identifier "
                        f"listed on 'https://huggingface.co/models'\n  (make sure "
                        f"'{model_name_or_path}' is not a path to a local directory with "
                        f"something else, in that case)\n\n- or '{model_name_or_path}' is "
                        f"the correct path to a directory containing a file "
                        f"named one of {WEIGHTS_NAME}\n\n"
                    )
                    if revision is not None:
                        msg += (
                            f"- or '{revision}' is a valid git identifier "
                            f"(branch name, a tag name, or a commit id) that "
                            f"exists for this model name as listed on its model "
                            f"page on 'https://huggingface.co/models'\n\n"
                        )
                    raise EnvironmentError(msg)
            else:
                resolved_weights_file = os.path.join(model_name_or_path, WEIGHTS_NAME)
            state_dict = torch.load(resolved_weights_file, {})
            model = model_class.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                revision=revision,
                state_dict=state_dict,
                **kwargs,
            )

            model_class._keys_to_ignore_on_load_unexpected = keys_to_ignore_on_load_unexpected
            model_class._keys_to_ignore_on_load_missing = keys_to_ignore_on_load_missing

            if not os.path.isdir(model_name_or_path) and not os.path.isfile(model_name_or_path):  # pragma: no cover
                # pylint: disable=E0611
                from packaging.version import Version

                if Version(transformers.__version__) < Version("4.22.0"):
                    from transformers.file_utils import cached_path, hf_bucket_url

                    weights_file = hf_bucket_url(model_name_or_path, filename=WEIGHTS_NAME, revision=revision)
                    try:
                        # Load from URL or cache if already cached
                        resolved_weights_file = cached_path(
                            weights_file,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            use_auth_token=use_auth_token,
                        )
                    except EnvironmentError as err:  # pragma: no cover
                        logger.error(err)
                        msg = (
                            f"Can't load weights for '{model_name_or_path}'. Make sure that:\n\n"
                            f"- '{model_name_or_path}' is a correct model identifier "
                            f"listed on 'https://huggingface.co/models'\n  (make sure "
                            f"'{model_name_or_path}' is not a path to a local directory with "
                            f"something else, in that case)\n\n- or '{model_name_or_path}' is "
                            f"the correct path to a directory containing a file "
                            f"named one of {WEIGHTS_NAME}\n\n"
                        )
                        if revision is not None:
                            msg += (
                                f"- or '{revision}' is a valid git identifier "
                                f"(branch name, a tag name, or a commit id) that "
                                f"exists for this model name as listed on its model "
                                f"page on 'https://huggingface.co/models'\n\n"
                            )
                        raise EnvironmentError(msg)
                else:
                    from pathlib import Path

                    from huggingface_hub import hf_hub_download
                    from transformers.utils import TRANSFORMERS_CACHE, is_offline_mode

                    local_files_only = False
                    if is_offline_mode():
                        logger.info("Offline mode: forcing local_files_only=True")
                        local_files_only = True
                    if cache_dir is None:
                        cache_dir = TRANSFORMERS_CACHE
                    if isinstance(cache_dir, Path):
                        cache_dir = str(cache_dir)
                    try:
                        resolved_weights_file = hf_hub_download(
                            repo_id=model_name_or_path,
                            filename=WEIGHTS_NAME,
                            revision=revision,
                            cache_dir=cache_dir,
                            local_files_only=local_files_only,
                        )
                    except EnvironmentError as err:
                        logger.error(err)
                        msg = (
                            f"Can't load weights for '{model_name_or_path}'. Make sure that:\n\n"
                            f"- '{model_name_or_path}' is a correct model identifier "
                            f"listed on 'https://huggingface.co/models'\n  (make sure "
                            f"'{model_name_or_path}' is not a path to a local directory with "
                            f"something else, in that case)\n\n- or '{model_name_or_path}' is "
                            f"the correct path to a directory containing a file "
                            f"named one of {WEIGHTS_NAME}\n\n"
                        )
                        if revision is not None:
                            msg += (
                                f"- or '{revision}' is a valid git identifier "
                                f"(branch name, a tag name, or a commit id) that "
                                f"exists for this model name as listed on its model "
                                f"page on 'https://huggingface.co/models'\n\n"
                            )
                        raise EnvironmentError(msg)

                q_model = load(resolved_weights_file, model)
            else:
                weights_file = os.path.join(os.path.abspath(os.path.expanduser(model_name_or_path)), WEIGHTS_NAME)
                q_model = load(weights_file, model)

            del model
            return q_model


def save_for_huggingface_upstream(model, tokenizer, output_dir):
    """Save the model and tokenizer in the output directory."""
    tokenizer.save_pretrained(output_dir)
    torch.save(model.quantized_state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
    # save configure dtype as int8 for load identification
    model.model.config.architectures = [model.model.__class__.__name__]
    model.model.config.torch_dtype = "int8"
    model.model.config.save_pretrained(output_dir)


def export_compressed_model(
    model,
    saved_dir=None,
    use_optimum_format=True,
    enable_full_range=False,
    compression_dtype=torch.int32,
    compression_dim=1,
    scale_dtype=torch.float32,
    device="cpu",
):
    """Support get compressed model from saved_dir.

    Args:
        model (torch.nn.Module): origin fp32 model.
        saved_dir (_type_, optional): the dir path of compression info. Defaults to None.
        use_optimum_format (bool, optional): whether use HuggingFace format. Defaults to True.
        enable_full_range (bool, optional): Whether to leverage the full compression range
                                            under symmetric quantization. Defaults to False.
        compression_dtype (torch.Tensor, optional): The target dtype after comoression.
                                                    Defaults to torch.int32.
        compression_dim (int, optional): Select from [0, 1], 0 is output channel,
                                            1 is input channel. Defaults to 1.
        scale_dtype (torch.Tensor, optional): Use float32 or float16.
                                                Defaults to torch.float32.
        device (str, optional): choose device for compression. Defaults to cpu.
    """
    stat_dict = os.path.join(saved_dir, "best_model.pt")
    qweight_config_path = os.path.join(saved_dir, "qconfig.json")
    gptq_config_path = os.path.join(saved_dir, "gptq_config.json")
    if not os.path.exists(gptq_config_path):
        gptq_config_path = None
    model.load_state_dict(torch.load(stat_dict))

    from neural_compressor.model import Model as INCModel

    # pylint: disable=E1101
    inc_model = INCModel(model)
    inc_model.export_compressed_model(
        qweight_config_path=qweight_config_path,
        enable_full_range=enable_full_range,
        compression_dtype=compression_dtype,
        compression_dim=compression_dim,
        scale_dtype=scale_dtype,
        gptq_config_path=gptq_config_path,
        device=device,
        use_optimum_format=use_optimum_format,
    )
    return inc_model.model
