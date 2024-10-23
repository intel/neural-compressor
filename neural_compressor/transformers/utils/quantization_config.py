# -*- coding: utf-8 -*-
# Copyright (c) 2024 Intel Corporation
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
"""Intel Neural Compressor Transformers-like Config."""

import os
from typing import Any, Dict, Tuple, Union

from neural_compressor.common.utils import LazyImport, logger

torch = LazyImport("torch")
transformers = LazyImport("transformers")

QUANT_CONFIG = "quantize_config.json"

if transformers.__version__ >= "4.32.0":
    from transformers.utils.quantization_config import QuantizationConfigMixin

    QuantizationConfig = QuantizationConfigMixin
else:
    from transformers import PretrainedConfig

    QuantizationConfig = PretrainedConfig
from enum import Enum


class QuantizationMethod(str, Enum):
    GPTQ = "gptq"
    RTN = "rtn"
    AWQ = "awq"
    AUTOROUND = "autoround"
    TEQ = "teq"


class INCQuantizationConfigMixin(QuantizationConfig):
    """Mixin class for quantization config."""

    def update(self, **kwargs):
        """Updates attributes of this class instance with attributes from `kwargs` if they match existing atributtes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs

    def post_init_cpu(self):
        r"""Safety checker that arguments are correct."""

        if self.compute_dtype is not None and self.compute_dtype not in [
            "fp32",
            "bf16",
            "int8",
        ]:
            raise ValueError("compute_dtype must be 'fp32', 'bf16', 'int8'.")
        elif self.compute_dtype is None:
            self.compute_dtype = "fp32"

        if self.bits is None:
            self.bits = 4
        elif self.bits is not None and self.bits not in [4, 8]:
            raise ValueError(f"Only support quantization to [4, 8] bits but found {self.bits}")

        if self.scale_dtype is not None and self.scale_dtype not in ["fp32", "bf16", "fp16"]:
            raise ValueError("scale_dtype must be a string in 'fp32', 'bf16' ")
        elif self.scale_dtype is None:
            self.scale_dtype = "fp32"

        if not isinstance(self.group_size, int):
            raise ValueError("group_size must be a int")

        if not isinstance(self.scheme, str):
            raise ValueError("scheme must be a string")

    def post_init_xpu(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

        if self.compute_dtype is not None and self.compute_dtype not in ["fp16"]:
            raise ValueError("compute_dtype must be 'fp16'.")
        elif self.compute_dtype is None:
            self.compute_dtype = "fp16"

        if self.bits is None:
            self.bits = 4
        elif self.bits not in [4]:
            raise ValueError(f"Only support quantization to [4] bits but found {self.bits}")

        if self.weight_dtype is None:
            self.weight_dtype = "int4_fullrange"
        elif self.weight_dtype == "int4":
            self.weight_dtype = "int4_fullrange"
        elif self.weight_dtype not in [
            "int4_fullrange",
        ]:
            raise ValueError(f"weight_dtype must be a string in 'int4_fullrange', but get {self.weight_dtype}.")

        if self.scale_dtype is not None and self.scale_dtype not in ["fp16"]:
            raise ValueError("scale_dtype must be a string in 'fp16'")
        elif self.scale_dtype is None:
            self.scale_dtype = "fp16"

        if not isinstance(self.group_size, int):
            raise ValueError("group_size must be a int")

        if self.scheme not in ["sym"]:
            raise ValueError("scheme: {} is not support, only support 'sym' now!".format(self.scheme))

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        # set tokenizer to None due to it doesn't support write to json
        if hasattr(self, "tokenizer"):
            self.tokenizer = None
        if hasattr(self, "calib_dataloader"):
            self.calib_dataloader = None
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def remove_redundant_parameters(self):
        remove_parameters = [
            "calib_dataloader",
            "dataset",
            "calib_func",
            "calib_iters",
            "calib_len",
            "mse_range",
            "scheme",
            "tokenizer",
            "use_layer_wise",
            "blocksize",
            "nsamples",
            "max_input_length",
            "static_groups",
            "lr",
            "minmax_lr",
            "iters",
            "use_quant_input",
            "device",
            "calib_dataset",
            "calib_pad_val",
            "calib_shuffle",
            "calib_padding",
            "example_inputs",
            "excluded_precisions",
            "op_name_dict",
            "op_type_dict",
            "train_dataloader",
            "train_func",
            "train_iters",
            "train_len",
            "train_padding",
            "train_dataset",
            "train_pad_val",
            "train_shuffle",
            "train_batch_size",
        ]
        for parameter in remove_parameters:
            if hasattr(self, parameter):
                delattr(self, parameter)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, QUANT_CONFIG)

        self.to_json_file(output_config_file, use_diff=False)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token", None),
            )

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cf = kwargs.pop("_configuration_file", QUANT_CONFIG)
        return super().get_config_dict(pretrained_model_name_or_path, _configuration_file=cf, **kwargs)


class RtnConfig(INCQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 32,
        compute_dtype: Any = None,
        scale_dtype: Any = None,
        sym: bool = True,
        use_layer_wise: bool = None,
        quant_lm_head: bool = False,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.RTN
        self.bits = bits
        self.compute_dtype = compute_dtype
        self.weight_dtype = "int4" if self.bits == 4 else "int8"
        self.scale_dtype = scale_dtype
        self.group_size = group_size
        self.use_layer_wise = use_layer_wise
        self.model_path = kwargs.get("model_path", "")
        self.sym = sym
        self.scheme = "sym" if self.sym else "asym"

        # "transformer.output_layer" for chatglm series model.
        # "embed_out" for dolly v2 series model.
        self.quant_lm_head = quant_lm_head
        self.modules_to_not_convert = kwargs.get(
            "modules_to_not_convert", ["lm_head", "transformer.output_layer", "embed_out"]
        )
        if self.quant_lm_head:
            self.modules_to_not_convert = []
        self.device = kwargs.get("device", "auto")

    def to_diff_dict(self) -> Dict[str, Any]:
        """Removes all attributes from config which correspond to the default config attributes
        for better readability and serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = RtnConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


class GPTQConfig(INCQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 4,
        tokenizer: Any = None,
        dataset: str = "NeelNanda/pile-10k",
        batch_size: int = 8,
        group_size: int = 32,
        compute_dtype: Any = None,
        scale_dtype: Any = None,
        sym: bool = True,
        blocksize: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        n_samples: int = 128,
        seq_len: int = 2048,
        static_groups: bool = False,
        use_mse_search: bool = False,
        true_sequential: bool = False,
        use_layer_wise: bool = None,
        quant_lm_head: bool = False,
        **kwargs,
    ):

        self.quant_method = QuantizationMethod.GPTQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.compute_dtype = compute_dtype
        self.weight_dtype = "int4" if self.bits == 4 else "int8"
        self.scale_dtype = scale_dtype
        self.sym = sym
        self.blocksize = blocksize
        self.n_samples = n_samples
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.static_groups = static_groups
        self.use_mse_search = use_mse_search
        self.true_sequential = true_sequential
        self.use_layer_wise = use_layer_wise
        self.model_path = kwargs.get("model_path", "")
        self.seq_len = seq_len
        self.quant_lm_head = quant_lm_head
        self.modules_to_not_convert = kwargs.get(
            "modules_to_not_convert", ["lm_head", "transformer.output_layer", "embed_out"]
        )
        if self.quant_lm_head:
            self.modules_to_not_convert = []
        self.device = kwargs.get("device", "auto")
        self.scheme = "sym" if self.sym else "asym"

        if isinstance(compute_dtype, torch.dtype):
            self.compute_dtype = compute_dtype
        else:
            self.compute_dtype = compute_dtype

        if isinstance(scale_dtype, torch.dtype):
            self.scale_dtype = scale_dtype
        else:
            self.scale_dtype = scale_dtype

        self.post_init_gptq()

    def post_init_gptq(self):
        r"""Safety checker that arguments are correct."""

        if self.bits not in [4, 8]:
            raise ValueError(f"Only support quantization to [4, 8] bits but found {self.bits}")

        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")

    def to_diff_dict(self) -> Dict[str, Any]:
        """Removes all attributes from config which correspond to the default config attributes
        for better readability and serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = GPTQConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


class AwqConfig(INCQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 4,
        tokenizer: Any = None,
        dataset: str = "NeelNanda/pile-10k",
        group_size: int = 32,
        compute_dtype: Any = None,
        weight_dtype: Any = None,
        scale_dtype: Any = None,
        use_layer_wise: bool = None,
        n_samples: int = 128,
        seq_len: int = 2048,
        auto_scale: bool = True,
        auto_clip: bool = True,
        zero_point: bool = True,
        absorb_layer_dict: dict = {},
        quant_lm_head: bool = False,
        backend: str = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.AWQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.compute_dtype = compute_dtype
        self.weight_dtype = "int4" if self.bits == 4 else "int8"
        self.scale_dtype = scale_dtype
        self.group_size = group_size
        self.zero_point = zero_point
        self.auto_scale = auto_scale
        self.auto_clip = auto_clip
        self.use_layer_wise = use_layer_wise
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.absorb_layer_dict = absorb_layer_dict
        self.quant_lm_head = quant_lm_head
        self.backend = backend
        self.modules_to_not_convert = kwargs.get(
            "modules_to_not_convert", ["lm_head", "transformer.output_layer", "embed_out"]
        )
        if self.quant_lm_head:
            self.modules_to_not_convert = []
        self.device = kwargs.get("device", "auto")
        self.scheme = "asym" if self.zero_point else "sym"
        self.sym = True if not self.zero_point else False
        self.batch_size = kwargs.pop("batch_size", 8)

    def to_diff_dict(self) -> Dict[str, Any]:
        """Removes all attributes from config which correspond to the default config attributes
        for better readability and serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = AwqConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


class TeqConfig(INCQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 4,
        tokenizer: Any = None,
        dataset: str = "NeelNanda/pile-10k",
        group_size: int = 32,
        compute_dtype: Any = None,
        weight_dtype: Any = None,
        scale_dtype: Any = None,
        use_layer_wise: bool = None,
        n_samples: int = 128,
        seq_len: int = 2048,
        sym: bool = True,
        absorb_layer_dict: dict = {},
        quant_lm_head: bool = False,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.TEQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.compute_dtype = compute_dtype
        self.weight_dtype = "int4" if self.bits == 4 else "int8"
        self.scale_dtype = scale_dtype
        self.group_size = group_size
        self.sym = sym
        self.scheme = "sym" if self.sym else "asym"
        self.use_layer_wise = use_layer_wise
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.absorb_layer_dict = absorb_layer_dict
        self.quant_lm_head = quant_lm_head
        self.modules_to_not_convert = kwargs.get(
            "modules_to_not_convert", ["lm_head", "transformer.output_layer", "embed_out"]
        )
        if self.quant_lm_head:
            self.modules_to_not_convert = []
        self.device = kwargs.get("device", "auto")
        self.batch_size = kwargs.pop("batch_size", 8)

    def to_diff_dict(self) -> Dict[str, Any]:
        """Removes all attributes from config which correspond to the default config attributes
        for better readability and serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = TeqConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


class AutoRoundConfig(INCQuantizationConfigMixin):
    def __init__(
        self,
        bits: int = 4,
        tokenizer: Any = None,
        dataset: str = "NeelNanda/pile-10k",
        group_size: int = 128,
        compute_dtype: Any = None,
        weight_dtype: Any = None,
        scale_dtype: Any = None,
        sym: bool = False,
        lr: float = None,
        minmax_lr: float = None,
        disable_quanted_input: bool = True,
        n_samples: int = 128,
        seq_len: int = 2048,
        iters: int = 200,
        use_layer_wise: bool = None,
        quant_lm_head: bool = False,
        **kwargs,
    ):

        from neural_compressor.transformers.quantization.utils import convert_dtype_torch2str

        self.quant_method = QuantizationMethod.AUTOROUND
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.compute_dtype = compute_dtype
        self.weight_dtype = "int4" if self.bits == 4 else "int8"
        self.scale_dtype = scale_dtype
        self.sym = sym
        self.n_samples = n_samples
        self.group_size = group_size
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.disable_quanted_input = disable_quanted_input
        self.iters = iters
        self.seq_len = seq_len
        self.quant_lm_head = quant_lm_head
        self.modules_to_not_convert = kwargs.get(
            "modules_to_not_convert", ["lm_head", "transformer.output_layer", "embed_out"]
        )
        if self.quant_lm_head:
            self.modules_to_not_convert = []
        self.batch_size = kwargs.pop("batch_size", 8)
        self.device = kwargs.get("device", "auto")
        calib_iters = kwargs.get("calib_iters", None)
        if iters is not None:
            self.calib_iters = iters
            if calib_iters is not None:
                logger.info(
                    "cannot be set simultaneously for 'iters' and 'calib_iters', "
                    "we will use 'iters' as calibration iterations!"
                )
        else:
            self.calib_iters = 200 if calib_iters is None else calib_iters
        self.scheme = "sym" if self.sym else "asym"
        if isinstance(compute_dtype, torch.dtype):
            self.compute_dtype = convert_dtype_torch2str(compute_dtype)
        else:
            self.compute_dtype = compute_dtype

        if isinstance(scale_dtype, torch.dtype):
            self.scale_dtype = convert_dtype_torch2str(scale_dtype)
        else:
            self.scale_dtype = scale_dtype
        self.use_layer_wise = use_layer_wise
        self.model_path = kwargs.get("model_path", "")

    def to_diff_dict(self) -> Dict[str, Any]:
        """Removes all attributes from config which correspond to the default config attributes
        for better readability and serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = AutoRoundConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict
