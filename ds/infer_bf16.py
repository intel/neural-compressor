# ==--------------------------------------------------------------------------==
# Patch for loading DS models
import os
from typing import Optional, Union
from zipfile import is_zipfile

import torch
from packaging import version
from transformers.integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from transformers.utils import is_safetensors_available, strtobool

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file


def is_fsdp_enabled():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def is_local_dist_rank_0():
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and int(os.environ.get("LOCAL_RANK", -1)) == 0
    )


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],
    is_quantized: bool = False,
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
):
    """Reads a PyTorch checkpoint file, returning properly formatted errors if they arise."""

    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        with safe_open(checkpoint_file, framework="pt") as f:
            metadata = f.metadata()
        if metadata is not None and metadata.get("format") not in ["pt", "tf", "flax", "mlx"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        return safe_load_file(checkpoint_file)
    try:
        if map_location is None:
            if (
                (
                    is_deepspeed_zero3_enabled()
                    and torch.distributed.is_initialized()
                    and torch.distributed.get_rank() > 0
                )
                or (is_fsdp_enabled() and not is_local_dist_rank_0())
            ) and not is_quantized:
                map_location = "meta"
            else:
                map_location = "cpu"
        extra_args = {}
        # mmap can only be used with files serialized with zipfile-based format.
        if (
            isinstance(checkpoint_file, str)
            and map_location != "meta"
            and version.parse(torch.__version__) >= version.parse("2.1.0")
            and is_zipfile(checkpoint_file)
        ):
            extra_args = {"mmap": True}
        weights_only_kwarg = {"weights_only": weights_only}
        return torch.load(
            checkpoint_file,
            map_location=map_location,
            **weights_only_kwarg,
            **extra_args,
        )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read(7) == "version":
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for '{checkpoint_file}' "
                f"at '{checkpoint_file}'. "
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


def set_initialized_submodules(model, state_dict_keys):
    """Sets the `_is_hf_initialized` flag in all submodules of a given model when all its weights are in the loaded state
    dict."""
    state_dict_keys = set(state_dict_keys)
    not_initialized_submodules = {}
    for module_name, module in model.named_modules():
        if module_name == "":
            # When checking if the root module is loaded there's no need to prepend module_name.
            module_keys = set(module.state_dict())
        else:
            module_keys = {f"{module_name}.{k}" for k in module.state_dict()}
        if module_keys.issubset(state_dict_keys):
            module._is_hf_initialized = True
        else:
            not_initialized_submodules[module_name] = module
    return not_initialized_submodules


# ==--------------------------------------------------------------------------==


def patch_transformers():
    import transformers

    transformers.modeling_utils.load_state_dict = load_state_dict
    transformers.modeling_utils.set_initialized_submodules = set_initialized_submodules


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def eval(model_path):
    import transformers
    from transformers.modeling_utils import no_init_weights

    # from patch_for_ds import patch_transformers
    # if not not_patch_lin:
    #     patch_lin()

    def _patch__initialize_weights(self, module):
        print("Skipping init_weights ")
        module._is_hf_initialized = True

    transformers.modeling_utils.PreTrainedModel._initialize_weights = _patch__initialize_weights
    patch_transformers()
    with no_init_weights():
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    logger.info(f"Patched model: {model}")
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    prompt = "Hi, who"
    encode = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_tokens = model.generate(encode, max_length=10)
        output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Output: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--qmodel_path", type=str, required=True)
    parser.add_argument("--not_patch_lin", action="store_true", help="Measure float model")
    args = parser.parse_args()
    eval(args.qmodel_path)
