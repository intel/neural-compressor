# Copyright (c) 2025 Intel Corporation
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
import os
from collections.abc import Callable
from typing import Any

_TRUTHY_ENV_VALUES = ("1", "true", "yes")


def _env_flag(env_name: str, default: str = "0") -> bool:
    value = os.getenv(env_name, default)
    return value.lower() in _TRUTHY_ENV_VALUES


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """Create a lambda that validates environment variable against allowed choices.

    Args:
        env_name: Name of the environment variable
        default: Default value if not set (can be None)
        choices: List of valid string options or callable that returns list
        case_sensitive: Whether validation should be case sensitive

    Returns:
        Lambda function for environment_variables dict
    """

    def _get_validated_env() -> str | None:
        value = os.getenv(env_name)
        if value is None:
            return default

        # Resolve choices if it's a callable (for lazy loading)
        actual_choices = choices() if callable(choices) else choices

        if not case_sensitive:
            check_value = value.lower()
            check_choices = [choice.lower() for choice in actual_choices]
        else:
            check_value = value
            check_choices = actual_choices

        if check_value not in check_choices:
            raise ValueError(f"Invalid value '{value}' for {env_name}. " f"Valid options: {actual_choices}.")

        if not case_sensitive:
            for choice in actual_choices:
                if choice.lower() == check_value:
                    return choice

        return value

    return _get_validated_env


environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_QDQ_TRACE": lambda: _env_flag("VLLM_QDQ_TRACE"),
    "VLLM_QDQ": lambda: _env_flag("VLLM_QDQ"),
    "VLLM_MARLIN_MOE_QDQ_MODE": env_with_choices(
        "VLLM_MARLIN_MOE_QDQ_MODE",
        default="0",
        choices=["0", "FORCE_MXFP4"],
        case_sensitive=False,
    ),
    # sage3 Triton attention plugin
    "VLLM_SAGE3_TRITON": lambda: _env_flag("VLLM_SAGE3_TRITON"),
    "VLLM_SAGE3_CUTE": lambda: _env_flag("VLLM_SAGE3_CUTE"),
    "SAGE3_QUANT_FORMAT": lambda: os.getenv("SAGE3_QUANT_FORMAT", "mxfp4"),
    "SAGE3_ACC_DTYPE": lambda: os.getenv("SAGE3_ACC_DTYPE", "fp32"),
    # Per-(layer, denoising-step) attention quant-config routing.
    # Path to a JSON route file; empty string disables routing.
    "SAGE3_ROUTE_FILE": lambda: os.getenv("SAGE3_ROUTE_FILE", ""),
    # When truthy, log which quant config the router resolves per (layer, step).
    "SAGE3_ROUTE_DEBUG": lambda: _env_flag("SAGE3_ROUTE_DEBUG"),
    # SpargeAttn block-sparse attention backend (mutually exclusive with sage3).
    "VLLM_SPARGE_ATTN": lambda: _env_flag("VLLM_SPARGE_ATTN"),
    # Path to the SpargeAttn repo; injected on sys.path at registration when the
    # spas_sage_attn package is not already importable. Empty string = rely on the
    # package already being installed/importable.
    "SPARGE_ATTN_REPO": lambda: os.getenv("SPARGE_ATTN_REPO", ""),
    # "topk" (block-sparsity by fraction kept) or "cdfthreshd" (adaptive CDF threshold).
    "SPARGE_MODE": env_with_choices(
        "SPARGE_MODE",
        default="topk",
        choices=["topk", "cdfthreshd"],
        case_sensitive=False,
    ),
    # Fraction of KV blocks to keep when SPARGE_MODE=topk. 1.0 = dense (keep all),
    # which isolates quantization error for correctness checks.
    "SPARGE_TOPK": lambda: os.getenv("SPARGE_TOPK", "1.0"),
    # CDF threshold when SPARGE_MODE=cdfthreshd (keep blocks up to this prob mass).
    "SPARGE_CDFTHRESHD": lambda: os.getenv("SPARGE_CDFTHRESHD", "0.98"),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
