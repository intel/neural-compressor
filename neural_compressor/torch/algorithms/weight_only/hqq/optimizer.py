# Copyright (c) 2023-2024 Mobiusml and Intel Corporation
#
# This code is based on Mobiusml's HQQ library.
# https://github.com/mobiusml/hqq
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
"""Optimization logic of HQQ."""

import numpy as np
import torch

from neural_compressor.torch.utils import logger
from neural_compressor.torch.utils.auto_accelerator import auto_detect_accelerator


# Proximal solver || W - dequantize(quantize(W))||_p^p
@torch.inference_mode()
def optimize_weights_proximal_legacy(
    tensor,
    scale,
    zero,
    min_max,
    axis=0,
    device="cuda",
    opt_params={"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
    verbose=False,
):
    """Quantize the scale/zero of quantized tensor using the HQQ.

    Args:
        tensor (torch.Tensor): The input tensor to optimize.
        scale (torch.Tensor): The scaling factor for quantization.
        zero (torch.Tensor): The zero-point for quantization.
        min_max (tuple): The minimum and maximum values for quantization.
        axis (int, optional): The axis along which to compute the mean for zero-point calculation. Defaults to 0.
        device (str, optional): The device to use for computation. Defaults to "cuda".
        opt_params (dict, optional): Optimization parameters.
            Defaults to {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20}.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        tuple: A tuple containing the optimized scale and zero-point tensors.
    """
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )
    device = auto_detect_accelerator().current_device()

    # TODO: refine it for cpu device
    if auto_detect_accelerator().name() == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    W_f = tensor.to(dtype).to(device)
    scale = scale.to(dtype).to(device)
    zero = zero.to(dtype).to(device)

    if lp_norm == 1:
        shrink_op = lambda x, beta: torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        shrink_op = lambda x, beta, p=lp_norm: torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), p - 1)
        )

    best_error = 1e4
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = shrink_op(W_f - W_r, beta)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(W_f - W_r).mean())
        if verbose:
            logger.info(i, np.round(current_error, 6))
        if current_error < best_error:
            best_error = current_error
        else:
            break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e
    auto_detect_accelerator().empty_cache()

    return scale, zero


optimize_weights_proximal = optimize_weights_proximal_legacy
