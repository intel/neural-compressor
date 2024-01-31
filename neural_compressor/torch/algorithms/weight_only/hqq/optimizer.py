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

# Refactored from https://github.com/mobiusml/hqq

import numpy as np
import torch
from auto_accelerator import Auto_Accelerator, auto_detect_accelerator
from utility import dump_elapsed_time

auto_acceleartor: Auto_Accelerator = auto_detect_accelerator()


# Proximal solver || W - dequantize(quantize(W))||_p^p
@torch.inference_mode()
@dump_elapsed_time("Optimize weights ....")
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
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )
    device = auto_acceleartor.current_device_name()
    # TODO: how about xpu?

    if auto_acceleartor.name() == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"[optimize_weights_proximal_legacy] dtype: {dtype}, accelerator: {auto_acceleartor.name}")
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
            print(i, np.round(current_error, 6))
        if current_error < best_error:
            best_error = current_error
        else:
            break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e
    auto_acceleartor.empty_cache()

    return scale, zero


optimize_weights_proximal = optimize_weights_proximal_legacy
