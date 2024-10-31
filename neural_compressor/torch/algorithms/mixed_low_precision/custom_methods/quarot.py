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

import typing

import torch
import tqdm

from .quarot_utils import get_hadK

# This code implements rotations to the model, and is based on the paper below:
# "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs". See https://github.com/spcl/QuaRot/tree/main.
# The script rotates weights/activations with Hadamard matrices to reduce outliers and improve quantization.
# Tested on llama2-7b, llama2-13b, llama3-8b.
# The code is compatible with GPTQ (rotation is applied first, followed by GPTQ on the rotated model).
# Rotating the weights and the values is done offline, and does not decrease inference speed.
# Rotating the MLP layers is partially online, which adds computational overhead.
# To rotate, call the function "rotate(model, args)".
# Calling the function rotates the weights. To also rotate the values and/or MLP, pass the arguments:
# args.rotate_values and/or args.rotate_mlp.

DEV = "hpu"


def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model) -> None:
    # Embedding fusion
    W = model.model.embed_tokens
    W_ = W.weight.data.double()
    W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    layers = [layer for layer in model.model.layers]
    for layer in layers:
        fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
        fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        W_norm = layer.post_attention_layernorm.weight.data
        # We moved the parameters to the weights matrices, thus we replace them with ones:
        layer.post_attention_layernorm.weight.data = torch.ones_like(W_norm)
        W_norm = layer.input_layernorm.weight.data
        layer.input_layernorm.weight.data = torch.ones_like(W_norm)
    fuse_ln_linear(model.model.norm, [model.lm_head])
    W_norm = model.model.norm.weight.data
    model.model.norm.weight.data = torch.ones_like(W_norm)


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output
    if K > 1:
        input = hadK.view(1, K, K).to(input) @ input
    return input.view(X.shape) / torch.tensor(n).sqrt()


def get_orthogonal_matrix(size, random=False):
    # See https://cornell-relaxml.github.io/quip-sharp/
    if random:
        Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
        Q = Q * 2 - 1
    else:
        Q = torch.ones(size, dtype=torch.float64)
    Q = torch.diag(Q)
    return matmul_hadU(Q)


def get_kron_hadamard(size):
    hadK, K = get_hadK(size)
    normalization = torch.sqrt(torch.tensor(K))
    hadK = hadK / normalization
    p_hadamard = get_orthogonal_matrix(int(size / K), random=False)
    return torch.kron(p_hadamard, hadK)


def rotate_head(model, Q: torch.Tensor) -> None:
    W = model.lm_head
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=model.dtype)


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    for W in [model.model.embed_tokens]:
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=model.dtype)


def rotate_attention_inputs(layer, Q) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="hpu", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q) -> None:
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q) -> None:
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotation_mlp_pre_hook(module, input):
    # This add online rotation in the mlp layer.
    rotated_input = torch.matmul(input[0], module.H_down)
    return rotated_input


def add_forward_mlp_wrapper(model) -> None:
    # This add online rotation in the mlp layer if and is used on you load a rotated model.
    config = model.config
    H_down = get_kron_hadamard(config.intermediate_size).to(device=DEV, dtype=torch.float64)
    layers = [layer for layer in model.model.layers]
    for layer in layers:
        W = layer.mlp.down_proj
        W.register_buffer("H_down", H_down.to(device="cpu", dtype=model.dtype))
        hook_handle = W.register_forward_pre_hook(rotation_mlp_pre_hook)


def rotate_mlp_output(layer, config, Q) -> None:
    # This function rotates the weight matrices at the output of the mlp layer.
    # when rotating the activations within the layer, the function adds a hook which perform online rotation.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if hasattr(config, "rotate_mlp") and config.rotate_mlp:
        H_down = get_kron_hadamard(config.intermediate_size).to(device=DEV, dtype=torch.float64)
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, H_down).to(device="cpu", dtype=dtype)
        W.register_buffer("H_down", H_down.to(device="cpu", dtype=dtype))
        hook_handle = W.register_forward_pre_hook(rotation_mlp_pre_hook)
    if W.bias is not None:
        b = W.bias.data.to(device=DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer) -> None:
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    dtype = o_proj.weight.dtype
    num_q_heads = layer.self_attn.num_heads
    num_kv_heads = layer.self_attn.num_key_value_heads
    head_dim = layer.self_attn.head_dim
    H_head = get_orthogonal_matrix(head_dim, random=False).to(device=DEV, dtype=torch.float64)
    I_v = torch.eye(num_kv_heads).to(device=DEV, dtype=torch.float64)
    I_out = torch.eye(num_q_heads).to(device=DEV, dtype=torch.float64)
    H_v = torch.kron(I_v, H_head)
    H_out = torch.kron(I_out, H_head)
    W_ = v_proj.weight.data.to(device=DEV, dtype=torch.float64)
    v_proj.weight.data = torch.matmul(H_v.T, W_).to(device="cpu", dtype=dtype)
    W_ = o_proj.weight.data.to(device=DEV, dtype=torch.float64)
    o_proj.weight.data = torch.matmul(W_, H_out).to(device="cpu", dtype=dtype)


def rotate_model(model) -> None:
    Q = get_orthogonal_matrix(model.config.hidden_size, random=True).to(device=DEV, dtype=torch.float64)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q)
        rotate_attention_output(layers[idx], Q)
        rotate_mlp_input(layers[idx], Q)
        rotate_mlp_output(layers[idx], model.config, Q)
        if hasattr(model.config, "rotate_values") and model.config.rotate_values:
            rotate_ov_proj(layers[idx])


def rotate(model, args) -> None:
    model.config.rotate_weights = True
    if args.rotate_mlp:
        model.config.rotate_mlp = True
    if args.rotate_values:
        model.config.rotate_values = True
    fuse_layer_norms(model)
    rotate_model(model)
