# Copyright (c) 2023 MIT HAN Lab
# This source code is licensed under the MIT license
#
# Copyright (c) 2023 Intel Corporation
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


import copy
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import onnx
import onnxruntime as ort
from packaging.version import Version

from neural_compressor.onnxrt.algorithms.weight_only.utility import (
    make_matmul_weight_only_node,
    pad_tensor,
    prepare_inputs,
    quant_tensor,
)
from neural_compressor.onnxrt.quantization.calibrate import CalibrationDataReader
from neural_compressor.onnxrt.quantization.config import GPTQConfig
from neural_compressor.onnxrt.utils.onnx_model import ONNXModel
from neural_compressor.onnxrt.utils.utility import (
    ONNXRT116_VERSION,
    ONNXRT1161_VERSION,
    dtype_mapping,
    simple_progress_bar,
)

__all__ = [
    "apply_gptq_on_model",
    "gptq_quantize",
]


def _gptq(
    W: np.array,
    H: np.array,
    num_bits: int = 4,
    group_size: int = 32,
    scheme: str = "asym",
    blocksize: int = 128,
    percdamp: float = 0.01,
    actorder: bool = False,
    mse: bool = False,
    perchannel: bool = True,
):
    """Quant the weight with GPTQ method.

    Args:
        W (np.array): weight.
        H (np.array): Hessian matrix.
        num_bits (int, optional): num_bits. Default is 4.
        group_size (int, optional): how many elements share one scale/zp. Default is 32.
        scheme (str, optional): sym or asym. Defaults to "asym".
        blocksize (int, optional): blocksize to quantize weight.
        percdamp (float, optional): percent of the average Hessian diagonal to use for dampening.
        actorder (bool, optional): whether rearrange Hessian matrix considering the diag's value.
        mse (bool, optional): whether get scale and zero point with mse error.
        perchannel (bool, optional): whether quantize weight per-channel.

    Returns:
        Q: fake quantized weight
    """
    Qs = []
    maxq = 2**num_bits - 1
    grid = 100
    maxshrink = 0.8
    norm = 2.4

    def find_params(weight):
        org_shape = weight.shape
        # find zp, scale
        if not perchannel:
            weight = np.expand_dims(weight.flatten(), axis=1)
        tmp = np.zeros(weight.shape[1])
        xmin = np.minimum(np.min(weight, axis=0), tmp)
        xmax = np.maximum(np.max(weight, axis=0), tmp)
        if scheme == "sym":
            xmax = np.maximum(np.abs(xmin), xmax)
            tmp = xmin < 0
            if np.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        scale = (xmax - xmin) / maxq
        if scheme == "sym":
            zero = np.ones(scale.shape) * (maxq + 1) / 2
        else:
            zero = np.round(-xmin / scale)
        if mse:
            best = np.ones([weight.shape[1]]) * float("inf")
            for i in range(int(maxshrink * grid)):
                p = 1 - i / grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / maxq
                zero1 = np.round(-xmin1 / scale1) if scheme != "sym" else zero
                q = np.clip(np.round(weight / scale1) + zero1, 0, maxq)
                q -= weight
                q = np.power(np.abs(q), norm)
                err = np.sum(q, 0)
                tmp = err < best
                if np.any(tmp):
                    best[tmp] = err[tmp]
                    scale[tmp] = scale1[tmp]
                    zero[tmp] = zero1[tmp]
        if not perchannel:
            tmp = org_shape[1]
            scale = np.repeat(scale, tmp)
            zero = np.repeat(zero, tmp)
        shape = [-1] + [1] * (len(org_shape) - 1)
        scale = np.reshape(scale, shape)
        zero = np.reshape(zero, shape)
        return scale, zero

    scales = []
    zps = []
    shape = W.shape
    scale, zp = find_params(W)
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    W[dead, :] = 0  # such channel makes no contribution to quantization computation

    # rearrange considering the diag's value
    if actorder:
        perm = np.argsort(np.diag(H))[::-1]
        W = W[perm, :]
        H = H[perm, :][:, perm]
    Losses = np.zeros(W.shape)
    Q = np.zeros(W.shape)
    damp = percdamp * np.mean(np.diag(H))
    diag = np.arange(shape[0])
    H[diag, diag] += damp  # add a average value of
    H = np.linalg.cholesky(np.linalg.inv(H)).T
    Hinv = H
    for i1 in range(0, shape[0], blocksize):
        i2 = min(i1 + blocksize, shape[0])
        count = i2 - i1

        W1 = copy.deepcopy(W[i1:i2, :])
        Q1 = np.zeros(W1.shape)
        Err1 = np.zeros(W1.shape)
        Losses1 = np.zeros(W1.shape)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):  # within a block, channel wise
            w = W1[i, :]
            d = Hinv1[i, i]

            if group_size != -1:
                if (i1 + i) % group_size == 0:
                    scale, zp = find_params(W[(i1 + i) : (i1 + i + group_size), :])

            q = (scale * (np.clip(np.round(np.expand_dims(w, axis=1) / scale) + zp, 0, maxq) - zp)).flatten()
            Q1[i, :] = q
            Losses1[i, :] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[i:, :] -= np.matmul(np.expand_dims(Hinv1[i:, i], axis=1), np.expand_dims(err1, axis=0))
            Err1[i, :] = err1

        Q[i1:i2, :] = Q1
        Losses[i1:i2, :] = Losses1 / 2

        W[i2:, :] -= np.matmul(Hinv[i2:, i1:i2], Err1)

    if actorder:
        invperm = np.argsort(perm)
        Q = Q[invperm, :]

    Q = np.reshape(Q, W.shape)
    del W
    return Q


def gptq_quantize(
    model: Union[onnx.ModelProto, ONNXModel, Path, str],
    data_reader: CalibrationDataReader,
    weight_config: dict = {},
    num_bits: int = 4,
    group_size: int = 32,
    scheme: str = "asym",
    percdamp: float = 0.01,
    blocksize: int = 128,
    actorder: bool = False,
    mse: bool = False,
    perchannel: bool = True,
    accuracy_level: int = 0,
    providers: List[str] = ["CPUExecutionProvider"],
    return_modelproto: bool = True,
):
    """Quant the model with GPTQ method.

    Args:
        model (Union[onnx.ModelProto, ONNXModel, Path, str]): onnx model.
        data_reader (CalibrationDataReader): data_reader for calibration.
        weight_config (dict, optional): quantization config
            For example,
            weight_config = {
                '(fc2, "MatMul")':
                    {
                        'weight_dtype': 'int',
                        'weight_bits': 4,
                        'weight_group_size': 32,
                        'weight_sym': True,
                        'accuracy_level': 0
                    }. Defaults to {}.
        num_bits (int, optional): number of bits used to represent weights. Defaults to 4.
        group_size (int, optional): size of weight groups. Defaults to 32.
        scheme (str, optional): indicates whether weights are symmetric. Defaults to "asym".
        percdamp (float, optional): percentage of Hessian's diagonal values' average, which will be added
            to Hessian's diagonal to increase numerical stability. Defaults to 0.01.
        blocksize (int, optional): execute GPTQ quantization per block. Defaults to 128.
        actorder (bool, optional): whether to sort Hessian's diagonal values to rearrange channel-wise
            quantization order. Defaults to False.
        mse (bool, optional): whether get scale and zero point with mse error. Defaults to False.
        perchannel (bool, optional): whether quantize weight per-channel. Defaults to True.
        accuracy_level (int, optional): accuracy level. Support 0 (unset),
            1(fp32 compute type of jblas kernel), 2 (fp16 compute type of jblas kernel),
            3 (bf16 compute type of jblas kernel), 4 (int8 compute type of jblas kernel). Defaults to 0.
        providers (list, optional): providers to use. Defaults to ["CPUExecutionProvider"].
        return_modelproto (bool, optionmal): whether to return onnx.Modelproto. set False for layer-wise quant.
            Default to True

    Returns:
        onnx.ModelProto: quantized onnx model
    """
    if not isinstance(model, ONNXModel):
        model = ONNXModel(model)
    base_dir = os.path.dirname(model.model_path) if model.model_path is not None else ""

    inputs, so = prepare_inputs(model, data_reader, providers)
    del data_reader
    org_output = copy.deepcopy(model.model.graph.output)
    model.remove_tensors_from_outputs([i.name for i in org_output])
    output_names = []
    for node in model.nodes():
        # check op_type of node is MatMul
        # check dim 1 of input is weight tensor
        # check weight_type is not "fp32"
        if (
            node.op_type in ["MatMul"]
            and model.get_initializer(node.input[1]) is not None
            and weight_config.get((node.name, node.op_type), {}).get("weight_dtype", "fp32") != "fp32"
        ):
            output_names.append(node.input[0])
    output_names = list(set(output_names))
    model.add_tensors_to_outputs(output_names)
    if model.is_large_model:
        onnx.save_model(
            model.model,
            model.model_path + "_augment.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            convert_attribute=False,
        )

    session = (
        ort.InferenceSession(model.model.SerializeToString(), so, providers=providers)
        if not model.is_large_model
        else ort.InferenceSession(model.model_path + "_augment.onnx", so, providers=providers)
    )

    for idx, input_name in enumerate(output_names):
        simple_progress_bar(len(output_names), idx + 1)
        node_list = []
        weights = []

        for node in model.input_name_to_nodes()[input_name]:
            # check op_type of node is MatMul
            # check dim 1 of input is weight tensor
            # check weight_type is not "fp32"
            if (
                node.op_type in ["MatMul"]
                and model.get_initializer(node.input[1]) is not None
                and weight_config.get((node.name, node.op_type), {}).get("weight_dtype", "fp32") != "fp32"
            ):
                weight = onnx.numpy_helper.to_array(
                    model.get_initializer(model.get_node(node.name).input[1]), base_dir
                ).copy()
                if len(weight.shape) != 2:
                    continue

                weights.append(weight)
                node_list.append(model.get_node(node.name))

        if len(weights) == 0:
            continue

        Hs = [np.zeros((i.shape[0], i.shape[0])) for i in weights]
        nsamples = 0
        for data in inputs:
            inp = session.run([input_name], data)[0]
            tmp = inp.shape[0]
            inp = np.reshape(inp, (-1, inp.shape[-1]))
            Hs = [i * (nsamples / (nsamples + tmp)) for i in Hs]
            nsamples += tmp
            inp = np.sqrt(2 / nsamples) * inp
            Hs = [i + np.matmul(inp.T, inp) for i in Hs]

        for (
            node,
            weight,
            H,
        ) in zip(node_list, weights, Hs):
            if node.name in weight_config:
                num_bits = weight_config[node.name]["bits"]
                group_size = weight_config[node.name]["group_size"]
                scheme = weight_config[node.name]["scheme"]
                accuracy_level = weight_config[(node.name, node.op_type)].accuracy_level
            group_size = group_size if group_size != -1 else weight.shape[0]
            dtype = weight.dtype

            q_weight = _gptq(
                weight,
                H,
                num_bits=num_bits,
                group_size=group_size,
                scheme=scheme,
                blocksize=blocksize,
                percdamp=percdamp,
                actorder=actorder,
                mse=mse,
                perchannel=perchannel,
            )

            weight_tensor = model.get_initializer(node.input[1])
            init_share_num = model.get_initializer_share_num(node.input[1])

            satisfy_MatMulNBits_condition = Version(ort.__version__) > ONNXRT1161_VERSION and num_bits == 4
            satisfy_MatMulFpQ4_condition = (
                Version(ort.__version__) >= ONNXRT116_VERSION and num_bits == 4 and group_size == 32
            )
            if ("CUDAExecutionProvider" in providers and satisfy_MatMulNBits_condition) or (
                "CUDAExecutionProvider" not in providers
                and (satisfy_MatMulFpQ4_condition or satisfy_MatMulNBits_condition)
            ):  # pragma: no cover
                # MatMulFpQ4 support 4 bits and 32 group_size with ort 1.16.0 and 1.16.1 versions, supported by CPU EP
                # MatMulNBits supports 4 bits and 2^n group_size with ort > 1.16.1, supported by CPU EP AND CUDA EP
                org_shape = weight.shape
                k_blocks = (org_shape[0] + group_size - 1) // group_size
                q_weight = pad_tensor(q_weight, group_size, k_blocks)
                q_weight, scale, zp = quant_tensor(q_weight.T, num_bits, group_size, scheme, "uint")
                q_matmul_node, new_inits = make_matmul_weight_only_node(
                    node=node,
                    weight_shape=org_shape,
                    num_bits=num_bits,
                    group_size=group_size,
                    k_blocks=k_blocks,
                    q_weight=q_weight.astype("uint8"),
                    scale=scale.astype(dtype),
                    zero_point=zp if scheme == "asym" else None,
                    accuracy_level=accuracy_level,
                )

                model.add_initializers(new_inits)
                model.remove_node(node)
                model.add_node(q_matmul_node)
            else:
                q_weight_tensor = onnx.helper.make_tensor(
                    name=node.input[1] + "_Q{}G{}".format(str(num_bits), str(group_size)),
                    data_type=dtype_mapping[str(dtype)],
                    dims=q_weight.shape,
                    vals=q_weight.astype(dtype).tobytes(),
                    raw=True,
                )
                model.add_initializer(q_weight_tensor)
                node.input[1] = q_weight_tensor.name
            if init_share_num == 1:
                model.remove_initializer(weight_tensor)

    model.remove_tensors_from_outputs(output_names)
    model.model.graph.output.MergeFrom(org_output)

    model.topological_sort()

    # reload external data to prevent external data file path errors
    if model.is_large_model:
        from onnx.external_data_helper import load_external_data_for_model

        load_external_data_for_model(model.model, os.path.split(model.model_path)[0])

    if return_modelproto:
        return model.model
    else:
        return model


def apply_gptq_on_model(
    model: Union[onnx.ModelProto, ONNXModel, Path, str],
    quant_config: dict,
    calibration_data_reader: CalibrationDataReader,
) -> onnx.ModelProto:
    """Apply GPTQ on onnx model.

    Args:
        model (Union[onnx.ModelProto, ONNXModel, Path, str]): onnx model.
        quant_config (dict): quantization config.
        calibration_data_reader (CalibrationDataReader): data_reader for calibration.

    Returns:
        onnx.ModelProto: quantized onnx model.
    """
    # check whether to do layer_wise quant
    layer_wise = quant_config.pop("layer_wise_quant", False)

    # set other model params
    quant_kwargs = {}
    quant_kwargs = {key: quant_config.pop(key) for key in GPTQConfig.model_params_list if key in quant_config}

    # change op config to dict type
    for op_name_type, op_config in quant_config.items():
        if isinstance(op_config, GPTQConfig):
            quant_config[op_name_type] = op_config.to_dict()

    if layer_wise:
        from neural_compressor.onnxrt.algorithms import layer_wise_quant

        quantized_model = layer_wise_quant(
            model,
            quant_func=gptq_quantize,
            weight_config=quant_config,
            data_reader=calibration_data_reader,
            **quant_kwargs
        )
    else:
        quantized_model = gptq_quantize(
            model, data_reader=calibration_data_reader, weight_config=quant_config, **quant_kwargs
        )

    if isinstance(quantized_model, ONNXModel):
        quantized_model = quantized_model.model
    return quantized_model
