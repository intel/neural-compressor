#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
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
"""SmoothQuant for onnxrt adaptor."""

import copy
import logging
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnx import onnx_pb as onnx_proto

from neural_compressor.adaptor.ox_utils.util import (
    _get_qrange_for_qType,
    is_B_transposed,
    quantize_data,
    simple_progress_bar,
    to_numpy,
)
from neural_compressor.model.model import BaseModel
from neural_compressor.model.onnx_model import ONNXModel

logger = logging.getLogger("neural_compressor")

dtype_map = {
    np.dtype("float32"): 1,
    np.dtype("uint8"): 2,
    np.dtype("int8"): 3,
    np.dtype("int32"): 6,
    np.dtype("int64"): 7,
    np.dtype("float16"): 10,
    np.dtype("double"): 11,
}


def get_quant_dequant_output(model, input_data, output_data, reduce_range, backend):
    """Get loss between fp32 output and QDQ output.

    Args:
        model (object): model
        input_data (numpy.ndarray): fp32 input
        output_data (numpy.ndarray): fp32 output
        reduce_range (bool): use 7 bit or not
        backend (str): execution provider
    """
    import onnxruntime as ort

    input_data = quant_dequant_data(input_data, reduce_range, 2, "asym")
    sess = ort.InferenceSession(model.SerializeToString(), providers=[backend])
    preds = sess.run(None, {model.graph.input[0].name: input_data})
    loss = np.sum(np.abs(output_data - preds) ** 2)
    return loss


def make_sub_graph(node, inits, input_data, output_data, reduce_range, opset, ir_version):
    """Build a model with the specific node.

    Args:
        node (object): node
        inits (list): initializer inputs of this node
        input_data (numpy.ndarray): fp32 input
        output_data (numpy.ndarray): fp32 output
        reduce_range (bool): use 7 bit or not
        opset (object): opset of the model
        ir_version (object): ir_version of the model
    """
    from onnx import TensorProto, helper, numpy_helper

    input = helper.make_tensor_value_info(node.input[0], dtype_map[input_data.dtype], input_data.shape)
    output = helper.make_tensor_value_info(node.output[0], dtype_map[output_data.dtype], output_data.shape)
    graph = helper.make_graph([node], "sub_graph", [input], [output], inits)
    model = helper.make_model(graph, opset_imports=opset)
    model.ir_version = ir_version
    return model


def quant_dequant_data(data, reduce_range=False, qType=3, scheme="sym"):
    """Quantize and then dequantize data.

    Args:
        data (numpy.ndarray): target data
        reduce_range (bool): use 7 bit or not
        qType (int): data type
        scheme (str): sym or asym quantization
    """
    rmin, rmax, zero_point, scale, quantized_data = quantize_data(
        data.flatten().tolist(), _get_qrange_for_qType(qType, reduce_range), qType, scheme
    )
    return ((quantized_data - zero_point) * scale).astype(data.dtype).reshape(data.shape)


class ORTSmoothQuant:
    """Fake input channel quantization.

    For more details please refer to:
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    We only support inplace mode which means the model weights will be changed,
    you can call recover function to recover the weights if needed.
    """

    def __init__(self, model, dataloader, reduce_range=False, backend="CPUExecutionProvider"):
        """Initialize the attributes of class."""
        self.model = model if isinstance(model, BaseModel) else ONNXModel(model)
        self.value_infos = {vi.name: vi for vi in self.model.model.graph.value_info}
        self.value_infos.update({ot.name: ot for ot in self.model.model.graph.output})
        self.value_infos.update({it.name: it for it in self.model.model.graph.input})
        self.dataloader = dataloader
        self.reduce_range = reduce_range
        self.backend = backend
        self.tensor_scales_info = {}
        self.new_added_mul_nodes = []
        self.new_added_value_info = []
        self.new_init_tensors = []  # scales_tensor
        self.alpha = None
        self.percentile = None
        self.op_types = None
        self.scales_per_op = None
        self.calib_iter = None
        self.max_vals_per_channel = {}
        self.shape_info = None
        self.tensors_to_node = {}
        self.replace_input = []
        self.ops_to_absorb = []
        self.record_max_info = False
        self._build_absorb_function()

    def transform(
        self,
        alpha=0.5,
        folding=True,
        percentile=99.999,
        op_types=["Gemm", "Conv", "MatMul", "FusedConv"],
        scales_per_op=True,
        calib_iter=100,
        quantize_config=None,
        auto_alpha_args={"alpha_min": 0.3, "alpha_max": 0.7, "alpha_step": 0.05, "attn_method": "min"},
    ):
        """The main entry of smooth quant.

        Args:
            alpha (float or str): alpha value to balance the quantization difficulty of activation and weight.
            folding (bool): whether fold those foldable Mul which are inserted for smooth quant
            percentile (float): percentile of calibration to remove outliers
            op_types (list): the op type to be smooth quantized
            scales_per_op (bool): True, each op will have an individual scale, mainlyfor accuracy
                                  False, ops with the same input will share a scale, mainly for performance
            calib_iter (int): iteration num for calibration
            quantize_config (dict): quantize config

        Returns:
            A FP32 model with the same architecture as the orig model but with different weight which will be
            benefit to quantization
        """
        self.clean()
        if isinstance(alpha, float) and (alpha < 0 or alpha > 1):
            logger.warning("alpha should be a float value in [0, 1] or 'auto' ")
            if alpha < 0:
                alpha = 0
                logger.warning("reset alpha to 0 ")
            elif alpha > 1.0:
                alpha = 1.0
                logger.warning("reset alpha to 1.0 ")

        need_calibration = self._check_need_calibration(alpha, percentile, op_types, scales_per_op, calib_iter)
        if need_calibration:
            self._dump_op_info(percentile, op_types, calib_iter, quantize_config)

        if self.record_max_info:
            return self.model

        if alpha == "auto":
            alpha = self._auto_tune_alpha(calib_iter, **auto_alpha_args)

        scales = self._get_smooth_scales(alpha)
        self._insert_smooth_mul_op(scales)
        self._adjust_weights(scales)

        self.model.add_nodes(self.new_added_mul_nodes)
        self.model.model.graph.value_info.extend(self.new_added_value_info)
        self.model.add_initializers(self.new_init_tensors)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)

        self.model.update()
        if folding:
            self._fold_scale(scales)
        self.model.topological_sort()
        self.model.remove_unused_nodes()
        return self.model

    def recover(self):
        """Recover the model weights."""
        for tensor_name, nodes in self.tensors_to_node.items():
            for node_info in nodes:
                key = node_info[0] if self.scales_per_op else tensor_name
                if key not in self.tensor_scales_info:
                    continue
                input = node_info[1][1]
                weight = numpy_helper.to_array(
                    self.model.get_initializer(input),
                    base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
                )
                scale = self.tensor_scales_info[key]
                new_weight = weight * scale
                self.model.set_initializer(input, new_weight)

        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, new_input_name, old_input_name)

        for value_info in self.new_added_value_info:
            self.model.model.graph.value_info.remove(value_info)

        self.model.remove_nodes(self.new_added_mul_nodes)
        self.model.remove_initializers(self.new_init_tensors)
        self.tensor_scales_info = {}
        self.new_added_mul_nodes = []
        self.new_init_tensors = []
        self.new_added_value_info = []
        self.replace_input = []

    def clean(self):
        """Clean data collected from calibration."""
        self.tensor_scales_info = {}
        self.new_added_mul_nodes = []
        self.new_init_tensors = []
        self.new_added_value_info = []
        self.replace_input = []

    def _check_need_calibration(self, alpha, percentile, op_types, scales_per_op, calib_iter):
        """Check need calibration or not.

        Args:
            alpha (float or str): current alpha
            percentile (float): current percentile
            op_types (list): current op_types
            scales_per_op (bool): current scales_per_op
            calib_iter (int): current calib_iter
        """
        need_calib = True

        if (
            self.percentile == percentile
            and self.op_types == op_types
            and self.scales_per_op == scales_per_op
            and self.calib_iter == calib_iter
        ):
            need_calib = False

        self.alpha = alpha
        self.percentile = percentile
        self.op_types = op_types
        self.scales_per_op = scales_per_op
        self.calib_iter = calib_iter
        return need_calib

    def _build_absorb_function(self):
        """Build function mapping for scale folding."""
        from onnx import numpy_helper

        def norm(node, scale):  # pragma: no cover
            for idx in [1, 2]:
                tensor = self.model.get_initializer(node.input[idx])
                new_tensor = (
                    numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale
                    if self.model.model_path is not None
                    else numpy_helper.to_array(tensor) * scale
                )
                self.model.set_initializer(node.input[idx], new_tensor)
                self.tensor_scales_info[node.input[idx]] = (
                    1.0 / scale
                    if node.input[idx] not in self.tensor_scales_info
                    else self.tensor_scales_info[node.input[idx]] * 1.0 / scale
                )
            return True

        def mul(node, scale):  # pragma: no cover
            if all([self.model.get_initializer(inp) is None for inp in node.input]):
                return False
            for inp in node.input:
                if self.model.get_initializer(inp) is not None:
                    # Ensure that mul operators with shared initializer will not be absorbed.
                    if self.model.get_initializer_share_num(inp) > 1:
                        return False
                    key = node.input[0].split("_smooth_output")[0]
                    tensor = self.model.get_initializer(inp)
                    new_tensor = (
                        numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale
                        if self.model.model_path is not None
                        else numpy_helper.to_array(tensor) * scale
                    )
                    self.model.set_initializer(inp, new_tensor)
                    self.tensor_scales_info[key] = (
                        1.0 / scale
                        if key not in self.tensor_scales_info
                        else 1.0 / scale * self.tensor_scales_info[key]
                    )
            return True

        def conv(node, scale):  # pragma: no cover
            if len(node.input) > 2:
                if self.model.get_initializer(node.input[2]) is not None:
                    tensor = self.model.get_initializer(node.input[2])
                    new_tensor = (
                        numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale
                        if self.model.model_path is not None
                        else numpy_helper.to_array(tensor) * scale
                    )
                    self.model.set_initializer(node.input[2], new_tensor)
                    self.tensor_scales_info[node.input[2]] = 1.0 / scale
                scale = scale.reshape(-1, 1, 1, 1)
                tensor = self.model.get_initializer(node.input[1])
                new_tensor = (
                    numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale
                    if self.model.model_path is not None
                    else numpy_helper.to_array(tensor) * scale
                )
                self.model.set_initializer(node.input[1], new_tensor)
                self.tensor_scales_info[node.input[1]] = (
                    1.0 / scale
                    if node.input[1] not in self.tensor_scales_info
                    else self.tensor_scales_info[node.input[1]] * 1.0 / scale
                )
            return True

        self.could_absorb_optype = {
            "LayerNormalization": norm,
            "BatchNormalization": norm,
            "InstanceNormalization": norm,
            "SimplifiedLayerNormalization": mul,
            "MatMul": mul,
            "Gemm": mul,
            "Conv": conv,
            "FusedConv": conv,
            "Mul": mul,
        }

    def _fold_scale(self, scales):
        """Absorb the scale to the operator at output channel.

        Args:
            scales (dict): scales for smooth quant, {tensor_name: smooth quant scale}
        """
        remove_nodes = []
        for node in self.model.nodes():
            if node.op_type == "Mul" and node.name.endswith("_smooth_mul") and node not in remove_nodes:
                parent = self.model.get_parent(node, 0)
                if parent is None:
                    continue
                if parent.op_type in self.could_absorb_optype and len(self.model.get_children(parent)) == 1:
                    if node.output[0].split("_smooth_output")[0] in scales:
                        if self.could_absorb_optype[parent.op_type](
                            parent, 1.0 / scales[node.output[0].split("_smooth_output")[0]]
                        ):
                            remove_nodes.append(node)
                            children = [i for i in self.model.nodes() if node.output[0] in i.input]
                            for child in children:
                                for idx, inp in enumerate(child.input):
                                    if inp == node.output[0]:
                                        child.input[idx] = node.input[0]
        self.model.remove_nodes(remove_nodes)

    def _dump_op_info(self, percentile, op_types, iterations, quantize_config=None):
        """Dump op info for smooth quant.

        Args:
            percentile (float): percentile of calibration to remove outliers
            op_types (list): the op type to be smooth quantized
            iterations (int): iterations
            quantize_config (dict): quantize config
        """
        from neural_compressor.adaptor.ox_utils.calibration import ONNXRTAugment

        augment = ONNXRTAugment(
            self.model,
            self.dataloader,
            [],
            iterations=list(range(0, iterations)),
            backend=self.backend,
            reduce_range=self.reduce_range,
        )
        self.max_vals_per_channel, self.shape_info, self.tensors_to_node = augment.calib_smooth(
            percentile, op_types, None
        )
        for node in self.model.nodes():
            for out in node.output:
                if (
                    out in self.tensors_to_node
                    and node.op_type in self.could_absorb_optype
                    and self.model.get_initializer(node.input[1]) is not None
                ):
                    self.ops_to_absorb.append(node.name)

    def _get_output_loss(self, node_name, scale, calib_iter):
        """Get output loss of specific node after inserting QDQ pair.

        Args:
            node_name (str): node name
            scale (float): scale of the specific node
            calib_iter (int): iterations
        """
        import onnxruntime as ort
        from onnx import helper

        node = [i for i in self.model.nodes() if i.name == node_name]
        loss = 0
        if len(node) > 0:
            node = node[0]
            orig_outputs = self.model.output()
            added_tensors = [node.input[0], node.output[0]]
            self.model.add_tensors_to_outputs(added_tensors)

            session = (
                ort.InferenceSession(self.model.model_path + "_augment.onnx", providers=[self.backend])
                if self.model.is_large_model
                else ort.InferenceSession(self.model.model.SerializeToString(), providers=[self.backend])
            )
            base_dir = "" if not self.model.is_large_model else os.path.dirname(self.model.model_path)
            weight = onnx.numpy_helper.to_array(self.model.get_initializer(node.input[1]), base_dir)
            weight_q = quant_dequant_data(weight)

            self.model.set_initializer(node.input[1], weight_q)
            inits = [self.model.get_initializer(i) for i in node.input if self.model.get_initializer(i) is not None]

            inputs_names = [i.name for i in session.get_inputs()]
            model = None
            ort_inputs = {}
            for idx, (inputs, labels) in enumerate(self.dataloader):
                if idx + 1 > calib_iter:
                    break

                if len(inputs_names) == 1:
                    if isinstance(inputs, dict):  # pragma: no cover
                        for name, input in inputs.items():
                            ort_inputs.update({name: to_numpy(input)})
                    else:
                        ort_inputs.update({inputs_names[0]: to_numpy(inputs)})
                else:  # pragma: no cover
                    assert len(inputs_names) == len(inputs), "number of input tensors must align with graph inputs"

                    if isinstance(inputs, dict):
                        for name, input in inputs.items():
                            ort_inputs.update({name: to_numpy(input)})
                    else:
                        ort_inputs = dict(zip(inputs_names, [to_numpy(i) for i in inputs]))

                outputs = session.run(added_tensors, ort_inputs)
                if model is None:
                    model = make_sub_graph(
                        node,
                        inits,
                        outputs[0],
                        outputs[1],
                        self.reduce_range,
                        self.model.model.opset_import,
                        self.model.model.ir_version,
                    )
                loss += get_quant_dequant_output(model, outputs[0] * scale, outputs[1], self.reduce_range, self.backend)

            self.model.remove_tensors_from_outputs([i for i in added_tensors if i not in orig_outputs])
            self.model.set_initializer(node.input[1], weight)
        return loss

    def _reshape_scale_for_input(self, tensor, key):
        """Reshape the scale for input feature in channel.

        Args:
            tensor (str): tensor name
            key (str): scale key of this tensor
        """
        if len(self.shape_info[tensor]) == 4:
            scale = np.reshape(self.tensor_scales_info[key], (1, self.tensor_scales_info[key].shape[1], 1, 1))
        else:
            scale = np.reshape(self.tensor_scales_info[key], (1, self.tensor_scales_info[key].shape[0]))
        return scale

    def _auto_tune_alpha(self, calib_iter, alpha_min=0.3, alpha_max=0.7, alpha_step=0.05, attn_method="min"):
        """Perform alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly.

        Args:
            calib_iter (int): iterations
            alpha_min (float): min value of alpha search space.
            alpha_max (float): max value of alpha search space.
            alpha_step (float): step size of alpha search space.
            attn_method (str): criterion method used on attention ops; currently min, max and mean are supported.
        """
        logger.info("auto tuning alpha")
        import copy

        alpha_space = np.arange(alpha_min, alpha_max, alpha_step).tolist()

        optimal_alphas = {}
        if self.model.is_large_model:
            onnx.save_model(
                self.model.model,
                self.model.model_path + "_augment.onnx",
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.pb",
                convert_attribute=False,
            )

        ## Searching optimal alphas
        for tensor_name, node_infos in self.tensors_to_node.items():
            for node_info in node_infos:
                loss_alpha = {}
                key = node_info[0] if self.scales_per_op else tensor_name
                node = self.model.get_node(node_info[0])
                for alpha in alpha_space:
                    scale = self._get_smooth_scales(alpha, [key])
                    self._adjust_weights(scale)
                    input_scale = (
                        self._reshape_scale_for_input(tensor_name, key)
                        if not (node.op_type == "Gemm" and is_B_transposed(node))
                        else self.tensor_scales_info[key]
                    )
                    loss = self._get_output_loss(node_info[0], input_scale, calib_iter)
                    loss_alpha[alpha] = loss
                    if key not in optimal_alphas:  # Update alpha results
                        optimal_alphas[key] = alpha
                    else:
                        optimal_alphas[key] = (
                            alpha
                            if optimal_alphas[key] in loss_alpha and loss < loss_alpha[optimal_alphas[key]]
                            else optimal_alphas[key]
                        )
                    self.recover()
        logger.info("auto tuning alpha done")
        if self.model.is_large_model:
            from onnx.external_data_helper import load_external_data_for_model

            load_external_data_for_model(self.model.model, os.path.split(self.model.model_path)[0])
            os.remove(self.model.model_path + "_augment.onnx")
            os.remove(os.path.join(os.path.dirname(self.model.model_path), "weights.pb"))
        return optimal_alphas

    def _get_smooth_scales(self, alpha, target_list=[]):
        """Get the smooth scales for.

        The ops with the same input will share one mul layer.
        TODO support individual scales for each layer.

        Args:
            alpha: smooth alpha in paper
            target_list: target objects to get scale, [] means get all scales

        Returns:
            the smooth scales for weights, currently one input tensor only have one scale
        """
        logger.info("Start smooth scales collection.")
        scales = {}
        for tensor, nodes in self.tensors_to_node.items():
            # if scales_per_op the key of scales is the node name, otherwise the activation of node
            if self.scales_per_op:
                for node_info in nodes:
                    node = self.model.input_name_to_nodes[node_info[1][1]][0]
                    if len(target_list) > 0 and node_info[0] not in target_list:
                        continue
                    weight = numpy_helper.to_array(
                        self.model.get_initializer(node_info[1][1]),
                        base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
                    )
                    if (len(weight.shape) == 4 and weight.shape[1] != 1) or (
                        node.op_type == "Gemm" and is_B_transposed(node)
                    ):
                        weight = np.moveaxis(weight, 0, 1)
                    specific_alpha = alpha[node_info[0]] if isinstance(alpha, dict) else alpha
                    scales[node_info[0]] = self._get_smooth_scale(weight, specific_alpha, tensor)
            else:
                if len(target_list) > 0 and tensor not in target_list:
                    continue
                weights_in_channel_max = []
                for node_info in nodes:
                    node = self.model.input_name_to_nodes[node_info[1][1]][0]
                    weight = numpy_helper.to_array(
                        self.model.get_initializer(node_info[1][1]),
                        base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
                    )
                    if (len(weight.shape) == 4 and weight.shape[1] != 1) or (
                        node.op_type == "Gemm" and is_B_transposed(node)
                    ):
                        weight = np.moveaxis(weight, 0, 1)
                    weight = weight.reshape(weight.shape[0], -1)
                    cur_max = np.amax(weight, axis=-1)
                    weights_in_channel_max.append(cur_max)
                weights_stack = np.stack(weights_in_channel_max, axis=-1)
                specific_alpha = alpha[tensor] if isinstance(alpha, dict) else alpha
                scales[tensor] = self._get_smooth_scale(weights_stack, specific_alpha, tensor)

        return scales

    def _get_smooth_scale(self, weights, specific_alpha, tensor):
        """Get smooth scale for specific weight.

        Args:
            weights (numpy.ndarray): weight data
            specific_alpha (float): current alpha for this weights
            tensor (str): tensor name
        """
        weights = np.abs(weights.reshape(weights.shape[0], -1))
        weights_max = np.amax(weights, axis=-1)
        input_power = np.power(self.max_vals_per_channel[tensor], specific_alpha)
        weight_power = np.power(weights_max, 1 - specific_alpha)
        weight_power = np.clip(weight_power, a_min=1e-5, a_max=None)
        scale = np.clip(input_power / weight_power, a_min=1e-5, a_max=None)
        return scale

    def _insert_smooth_mul_op(self, scales):
        """Insert the Mul after inupt.

        The ops with the same input will share one mul layer.

        Args:
            scales (dict): The smooth scales
        """
        for key in scales.keys():
            input_name = key if not self.scales_per_op else self.model.get_node(key).input[0]
            weight_name = (
                self.tensors_to_node[key][0][1][1] if not self.scales_per_op else self.model.get_node(key).input[1]
            )
            scale_factor = 1.0 / scales[key]
            if (
                len(self.shape_info[weight_name]) == 3 or len(self.shape_info[weight_name]) == 2
            ):  # the last dim is input channel
                pass
            elif len(self.shape_info[weight_name]) == 4:
                scale_factor = np.reshape(scale_factor, (1, -1, 1, 1))
            else:
                assert False, "not support"
            name = key + "_" + "smooth_scale"
            scale_tensor = helper.make_tensor(
                name=key + "_" + "smooth_scale",
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=scale_factor.shape,
                vals=scale_factor.flatten().tolist(),
            )
            self.new_init_tensors.append(scale_tensor)
            mul_output_name = key + "_smooth_output"
            mul_node = helper.make_node(
                "Mul",
                inputs=[input_name, key + "_" + "smooth_scale"],
                outputs=[mul_output_name],
                name=key + "_smooth_mul",
            )
            self.new_added_mul_nodes.append(mul_node)
            if input_name in self.value_infos:
                value_info = copy.deepcopy(self.value_infos[input_name])
                value_info.name = mul_node.output[0]
                self.new_added_value_info.append(value_info)
            if self.scales_per_op:
                self.replace_input.append([self.model.get_node(key), input_name, mul_output_name])
            else:
                for node_info in self.tensors_to_node[key]:
                    self.replace_input.append([self.model.get_node(node_info[0]), key, mul_output_name])

    def _adjust_weights(self, scales):
        """Adjust the weights with scale.

        Args:
            scales (dict): The input scales
        """
        for idx, (tensor_name, nodes) in enumerate(self.tensors_to_node.items()):
            simple_progress_bar(len(self.tensors_to_node), idx + 1)
            for node_info in nodes:
                key = node_info[0] if self.scales_per_op else tensor_name
                if key not in scales:
                    continue
                input = node_info[1][1]
                node = self.model.input_name_to_nodes[input][0]
                weight = numpy_helper.to_array(
                    self.model.get_initializer(input),
                    base_dir=os.path.dirname(self.model.model_path) if self.model.model_path is not None else "",
                )
                if len(weight.shape) == 2:
                    scale = (
                        np.expand_dims(scales[key], axis=0)
                        if node.op_type == "Gemm" and is_B_transposed(node)
                        else np.expand_dims(scales[key], axis=-1)
                    )
                    new_weight = weight * scale
                elif len(weight.shape) == 4:  # TODO need to check conv
                    node = self.model.input_name_to_nodes[input][0]
                    if (
                        weight.shape[1] == 1
                        and "group" in [i.name for i in node.attribute]
                        and [i for i in node.attribute if i.name == "group"][0].i > 1
                    ):
                        scale = np.reshape(scales[key], (-1, 1, 1, 1))
                    else:
                        scale = np.reshape(scales[key], (1, -1, 1, 1))
                    new_weight = weight * scale
                else:
                    assert False, "not support"
                self.tensor_scales_info[key] = 1.0 / scale

                new_tensor = numpy_helper.from_array(new_weight, input)
                self.model.get_initializer(input).CopyFrom(new_tensor)
