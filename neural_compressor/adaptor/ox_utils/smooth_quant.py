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
#

import numpy as np
from neural_compressor.adaptor.ox_utils.util import find_by_name, quantize_data, _get_qrange_for_qType

dtype_map = {np.float32: 1,
             np.uint8: 2
             np.int8: 3,
             np.int32: 6,
             np.int64: 7,
             np.float16: 10,
             np.double: 11}
 
def get_quant_dequant_output(node, inits, input_data, output_data, reduce_range):
    import onnxruntime as ort
    from onnx import helper, TensorProto, numpy_helper
    inputs = []
    outputs = []
    for idx, inp in enumerate(node.input):
        inputs.append(helper.make_tensor_value_info(inp, dtype_map[inputs[idx].dtype, inputs[idx].shape))
    for idx, out in enumerate(node.output):
        outputs.append(helper.make_tensor_value_info(out, dtype_map[outputs[idx].dtype, outputs[idx].shape))
    for init in inits:
        q_dq_val = quant_dequant_data(numpy_helper.to_array(init), reduce_range)
        new_tensor = helper.make_tensor(
                            name=init.name,
                            data_type=dtype_map[numpy_helper.to_array(init).dtype],
                            dims=numpy_helper.to_array(init).shape if \
                                len(numpy_helper.to_array(init).shape) != 0 else [],
                            vals=q_dq_val if \
                                len(numpy_helper.to_array(init)) != 0 else [numpy_helper.to_array(init)])
        init.CopyFrom(new_tensor)
    for idx, data in enumerate(input_data):
        input_data[idx] = quant_dequant_data(data, reduce_range, 2, 'asym')
    graph = helper.make_graph([node], 'sub_graph', inputs, outputs, inits)
    model = helper.make_model(graph)
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    preds = sess.run([i.name for i in outputs], dict(zip(node.input, input_data)))
    loss = np.sum(np.abs(output_data - preds) ** 2)
    return preds

def quant_dequant_data(data, reduce_range=False, qType=3, scheme='sym'):
    rmin, rmax, zero_point, scale, quantized_data = quantize_data(
        weight.flatten().tolist(), _get_qrange_for_qType(qType, reduce_range), qType, scheme)
    return (quantized_data - zero_point) * scale

class ORTSmoothQuant:
    """
    Fake input channel quantization, for more details please refer to
    [1] SmoothQuant: Accurate and Efficient
    Post-Training Quantization for Large Language Models
    [2] SPIQ: Data-Free Per-Channel Static Input Quantization
    We only support inplace mode which means the model weights will be changed, you can call recover function
    to recover the weights if needed
    """
    def __init__(self, model, dataloader, q_func=None):
        self.mdoel = model
        self.dataloader = dataloader
        self.q_func = q_func
        self.tensor_scales_info = {}
        self.new_added_mul_nodes = []
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
        self.could_absorb_optype = ["LayerNormalization", "BatchNormalization", "InstanceNormalization",
                                    "SimplifiedLayerNormalization", "MatMul", "Gemm", "Conv", "FusedConv", "Mul"]
        
    def transform(self, alpha=0.5, folding=False, percentile=99.999, op_types=['Gemm', 'Conv', 'MatMul', 'FusedConv'],
                  scales_per_op=False, calib_iter=100,
                  auto_alpha_args={'alpha_min': 0.3, 'alpha_max': 0.7, 'alpha_step': 0.05, 'attn_method': 'min'}):
        """
        The main entry of smooth quant
        :param alpha: Alpha value to balance the quantization difficulty of activation and weight, please refer
        to the paper for more details
        :param folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
        :param percentile: Not supported now
        :param op_types: The op typed to be smooth quantized
        :param scales_per_op: Not supported now
        :param calib_iter: Data size for calibration
        :return: A FP32 model with the same architecture as the orig model but with different weight which will be
        benefit to quantization
        """
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

            if alpha == 'auto':
                alpha = self._auto_tune_alpha(input_maxes_abs, **auto_alpha_args)

            scales = self._get_smooth_scales(alpha)
            self._insert_smooth_mul_op(scales)
            self._adjust_weights(scales)

        self.model.add_nodes(self.new_added_mul_nodes)
        self.model.add_initializers(self.new_init_tensors)
        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, old_input_name, new_input_name)

        self.model.update()
        self._fold_scale(scales)
        self.model.topological_sort()
        self.model.remove_unused_constant()
        return self.model

    def recover(self):
        """
        recover the model weights
        :return:
        """
        for tensor_name, scale in self.tensor_scales_info.items():
            tensor = self.model.get_initializer(tensor_name)
            new_tensor = numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale if \
                self.model.model_path is not None else numpy_helper.to_array(tensor) * scale
            self.model.set_initializer(tensor_name, new_tensor)

        for node, old_input_name, new_input_name in self.replace_input:
            self.model.replace_node_input(node, new_input_name, old_input_name)

        self.model.remove_nodes(self.new_added_mul_nodes)
        self.model.remove_initializers(self.new_init_tensors)
        self.tensor_scales_info = {}

    def _check_need_calibration(self, alpha, percentile, op_types, scales_per_op, calib_iter):
        """
        check need calibration or not
        :param alpha: current alpha
        :param percentile: current percentile
        :param op_types: current op_types
        :param scales_per_op: current scales_per_op
        :param calib_iter:: current scales_per_op
        :return:
        """
        need_calib = True

        if self.percentile == percentile and self.op_types == op_types \
                and self.scales_per_op == scales_per_op and self.calib_iter == calib_iter:
            need_calib = False

        self.alpha = alpha
        self.percentile = percentile
        self.op_types = op_types
        self.scales_per_op = scales_per_op
        self.calib_iter = calib_iter
        return need_calib

    def _fold_scale(self, scales):
        """Absorb the scale to the operator at output channel.
        Args:
            scales: A dict, tensor: smooth quant scale
        """
        from onnx import numpy_helper
        def norm(node, scale): # pragma: no cover
            for idx in [1, 2]:
                tensor = self.model.get_initializer(node.input[idx])
                new_tensor = numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale if \
                    self.model.model_path is not None else numpy_helper.to_array(tensor) * scale
                self.model.set_initializer(node.input[idx], new_tensor)
                self.tensor_scales_info[node.input[idx]] = 1. / scale
            return True
    
        def mul(node, scale): # pragma: no cover
            if all([self.model.get_initializer(inp) is None for inp in node.input]):
                return False
            for inp in node.input:
                if self.model.get_initializer(inp) is not None:
                    tensor = self.model.get_initializer(inp)
                    new_tensor = numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale if \
                        self.model.model_path is not None else numpy_helper.to_array(tensor) * scale
                    self.model.set_initializer(inp, new_tensor)
                    self.tensor_scales_info[inp] = 1. / scale
            return True
    
        def conv(node, scale): # pragma: no cover
            if len(node.input) > 2:
                if self.model.get_initializer(node.input[2]) is not None:
                    tensor = self.model.get_initializer(node.input[2])
                    new_tensor = numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale if \
                        self.model.model_path is not None else numpy_helper.to_array(tensor) * scale
                    self.model.set_initializer(node.input[2], new_tensor)
                    self.tensor_scales_info[node.input[2]] = 1. / scale
                scale = scale.reshape(-1, 1, 1, 1)
                tensor = self.model.get_initializer(node.input[1])
                new_tensor = numpy_helper.to_array(tensor, os.path.dirname(self.model.model_path)) * scale if \
                    self.model.model_path is not None else numpy_helper.to_array(tensor) * scale
                self.model.set_initializer(node.input[1], new_tensor)
                self.tensor_scales_info[node.input[1]] = 1. / scale
            return True
    
        could_absorb_optype = {"LayerNormalization": norm,
                               "BatchNormalization": norm,
                               "InstanceNormalization": norm,
                               "SimplifiedLayerNormalization": mul,
                               "MatMul": mul, 
                               "Gemm": mul,
                               "Conv": conv,
                               "FusedConv": conv,
                               "Mul": mul
                               }
        remove_nodes = []
    
        scales_per_op = self.model.get_initializer(list(scales.keys())[0]) is None
    
        for node in self.model.nodes():
            if node.op_type == "Mul"  and node.name.endswith("_smooth_mul"):
                parent = self.model.get_parent(node, 0)
                if parent is None:
                    continue
                if parent.op_type in could_absorb_optype and len(self.model.get_children(parent)) == 1:
                    if node.output[0].split("_smooth_output")[0] in scales:
                        if could_absorb_optype[parent.op_type](parent,
                                1.0 / scales[node.output[0].split("_smooth_output")[0]]):
                            remove_nodes.append(node)
                            children = [i for i in self.model.nodes() if node.output[0] in i.input]
                            for child in children:
                                for idx, inp in enumerate(child.input):
                                    if inp == node.output[0]:
                                        child.input[idx] = node.input[0]
        self.model.remove_nodes(remove_nodes)

    def _dump_op_info(self, percentile, op_types, iterations, quantize_config=None):
        from neural_compressor.adaptor.ox_utils.calibration import ONNXRTAugment
        if quantize_config is not None:
            black_nodes = [node for node in quantize_config if quantize_config[node] == 'fp32']
            white_nodes = [node for node in quantize_config if quantize_config[node] != 'fp32']
        augment = ONNXRTAugment(self.model,
                                self.dataloader,
                                quantizable_op_types,
                                black_nodes=black_nodes,
                                white_nodes=white_nodes,
                                iterations=list(range(0, iterations)),
                                backend=self.backend,
                                reduce_range=self.reduce_range)
        self.max_vals_per_channel, self.shape_infos, self.tensors_to_node = \
                                            augment.calib_smooth(op_types, quantize_config)
        for node in self.model.nodes():
            for out in node.output:
                if out in self.tensors_to_node and node.op_type in self.could_absorb_optype and \
                    self.model.get_initializer(node.input[1]) is not None :
                     self.ops_to_absorb.append(node.name)

    def _get_output_loss(self, node_name, scale):
        from onnx import helper
        import onnxruntime as ort
        node = [i for i in self.model.nodes() if i.name == node_name]
        weights = []
        if len(node) > 0:
            if node[0].op_type in ['Conv', 'FusedConv']:
                weight = onnx.numpy_helper.to_array(self.model.get_initializer(node.input[1]), base_dir)
                weight_q = quant_dequant_w(weight)
            elif node[0].op_type in ['MatMul', 'Gemm']:
                weight = onnx.numpy_helper.to_array(self.model.get_initializer(node.input[1]), base_dir)
                weight_q = quant_dequant_w(weight)

        added_tensors = [node.input[1], node.output[0]]
        self.model.add_tensors_to_outputs(added_tensors)

        session = ort.InferenceSession(self.model.model_path  + '_augment.onnx',
                                       providers=['CPUExecutionProvider']) if \
                                       self.model.is_large_model else \
                  ort.InferenceSession(self.model.model.SerializeToString(),
                                       providers=['CPUExecutionProvider'])

        inputs_names = [i.name for i in session.get_inputs()]
        for idx, (inputs, labels) in enumerate(self.dataloader):
            if isinstance(inputs, dict):
                ort_inputs = inputs
            elif len(inputs_names) == 1:
                ort_inputs = {inputs_names[0]: inputs})
            else:
                ort_inputs = dict(zip(inputs_names, [np.array(i) for i in inputs]))
            outputs = session.run(added_tensors, ort_inputs)
            break

        self.model.remove_tensors_from_outputs(added_tensors)
        loss = get_quant_dequant_output(node, inits, outputs[0], outputs[1], self.reduce_range)
        return loss

    def _reshape_scale_for_input(self, tensor):
        """
        reshape the scale for input feature in channel
        :param layer:
        :param scale:
        :return:
        """
        if len(self.shape_info[tensor]) == 4:
            scale = np.reshape(self.tensor_scales_info[tensor], (1, self.tensor_scales_info[tensor].shape[0], 1, 1))
        else:
            scale = np.reshape(self.tensor_scales_info[tensor], (1, self.tensor_scales_info[tensor].shape[0]))
        return scale

    def _auto_tune_alpha(self, input_maxes, alpha_min=0.3, alpha_max=0.7, alpha_step=0.05, attn_method='min'):
        """
        Perform alpha-tuning to obtain layer-wise optimal alpha values and adjust parameters accordingly.
        input_maxes:
        alpha_min: min value of alpha search space.
        alpha_max: max value of alpha search space.
        alpha_step: step size of alpha search space.
        attn_method: criterion method used on attention ops; currently min, max and mean are supported.
        """
        logger.info("auto tuning alpha")
        import copy
        alpha_scale = 100
        alpha_space = list(range(round(alpha_min * alpha_scale), round((alpha_max + alpha_step) * alpha_scale),
                                 round(alpha_step * alpha_scale)))
        alpha_space = [alpha / alpha_scale for alpha in alpha_space]

        optimal_alphas = {}
        if self.model.is_large_model:
            onnx.save_model(self.model.model,
                            self.model.model_path + '_augment.onnx',
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location="weights.pb",
                            convert_attribute=False)

        ## Searching optimal alphas
        for tensor_name, node_infos in self.tensors_to_node.items():
            loss_all_ops = {}
            for node_info in node_infos:
                loss_alpha = {}
                key = node_info[0] if self.scales_per_op else tensor_name

                for alpha in alpha_space:
                    scale = self._get_smooth_scales(alpha, [key])
                    self._adjust_weights(scale)
                    loss = self._get_output_loss(node_info[0], input_scale)
                    loss_alpha[alpha] = loss
                    self.recover()
                    if key not in optimal_alphas:  # Update alpha results
                        optimal_alphas[key] = alpha
                    else:
                        optimal_alphas[key] = alpha if loss < loss_alpha[optimal_alphas[key]] \
                                                                          else optimal_alphas[key]
                loss_all_ops[key] = loss_alpha
        logger.info("auto tuning alpha done")
        if self.model.is_large_model:
            from onnx.external_data_helper import load_external_data_for_model
            load_external_data_for_model(self.model.model, os.path.split(model.model_path)[0])
            os.remove(self.model.model_path + '_augment.onnx')
            os.remove(os.path.join(os.path.dirname(self.model.model_path, "weights.pb")
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
        scales = {}
        for tensor, nodes in self.tensors_to_node.items():
            if self.scales_per_op:
                for node_info in nodes:
                    if len(target_list) > 0 and node_info[0] not in target_list:
                        continue
                    weight = numpy_helper.to_array(self.model.get_initializer(node_info[1][1]))
                    if len(weight.shape) == 4:  # conv
                        if weight.shape[1] == 1:  # depthwise conv
                            pass
                        else:
                            weight = np.moveaxis(weight, 0, 1)
                    specific_alpha = alpha[node_info[0]] if isinstance(alpha, dict) else alpha
                    scales[node_info[0]] = self._get_smooth_scale(weigths_stack, specific_alpha)
            else:
                if len(target_list) > 0 and tensor not in target_list:
                    continue
                weights = [numpy_helper.to_array(self.model.get_initializer(node_info[1][1])) for \
                    node_info in nodes]
                weights_in_channel_max = []
                for weight in weights:  # mamul ic*oc, conv oc*ic*k*k
                    if len(weight.shape) == 4:  # conv
                        if weight.shape[1] == 1:  # depthwise conv
                            pass
                        else:
                            weight = np.moveaxis(weight, 0, 1)
                    weight = weight.reshape(weight.shape[0], -1)
                    cur_max = np.amax(weight, axis=-1)
                    weights_in_channel_max.append(cur_max)
                weigths_stack = np.stack(weights_in_channel_max, axis=-1)
                specific_alpha = alpha[tensor] if isinstance(alpha, dict) else alpha
                scales[tensor] = self._get_smooth_scale(weigths_stack, specific_alpha)
 
        return scales
    
    def _get_smooth_scale(self, weight, specific_alpha):
        weigths = np.abs(weigths.reshape(weigths.shape[0], -1))
        weights_max = np.amax(weigths, axis=-1)
        input_power = np.power(self.max_vals_per_channel[tensor], specific_alpha)
        weight_power = np.power(weights_max, 1 - specific_alpha)
        scale = np.clip(input_power / weight_power, a_min=1e-5, a_max=None)
        return scale

    def _insert_smooth_mul_op(self, scales):
        """Insert the mul layer after inupt.
    
        The ops with the same input will share one mul layer.
    
        Args:
            scales: The smooth scales
        """
        for key in scales.keys():
            scale_factor = 1.0 / scales[key]
            if len(self.shape_info[key]) == 3 or len(self.shape_info[key]) == 2:  # the last dim is input channel
                pass
            elif len(self.shape_info[key]) == 4:
                scale_factor = np.reshape(scale_factor, (1, -1, 1, 1))
            else:
                assert False, "not support"
            name = key + "_" + "smooth_scale"
            scale_tensor = helper.make_tensor(
                name=name,
                data_type=onnx_proto.TensorProto.FLOAT,
                dims=scale_factor.shape,
                vals=scale_factor.flatten().tolist())
            self.new_init_tensors.append(scale_tensor)
            mul_output_name = key + "_smooth_output"
            mul_node = helper.make_node(
                "Mul",
                inputs=[key, key + "_" + "smooth_scale"],
                outputs=[mul_output_name],
                name=key + "_smooth_mul"
            )
            self.new_added_mul_nodes.append(mul_node)
            if self.scales_per_op:
                self.replace_input.append([find_by_name(key, self.model.nodes()), key, mul_output_name])
            else:
                for node_info in self.tensors_to_node[key]:
                    self.replace_input.append([find_by_name(node_info[0], self.model.nodes()),
                                                             node_info[1][0], mul_output_name])
    
    def _adjust_weights(self, scales):
        """Adjust the weights per input scale.
    
        Each op has one individual Mul layer.
    
        Args:
            scales: The input scales
        """
        for tensor_name, nodes in self.tensors_to_node.items():
            for node_info in nodes:
                key = node_info[0] if self.scales_per_op else tensor_name
                if key not in scales:
                    continue
                input = node_info[1][1]
                weight = numpy_helper.to_array(self.model.get_initializer(input))
                if len(weight.shape) == 2:
                    scale = np.expand_dims(scales[key],
                                           axis=-1)  # TODO, to support conv
                    new_weight = weight * scale
                elif len(weight.shape) == 4:  # TODO need to check conv
                    scale = np.reshape(scales[key], (1, -1, 1, 1))
                    new_weight = weight * scale
                else:
                    assert False, "not support"
                self.tensor_scales_info[key] = 1. / scale
                self.model.set_initializer(input, new_weight)
