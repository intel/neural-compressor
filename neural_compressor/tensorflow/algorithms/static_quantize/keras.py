#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import json
import math
import os
from collections import OrderedDict, UserDict

import numpy as np
import tensorflow as tf
import yaml

from neural_compressor.common.logger import Logger
from neural_compressor.tensorflow.utils import deep_get, dump_elapsed_time

logger = Logger().get_logger()


def _add_supported_quantized_objects(custom_objects):
    """Map all the quantized objects."""
    from .keras_utils.conv2d import QConv2D
    from .keras_utils.dense import QDense
    from .keras_utils.depthwise_conv2d import QDepthwiseConv2D
    from .keras_utils.pool2d import QAvgPool2D, QMaxPool2D
    from .keras_utils.quantizer import DeQuantize, FakeQuant, Quantize
    from .keras_utils.separable_conv2d import QSeparableConv2D

    custom_objects["Quantize"] = Quantize
    custom_objects["DeQuantize"] = DeQuantize
    custom_objects["FakeQuant"] = FakeQuant
    custom_objects["QConv2D"] = QConv2D
    custom_objects["QDepthwiseConv2D"] = QDepthwiseConv2D
    custom_objects["QSeparableConv2D"] = QSeparableConv2D
    custom_objects["QDense"] = QDense
    custom_objects["QMaxPool2D"] = QMaxPool2D
    custom_objects["QAvgPool2D"] = QAvgPool2D
    custom_objects["QMaxPooling2D"] = QMaxPool2D
    custom_objects["QAveragePooling2D"] = QAvgPool2D
    return custom_objects


class KerasAdaptor:
    """The keras class of framework adaptor layer."""

    def __init__(self, framework_specific_info):
        self.framework_specific_info = framework_specific_info
        self.approach = deep_get(self.framework_specific_info, "approach", False)
        self.quantize_config = {"op_wise_config": {}}
        self.device = self.framework_specific_info["device"]
        self.backend = self.framework_specific_info["backend"]
        self.recipes = deep_get(self.framework_specific_info, "recipes", {})
        self.supported_op = [
            "Conv2D",
            "Dense",
            "SeparableConv2D",
            "DepthwiseConv2D",
            "AveragePooling2D",
            "MaxPooling2D",
            "AvgPool2D",
            "MaxPool2D",
        ]

        self.pre_optimized_object = None
        self.pre_optimizer_handle = None
        self.bf16_ops = []
        self.fp32_ops = []
        self.query_handler = KerasQuery(local_config_file=os.path.join(os.path.dirname(__file__), "keras.yaml"))

        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.callbacks = []

        self.conv_format = {}

    def _check_itex(self):
        """Check if the Intel® Extension for TensorFlow has been installed."""
        try:
            import intel_extension_for_tensorflow
        except:
            raise ImportError(
                "The Intel® Extension for TensorFlow is not installed. "
                "Please install it to run models on ITEX backend"
            )

    def tuning_cfg_to_fw(self, tuning_cfg):
        """Parse tune_config and set framework variables."""
        self.quantize_config["calib_iteration"] = tuning_cfg["calib_iteration"]
        self.quantize_config["device"] = self.device
        self.quantize_config["advance"] = deep_get(tuning_cfg, "advance")
        fp32_ops = []
        bf16_ops = []
        bf16_type = set(self.query_handler.get_op_types_by_precision(precision="bf16"))
        dispatched_op_names = [j[0] for j in tuning_cfg["op"]]
        invalid_op_names = [i for i in self.quantize_config["op_wise_config"] if i not in dispatched_op_names]

        for op_name in invalid_op_names:
            self.quantize_config["op_wise_config"].pop(op_name)

        for each_op_info in tuning_cfg["op"]:
            op_name = each_op_info[0]

            if tuning_cfg["op"][each_op_info]["activation"]["dtype"] == "bf16":
                if each_op_info[1] in bf16_type:
                    bf16_ops.append(op_name)
                continue

            if tuning_cfg["op"][each_op_info]["activation"]["dtype"] == "fp32":
                if op_name in self.quantize_config["op_wise_config"]:
                    self.quantize_config["op_wise_config"].pop(op_name)
                    fp32_ops.append(op_name)
                continue

            is_perchannel = False
            bit = None
            if "weight" in tuning_cfg["op"][each_op_info]:
                is_perchannel = tuning_cfg["op"][each_op_info]["weight"]["granularity"] == "per_channel"
                # bit = tuning_cfg['op'][each_op_info]['weight']['bit']
            weight_bit = bit if bit else 7.0
            algorithm = tuning_cfg["op"][each_op_info]["activation"]["algorithm"]
            is_asymmetric = False
            if "activation" in tuning_cfg["op"][each_op_info]:
                is_asymmetric = tuning_cfg["op"][each_op_info]["activation"]["scheme"] == "asym"
            self.quantize_config["op_wise_config"][op_name] = (is_perchannel, algorithm, is_asymmetric, weight_bit)
        self.bf16_ops = bf16_ops
        if self.bf16_ops:
            self.bf16_ops.pop(-1)
        self.fp32_ops = fp32_ops

    def _pre_optimize(self, model):
        """Apply pre-optimization."""
        model = self._check_quantize_format(model)
        model = self._fuse_bn(model)
        return model

    def _check_quantize_format(self, model):
        """The function that checks format for conv ops."""
        json_model = copy.deepcopy(json.loads(model.to_json()))
        config = json_model["config"]
        fp32_layers = config["layers"]
        name_op_map = {}

        for idx, layer in enumerate(copy.deepcopy(fp32_layers)):
            name_op_map[layer["config"]["name"]] = layer

        for idx, layer in enumerate(copy.deepcopy(fp32_layers)):
            layer_config = layer["config"]
            if layer["class_name"] in self.supported_op:
                if "inbound_nodes" in layer:
                    check_layer = name_op_map[layer["inbound_nodes"][0][0][0]]
                else:
                    check_layer = fp32_layers[idx - 1]
                if check_layer["class_name"] in ["Activation"] and check_layer["config"]["activation"] in ["relu"]:
                    self.conv_format[layer["config"]["name"]] = "u8"
                else:
                    self.conv_format[layer["config"]["name"]] = "s8"
        return model

    def _fuse_bn(self, model):
        """Fusing Batch Normalization."""
        json_model = copy.deepcopy(json.loads(model.to_json()))
        config = json_model["config"]
        fp32_layers = config["layers"]

        def fuse_conv_bn(conv_weight, bn_weight, conv_type="Conv2D", eps=1.0e-5):
            assert conv_type in [
                "Conv2D",
                "DepthwiseConv2D",
                "SeparableConv2D",
            ], "only support Conv2D, DepthwiseConv2D, SeparableConv2D..."
            if len(bn_weight) > 3:
                if conv_type == "DepthwiseConv2D":
                    gamma = bn_weight[0].reshape(1, 1, bn_weight[0].shape[0], 1)
                    var = bn_weight[3].reshape(1, 1, bn_weight[3].shape[0], 1)
                else:
                    gamma = bn_weight[0].reshape(1, 1, 1, bn_weight[0].shape[0])
                    var = bn_weight[3].reshape(1, 1, 1, bn_weight[3].shape[0])
                beta = bn_weight[1]
                mean = bn_weight[2]
            else:
                gamma = 1.0
                beta = bn_weight[0]
                mean = bn_weight[1]
                if conv_type == "DepthwiseConv2D":
                    var = bn_weight[2].reshape(1, 1, bn_weight[2].shape[0], 1)
                else:
                    var = bn_weight[2].reshape(1, 1, 1, bn_weight[2].shape[0])

            if len(conv_weight) == 1:
                weight = conv_weight[0]
                bias = np.zeros_like(beta)
            elif len(conv_weight) == 2 and conv_type == "SeparableConv2D":
                depth_weight = conv_weight[0]
                weight = conv_weight[1]
                bias = np.zeros_like(beta)
            elif len(conv_weight) == 2 and conv_type != "SeparableConv2D":
                weight = conv_weight[0]
                bias = conv_weight[1]
            elif len(conv_weight) == 3:
                depth_weight = conv_weight[0]
                weight = conv_weight[1]
                bias = conv_weight[2]
            scale_value = gamma / np.sqrt(var + eps)
            weight = weight * scale_value
            bias = beta + (bias - mean) * scale_value.reshape(-1)
            bias = bias.reshape(-1)
            return [depth_weight, weight, bias] if conv_type == "SeparableConv2D" else [weight, bias]

        node_map = {}
        for idx, layer in enumerate(copy.deepcopy(fp32_layers)):
            layer_config = layer["config"]
            if "inbound_nodes" in layer:
                node_map[layer["name"]] = layer

        fuse_layers = []
        fold_conv = []
        for idx, layer in enumerate(copy.deepcopy(fp32_layers)):
            layer_config = layer["config"]
            if "inbound_nodes" in layer:
                if layer["class_name"] in ["BatchNormalization"]:
                    bn_inbound_node = node_map[layer_config["name"]]["inbound_nodes"][0][0]
                    if bn_inbound_node[0] in self.conv_weights.keys():
                        conv_weight = self.conv_weights[bn_inbound_node[0]]
                        conv_layer = node_map[bn_inbound_node[0]]
                        bn_weight = self.bn_weights[layer_config["name"]]
                        self.layer_weights[bn_inbound_node[0]] = fuse_conv_bn(
                            conv_weight, bn_weight, conv_layer["class_name"], layer["config"]["epsilon"]
                        )
                        fold_conv.append(bn_inbound_node[0])
                    else:
                        fuse_layers.append(layer)
                elif len(layer["inbound_nodes"]):
                    new_bound_nodes = []
                    # OpLambda node will have different bound node
                    if layer["class_name"] in ["TFOpLambda", "SlicingOpLambda"]:
                        fuse_layers.append(layer)
                    else:
                        for bound_node in layer["inbound_nodes"][0]:
                            if bound_node[0] in self.bn_weights.keys():
                                bn_inbound_node = node_map[bound_node[0]]["inbound_nodes"][0][0]
                                if bn_inbound_node[0] in self.conv_weights.keys():
                                    new_bound_nodes.append(bn_inbound_node)
                                else:
                                    new_bound_nodes.append(bound_node)
                            else:
                                new_bound_nodes.append(bound_node)
                        layer["inbound_nodes"] = [new_bound_nodes]
                        fuse_layers.append(layer)
                else:
                    fuse_layers.append(layer)
            else:
                if (
                    idx > 0
                    and layer["class_name"] in ["BatchNormalization"]
                    and fp32_layers[idx - 1]["class_name"] in ["Conv2D"]
                ):
                    conv_name = fp32_layers[idx - 1]["config"]["name"]
                    conv_weight = self.conv_weights[conv_name]
                    bn_weight = self.bn_weights[layer_config["name"]]
                    conv_type = fp32_layers[idx - 1]["class_name"]
                    self.layer_weights[conv_name] = fuse_conv_bn(
                        conv_weight, bn_weight, conv_type, layer["config"]["epsilon"]
                    )
                    fold_conv.append(conv_name)
                else:
                    fuse_layers.append(layer)

        # bn folding will have a shift bias
        for idx, layer in enumerate(fuse_layers):
            layer_config = layer["config"]
            if (
                layer["class_name"] in ["Conv2D", "DepthwiseConv2D", "SeparableConv2D"]
                and layer_config["name"] in fold_conv
            ):
                layer_config["use_bias"] = True

        json_model["config"]["layers"] = fuse_layers
        fused_model = self._restore_model_from_json(json_model)
        return fused_model

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, dataloader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg(dict): The chosen tuning configuration.
            model (object): The model to do quantization.
            dataloader(object): The dataloader used to load quantization dataset.
            q_func (optional): training function for quantization aware training mode.
        """
        self.tuning_cfg_to_fw(tune_cfg)
        # just convert the input model to mixed_bfloat16
        if self.bf16_ops and not self.quantize_config["op_wise_config"]:
            converted_model = self.convert_bf16()
            return converted_model

        # if self.backend == "itex":
        #     self._check_itex()
        logger.debug("Dump quantization configurations:")
        logger.debug(self.quantize_config)
        calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)

        if hasattr(dataloader, "batch_size") and calib_sampling_size % dataloader.batch_size != 0:
            iter = self.quantize_config["calib_iteration"]
            logger.warning(
                "Please note that calibration sampling size {} "
                "isn't divisible exactly by batch size {}. "
                "So the real sampling size is {}.".format(
                    calib_sampling_size, dataloader.batch_size, dataloader.batch_size * iter
                )
            )

        q_layers = []
        self.inbound_nodes_map = {}
        for idx, layer in enumerate(copy.deepcopy(self.fp32_layers)):
            layer_config = layer["config"]
            if (
                layer["class_name"] in self.supported_op
                and layer["config"]["name"] in self.quantize_config["op_wise_config"]
            ):
                op_config = self.quantize_config["op_wise_config"][layer["config"]["name"]]
                mode = "per_channel" if op_config[0] else "per_tensor"
                fake_q_name = "fake_quant_" + str(idx)
                fake_q_layer = {
                    "class_name": "FakeQuant",
                    "name": fake_q_name,
                    "T": self.conv_format[layer["config"]["name"]],
                    "config": {"mode": "per_tensor", "name": fake_q_name},
                }
                if "inbound_nodes" in layer:
                    fake_q_layer["inbound_nodes"] = layer["inbound_nodes"]
                    layer["inbound_nodes"] = [[[fake_q_name, 0, 0, {}]]]
                    self.inbound_nodes_map[fake_q_name] = layer

                q_layers.append(fake_q_layer)
                q_layers.append(layer)
            else:
                q_layers.append(layer)

        json_model = copy.deepcopy(json.loads(self.pre_optimized_object.to_json()))
        json_model["config"]["layers"] = q_layers
        quantized_model = self._restore_model_from_json(json_model)

        converted_model = self._calibrate(quantized_model, dataloader, self.quantize_config["calib_iteration"])

        return converted_model

    def _calibrate(self, model, dataloader, calib_interation):
        """Apply calibration."""
        # run eagerly to fetch the numpy min/max
        model.compile(run_eagerly=True)
        results = {}
        for idx, (inputs, labels) in enumerate(dataloader):
            outputs = model.predict_on_batch(inputs)
            json_model = copy.deepcopy(json.loads(model.to_json()))
            config = json_model["config"]
            layers = config["layers"]
            for layer in layers:
                if layer["class_name"] == "FakeQuant":
                    min_value = layer["config"]["min_value"]
                    max_value = layer["config"]["max_value"]
                    if layer["config"]["name"] not in results:
                        results[layer["config"]["name"]] = {"min": [min_value], "max": [max_value]}
                    else:
                        results[layer["config"]["name"]]["min"].append(min_value)
                        results[layer["config"]["name"]]["max"].append(max_value)
            if idx + 1 == calib_interation:
                break

        # insert the calibrated min/max to Q/DQ
        json_model = copy.deepcopy(json.loads(model.to_json()))
        config = json_model["config"]
        layers = config["layers"]
        q_layers = []
        # quantize_mode = self._check_quantize_mode(json_model)
        inbound_reverse_map = {}
        for idx, layer in enumerate(layers):
            layer_config = copy.deepcopy(layer["config"])
            if layer["class_name"] == "FakeQuant":
                min_value = min(results[layer["config"]["name"]]["min"])
                max_value = max(results[layer["config"]["name"]]["max"])
                quantize_layer = {
                    "class_name": "Quantize",
                    "name": "quantize_" + str(idx),
                    "config": {
                        "min_range": min_value,
                        "max_range": max_value,
                        "T": layer_config["T"],
                        "name": "quantize_" + str(idx),
                    },
                }
                dequantize_layer = {
                    "class_name": "DeQuantize",
                    "name": "dequantize_" + str(idx),
                    "config": {
                        "min_range": min_value,
                        "max_range": max_value,
                        # 'mode': quantize_mode,
                        "name": "dequantize_" + str(idx),
                    },
                }
                if "inbound_nodes" in layer:
                    quantize_layer["inbound_nodes"] = layer["inbound_nodes"]
                    dequantize_layer["inbound_nodes"] = [[["quantize_" + str(idx), 0, 0, {}]]]
                    # find the conv/dense layer from fake quant map and
                    # change the conv/dense node inbound to dequantize
                    layer_name = self.inbound_nodes_map[layer["name"]]["name"]
                    inbound_reverse_map[layer_name] = [[["dequantize_" + str(idx), 0, 0, {}]]]

                q_layers.append(quantize_layer)
                q_layers.append(dequantize_layer)
            elif (
                layer["class_name"] in self.supported_op
                and layer["config"]["name"] in self.quantize_config["op_wise_config"]
            ):
                # index 0 is weight, index 1 is bias
                q_layer_name = "Q" + layer["class_name"]
                # this is for inbounds search
                q_name = layer["config"]["name"]
                # for layers that have weights
                if layer["config"]["name"] in self.layer_weights:
                    kernel = self.layer_weights[layer["config"]["name"]][0]
                    dim = list(range(0, kernel.ndim))
                    t_dim = [dim.pop(-1)]
                    t_dim.extend(dim)
                    channel_size = kernel.shape[-1]
                    kernel_channel = kernel.transpose(t_dim).reshape(channel_size, -1)
                    layer_config["min_value"] = json.dumps(np.min(kernel_channel, axis=1).tolist())
                    layer_config["max_value"] = json.dumps(np.max(kernel_channel, axis=1).tolist())
                else:
                    # default value, but never expected to be used
                    # cause no kernel weights for this layer
                    layer_config["min_value"] = json.dumps([-10000])
                    layer_config["max_value"] = json.dumps([10000])
                layer_config["name"] = q_name
                q_layer = {"class_name": q_layer_name, "name": q_name, "config": layer_config}
                if "inbound_nodes" in layer:
                    q_layer["inbound_nodes"] = inbound_reverse_map[layer["name"]]
                q_layers.append(q_layer)
            else:
                q_layers.append(layer)

        json_model["config"]["layers"] = q_layers
        quantized_model = self._restore_model_from_json(json_model)
        return quantized_model

    def convert_bf16(self):
        """Execute the BF16 conversion."""
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        json_model = copy.deepcopy(json.loads(self.pre_optimized_object.to_json()))

        for layer in json_model["config"]["layers"]:
            if layer["config"]["name"] in self.bf16_ops:
                layer["config"]["dtype"] = "mixed_bfloat16"

        converted_model = self._restore_model_from_json(json_model)
        tf.keras.mixed_precision.set_global_policy("float32")

        return converted_model

    # (TODO) choose the properly quantize mode
    def _check_quantize_mode(self, json_model):
        """Check what quantize mode to use."""
        config = json_model["config"]
        layers = config["layers"]
        for idx, layer in enumerate(layers):
            if "ReLU" in layer["class_name"]:
                return "MIN_FIRST"
        return "SCALED"

    def _restore_model_from_json(self, json_model):
        """Generate a keras model from json files."""
        from tensorflow.keras.models import model_from_json

        custom_objects = {}
        # We need to keep a dictionary of custom objects as our quantized library
        # is not recognized by keras.
        custom_objects = _add_supported_quantized_objects(custom_objects)
        json_model_file = json.dumps(json_model)
        qmodel = model_from_json(json_model_file, custom_objects=custom_objects)
        qmodel = self._set_weights(qmodel, self.layer_weights)
        return qmodel

    # set fp32 weights to qmodel
    def _set_weights(self, qmodel, layer_weights):
        for qlayer in qmodel.layers:
            if qlayer.get_weights():
                if qlayer.name in layer_weights:
                    qlayer.set_weights(layer_weights[qlayer.name])
                else:
                    hit_layer = False
                    for sub_layer in qlayer.submodules:
                        if sub_layer.name in layer_weights:
                            qlayer.set_weights(layer_weights[sub_layer.name])
                            hit_layer = True
                            break
                    if not hit_layer:
                        raise ValueError("Can not match the module weights....")
        return qmodel

    @dump_elapsed_time(customized_msg="Model inference")
    def evaluate(
        self,
        model,
        dataloader,
        postprocess=None,
        metrics=None,
        measurer=None,
        iteration=-1,
        tensorboard=False,
        fp32_baseline=False,
    ):
        """The function is used to run evaluation on validation dataset.

        Args:
            model (object): The model to do calibration.
            dataloader (generator): generate the data and labels.
            postprocess (object, optional): process the result from the model
            metric (object, optional): Depends on model category. Defaults to None.
            measurer (object, optional): for precise benchmark measurement.
            iteration(int, optional): control steps of mini-batch
            tensorboard (boolean, optional): for tensorboard inspect tensor.
            fp32_baseline (boolean, optional): only for compare_label=False pipeline
        """
        # use keras object
        keras_model = model.model
        logger.info("Start to evaluate the Keras model.")
        results = []
        for idx, (inputs, labels) in enumerate(dataloader):
            # use predict on batch
            if measurer is not None:
                measurer.start()
                predictions = keras_model.predict_on_batch(inputs)
                measurer.end()
            else:
                predictions = keras_model.predict_on_batch(inputs)

            if self.fp32_preds_as_label:
                self.fp32_results.append(predictions) if fp32_baseline else results.append(predictions)

            if postprocess is not None:
                predictions, labels = postprocess((predictions, labels))
            if metrics:
                for metric in metrics:
                    if not hasattr(metric, "compare_label") or (
                        hasattr(metric, "compare_label") and metric.compare_label
                    ):
                        metric.update(predictions, labels)
            if idx + 1 == iteration:
                break

        acc = 0 if metrics is None else [metric.result() for metric in metrics]

        return acc if not isinstance(acc, list) or len(acc) > 1 else acc[0]

    def query_fw_capability(self, model):
        """The function is used to return framework tuning capability.

        Args:
            model (object): The model to query quantization tuning capability.
        """
        fp32_config = {"weight": {"dtype": "fp32"}, "activation": {"dtype": "fp32"}}
        bf16_config = {"weight": {"dtype": "bf16"}, "activation": {"dtype": "bf16"}}
        int8_type = self.query_handler.get_op_types_by_precision(precision="int8")
        op_capability = self.query_handler.get_quantization_capability()
        conv_config = copy.deepcopy(op_capability["int8"]["Conv2D"])
        conv_config = copy.deepcopy(op_capability["int8"]["SeparableConv2D"])
        conv_config = copy.deepcopy(op_capability["int8"]["DepthwiseConv2D"])
        dense_config = copy.deepcopy(op_capability["int8"]["Dense"])
        maxpool_config = copy.deepcopy(op_capability["int8"]["MaxPooling2D"])
        avgpool_config = copy.deepcopy(op_capability["int8"]["AveragePooling2D"])
        other_config = copy.deepcopy(op_capability["int8"]["default"])

        # # get fp32 layer weights
        keras_object = model
        self.conv_weights = {}
        self.bn_weights = {}
        self.layer_weights = {}
        for layer in keras_object.layers:
            if layer.get_weights():
                if (
                    isinstance(layer, tf.keras.layers.Conv2D)
                    or isinstance(layer, tf.keras.layers.DepthwiseConv2D)
                    or isinstance(layer, tf.keras.layers.SeparableConv2D)
                ):
                    self.conv_weights[layer.name] = copy.deepcopy(layer.get_weights())
                elif isinstance(layer, tf.keras.layers.BatchNormalization):
                    self.bn_weights[layer.name] = copy.deepcopy(layer.get_weights())
                self.layer_weights[layer.name] = copy.deepcopy(layer.get_weights())
        self.pre_optimized_object = self._pre_optimize(keras_object)

        json_model = copy.deepcopy(json.loads(self.pre_optimized_object.to_json()))
        config = json_model["config"]
        self.fp32_layers = config["layers"]

        quantizable_op_details = OrderedDict()
        for details in self.fp32_layers:
            node_op = details["class_name"]
            node_name = details["config"]["name"]
            if node_op == "Conv2D":
                quantizable_op_details[(node_name, node_op)] = [conv_config, bf16_config, fp32_config]
            elif node_op == "Dense":
                quantizable_op_details[(node_name, node_op)] = [dense_config, bf16_config, fp32_config]
            elif node_op in {"AveragePooling2D", "AvgPool2D"}:
                quantizable_op_details[(node_name, node_op)] = [avgpool_config, bf16_config, fp32_config]
            elif node_op in {"MaxPooling2D", "MaxPool2D"}:
                quantizable_op_details[(node_name, node_op)] = [maxpool_config, bf16_config, fp32_config]
            else:
                quantizable_op_details[(node_name, node_op)] = [bf16_config, fp32_config]

        capability = {
            "opwise": copy.deepcopy(quantizable_op_details),
            "optypewise": self.get_optype_wise_ability(quantizable_op_details),
        }
        logger.debug("Dump framework quantization capability:")
        logger.debug(capability)

        return capability

    def get_optype_wise_ability(self, quantizable_op_details):
        """Get the op type wise capability by generating the union value of each op type.

        Returns:
            [string dict]: the key is op type while the value is the
                           detail configurations of activation and weight for this op type.
        """
        res = OrderedDict()
        for op in quantizable_op_details:
            if op[1] not in res:
                res[op[1]] = {"activation": quantizable_op_details[op][0]["activation"]}
                if "weight" in quantizable_op_details[op][0]:
                    res[op[1]]["weight"] = quantizable_op_details[op][0]["weight"]
        return res


class KerasQuery:
    """Class that queries configs from yaml settings."""

    def __init__(self, local_config_file=None):
        """Initialize KerasQuery."""
        self.version = tf.version.VERSION
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _one_shot_query(self):
        """Query cur configs in one shot."""
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e:
                logger.info("Fail to parse {} due to {}.".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError(
                    "Please check if the format of {} follows Neural Compressor yaml schema.".format(self.cfg)
                )

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime.
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        default_config = None
        for sub_data in data:
            if sub_data["version"]["name"] == self.version:
                return sub_data

            if sub_data["version"]["name"] == "default":
                default_config = sub_data

        return default_config

    def get_version(self):
        """Get the current backend version information.

        Returns:
            [string]: version string.
        """
        return self.cur_config["version"]["name"]

    def get_precisions(self):
        """Get supported precisions for current backend.

        Returns:
            [string list]: the precisions' name.
        """
        return self.cur_config["precisions"]["names"]

    def get_op_types(self):
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return self.cur_config["ops"]

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        return self.cur_config["capabilities"]

    def get_op_types_by_precision(self, precision):
        """Get op types per precision.

        Args:
            precision (string): precision name

        Returns:
            [string list]: A list composed of op type.
        """
        assert precision in list(self.cur_config["ops"].keys())
        return self.cur_config["ops"][precision]
