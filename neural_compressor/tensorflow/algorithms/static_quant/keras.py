#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
"""The quantization classes for Keras."""

import copy
import json
import os
from collections import OrderedDict, UserDict
from typing import Callable, Dict

import numpy as np
import tensorflow as tf
import yaml

from neural_compressor.common import logger
from neural_compressor.common.utils import DEFAULT_WORKSPACE
from neural_compressor.tensorflow.keras.layers import (
    QAvgPool2D,
    QConv2D,
    QDense,
    QDepthwiseConv2D,
    QMaxPool2D,
    QSeparableConv2D,
)
from neural_compressor.tensorflow.quantization.config import StaticQuantConfig
from neural_compressor.tensorflow.utils import deep_get, dump_elapsed_time, version1_gte_version2


class KerasAdaptor:
    """The keras class of framework adaptor layer."""

    supported_op = [
        "Conv2D",
        "Dense",
        "SeparableConv2D",
        "DepthwiseConv2D",
        "AveragePooling2D",
        "MaxPooling2D",
        "AvgPool2D",
        "MaxPool2D",
    ]

    custom_layers = {
        "QConv2D": QConv2D,
        "QDepthwiseConv2D": QDepthwiseConv2D,
        "QSeparableConv2D": QSeparableConv2D,
        "QDense": QDense,
        "QMaxPool2D": QMaxPool2D,
        "QAvgPool2D": QAvgPool2D,
        "QMaxPooling2D": QMaxPool2D,
        "QAveragePooling2D": QAvgPool2D,
    }

    def __init__(self, framework_specific_info):
        """Initialize the KerasAdaptor class with framework specific information."""
        self.framework_specific_info = framework_specific_info
        self.approach = deep_get(self.framework_specific_info, "approach", False)
        self.quantize_config = {"op_wise_config": {}}
        self.device = self.framework_specific_info["device"]
        self.backend = self.framework_specific_info["backend"]
        self.recipes = deep_get(self.framework_specific_info, "recipes", {})

        self.pre_optimized_model = None
        self.pre_optimizer_handle = None
        self.bf16_ops = []
        self.fp32_ops = []
        self.query_handler = KerasQuery(local_config_file=os.path.join(os.path.dirname(__file__), "keras.yaml"))

        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.callbacks = []

        self.conv_format = {}
        self.fold_conv = []
        self.keras3 = True if version1_gte_version2(tf.__version__, "2.16.1") else False
        if not os.path.exists(DEFAULT_WORKSPACE):
            os.makedirs(DEFAULT_WORKSPACE)
        self.tmp_dir = (DEFAULT_WORKSPACE + "tmp_model.keras") if self.keras3 else (DEFAULT_WORKSPACE + "tmp_model")

    def _set_weights(self, qmodel, layer_weights):
        """Set fp32 weights to qmodel."""
        for qlayer in qmodel.layers:
            if qlayer.get_weights():
                if qlayer.name in layer_weights:
                    qlayer.set_weights(layer_weights[qlayer.name])
                else:  # pragma: no cover
                    hit_layer = False
                    for sub_layer in qlayer.submodules:
                        if sub_layer.name in layer_weights:
                            qlayer.set_weights(layer_weights[sub_layer.name])
                            hit_layer = True
                            break
                    if not hit_layer:
                        raise ValueError("Can not match the module weights....")
        return qmodel

    def _check_quantize_format(self, model):
        """The function that checks format for conv ops."""
        input_layer_dict = {}
        layer_name_mapping = {}

        for layer in model.layers:
            layer_name_mapping[layer.name] = layer
            for node in layer._outbound_nodes:
                layer_name = node.operation.name if self.keras3 else node.outbound_layer.name
                if layer_name not in input_layer_dict:
                    input_layer_dict[layer_name] = [layer.name]
                else:
                    input_layer_dict[layer_name].append(layer.name)

        for layer in model.layers:
            if layer.__class__.__name__ in self.supported_op:
                self.conv_format[layer.name] = "s8"
                input_layer_names = input_layer_dict[layer.name]
                for input_layer_name in input_layer_names:
                    check_layer = layer_name_mapping[input_layer_name]
                    if check_layer.__class__.__name__ == "Activation" and check_layer.activation.__name__ in ["relu"]:
                        self.conv_format[layer.name] = "u8"
                        break

    def _fuse_bn_keras3(self, fuse_conv_bn, fp32_layers):  # pragma: no cover
        fuse_layers = []
        fused_bn_name = ""
        for idx, layer in enumerate(fp32_layers):
            if hasattr(layer, "_outbound_nodes"):
                if layer.name == fused_bn_name:
                    continue

                if layer.name in self.conv_weights.keys():
                    new_outbound_nodes = []
                    conv_weight = self.conv_weights[layer.name]
                    for outbound_node in layer._outbound_nodes:
                        outbound_layer = outbound_node.operation
                        if outbound_layer.__class__.__name__ in ("BatchNormalization"):
                            fused_bn_name = outbound_layer.name
                            bn_weight = self.bn_weights[fused_bn_name]
                            self.layer_weights[layer.name] = fuse_conv_bn(
                                conv_weight, bn_weight, layer.__class__.__name__, outbound_layer.epsilon
                            )
                            self.fold_conv.append(layer.name)
                            for node in outbound_layer._outbound_nodes:
                                new_outbound_nodes.append(node)
                        else:
                            new_outbound_nodes.append(outbound_node)
                    layer._outbound_nodes.clear()
                    for node in new_outbound_nodes:
                        layer._outbound_nodes.append(node)

                fuse_layers.append(layer)
            else:
                if (
                    idx > 0
                    and layer.__class__.__name__ == "BatchNormalization"
                    and fp32_layers[idx - 1].__class__.__name__ == "Conv2D"
                ):
                    conv_name = fp32_layers[idx - 1].name
                    conv_weight = self.conv_weights[conv_name]
                    bn_weight = self.bn_weights[layer.name]
                    conv_type = fp32_layers[idx - 1].__class__.__name__

                    self.layer_weights[conv_name] = fuse_conv_bn(conv_weight, bn_weight, conv_type, layer.epsilon)
                    self.fold_conv.append(conv_name)
                else:
                    fuse_layers.append(layer)

        return fuse_layers

    def _fuse_bn_keras2(self, fuse_conv_bn, fp32_layers):  # pragma: no cover
        fuse_layers = []
        for idx, layer in enumerate(fp32_layers):
            if hasattr(layer, "_inbound_nodes"):
                if layer.__class__.__name__ in ("BatchNormalization"):
                    for bn_inbound_node in layer._inbound_nodes:
                        inbound_layer = bn_inbound_node.inbound_layers
                        if inbound_layer.name in self.conv_weights.keys():
                            conv_layer = inbound_layer
                            conv_weight = self.conv_weights[conv_layer.name]
                            bn_weight = self.bn_weights[layer.name]

                            self.layer_weights[conv_layer.name] = fuse_conv_bn(
                                conv_weight, bn_weight, conv_layer.__class__.__name__, layer.epsilon
                            )
                            self.fold_conv.append(conv_layer.name)
                        else:
                            fuse_layers.append(layer)
                elif len(layer._inbound_nodes):
                    new_bound_nodes = []
                    # OpLambda node will have different bound node
                    if layer.__class__.__name__ in ("TFOpLambda", "SlicingOpLambda"):
                        fuse_layers.append(layer)
                    else:
                        for bound_node in layer._inbound_nodes:
                            inbound_layers = bound_node.inbound_layers
                            if not isinstance(inbound_layers, list):
                                inbound_layers = [inbound_layers]
                            for inbound_layer in inbound_layers:
                                if inbound_layer in self.bn_weights.keys():
                                    for bn_inbound_node in inbound_layer._inbound_nodes:
                                        bn_inbound_layer = bn_inbound_node.inbound_layers
                                        if bn_inbound_layer.name in self.conv_weights.keys():
                                            new_bound_nodes.append(bn_inbound_node)
                                        else:
                                            if bound_node not in new_bound_nodes:
                                                new_bound_nodes.append(bound_node)
                                else:
                                    new_bound_nodes.append(bound_node)

                        layer._inbound_nodes.clear()
                        for bound_node in new_bound_nodes:
                            layer._inbound_nodes.append(bound_node)
                        fuse_layers.append(layer)
                else:
                    fuse_layers.append(layer)
            else:
                if (
                    idx > 0
                    and layer.__class__.__name__ == "BatchNormalization"
                    and fp32_layers[idx - 1].__class__.__name__ == "Conv2D"
                ):
                    conv_name = fp32_layers[idx - 1].name
                    conv_weight = self.conv_weights[conv_name]
                    bn_weight = self.bn_weights[layer.name]
                    conv_type = fp32_layers[idx - 1].__class__.__name__

                    self.layer_weights[conv_name] = fuse_conv_bn(conv_weight, bn_weight, conv_type, layer.epsilon)
                    self.fold_conv.append(conv_name)
                else:
                    fuse_layers.append(layer)

        return fuse_layers

    def _fuse_bn(self, model):  # pragma: no cover
        """Fusing Batch Normalization."""
        model.save(self.tmp_dir)
        fuse_bn_model = tf.keras.models.load_model(self.tmp_dir)
        fp32_layers = fuse_bn_model.layers

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

        fuse_bn_function = self._fuse_bn_keras3 if self.keras3 else self._fuse_bn_keras2
        fused_layers = fuse_bn_function(fuse_conv_bn, fp32_layers)

        for idx, layer in enumerate(fused_layers):
            if (
                layer.__class__.__name__ in ("Conv2D", "DepthwiseConv2D", "SeparableConv2D")
                and layer.name in self.fold_conv
            ):
                conv_config = layer.get_config()
                conv_config["use_bias"] = True
                conv_layer = type(layer).from_config(conv_config)
                for node in layer._outbound_nodes:
                    conv_layer._outbound_nodes.append(node)
                fused_layers[idx] = conv_layer

        bn_surgery = KerasSurgery(model)
        bn_fused_model = bn_surgery.convert(fused_layers, self.conv_weights.keys())
        bn_fused_model = self._set_weights(bn_fused_model, self.layer_weights)

        bn_fused_model.save(self.tmp_dir)
        bn_fused_model = tf.keras.models.load_model(self.tmp_dir)

        return bn_fused_model

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, quant_config, model, dataloader, iteration, calib_func=None):
        """Execute the quantize process on the specified model.

        Args:
            quant_config(dict): The user defined 'StaticQuantConfig' class.
            model (object): The model to do quantization.
            dataloader(object): The calibration dataloader used to load quantization dataset.
            iteration(int): The iteration of calibration.
            calib_func (optional): the function used for calibration, should be a substitution for calibration
            dataloader when the built-in calibration function of INC does not work for model inference.
        """
        assert calib_func is None, "The calibration function is not supported on Keras backend yet"
        self.query_fw_capability(model)
        converter = KerasConfigConverter(quant_config, iteration)
        tune_cfg = converter.parse_to_tune_cfg()
        self.tuning_cfg_to_fw(tune_cfg)

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

        from neural_compressor.tensorflow.keras.layers import layer_initializer_dict

        q_layer_dict = {}
        for layer in self.pre_optimized_model.layers:
            if layer.__class__.__name__ in self.supported_op and layer.name in self.quantize_config["op_wise_config"]:
                op_config = self.quantize_config["op_wise_config"][layer.name]
                granularity = "per_channel" if op_config[0] else "per_tensor"
                q_layer_class = "Q" + layer.__class__.__name__
                q_config = {"T": self.conv_format[layer.name], "granularity": granularity}
                q_layer = layer_initializer_dict[q_layer_class](layer, q_config)
                q_layer_dict[layer.name] = q_layer

        calib_surgery = KerasSurgery(self.pre_optimized_model)
        calibration_model = calib_surgery.convert(q_layer_dict=q_layer_dict)
        calibration_model = self._set_weights(calibration_model, self.layer_weights)

        quantized_model = self._calibrate(
            calibration_model,
            dataloader,
            self.quantize_config["calib_iteration"],
        )

        return quantized_model

    def _calibrate(self, model, dataloader=None, calib_interation=None):
        """Apply calibration.

        Args:
            model(tf.keras.Model): The model inserted with FakeQuant layers for calibration.
            dataloader(object): The calibration dataloader used to load quantization dataset.
            calib_interation(int): The iteration of calibration.
        """
        # run eagerly to fetch the numpy min/max
        results = {}
        model.compile(run_eagerly=True)
        for idx, (inputs, _) in enumerate(dataloader):
            _ = model.predict_on_batch(inputs)
            for layer in model.layers:
                if (
                    layer.__class__.__name__[1:] in self.supported_op
                    and layer.name in self.quantize_config["op_wise_config"]
                ):
                    min_value = layer.act_min_value.numpy()
                    max_value = layer.act_max_value.numpy()

                    if layer.name not in results:
                        results[layer.name] = {"min": [min_value], "max": [max_value]}
                    else:
                        results[layer.name]["min"].append(min_value)
                        results[layer.name]["max"].append(max_value)
            if idx + 1 == calib_interation:
                break

        for idx, layer in enumerate(model.layers):
            if (
                layer.__class__.__name__[1:] in self.supported_op
                and layer.name in self.quantize_config["op_wise_config"]
            ):
                layer.act_min_value = np.min(results[layer.name]["min"])
                layer.act_max_value = np.max(results[layer.name]["max"])
                layer.quant_status = "quantize"
                # for layers that have weights
                if layer.name in self.layer_weights:
                    kernel = self.layer_weights[layer.name][0]
                    dim = list(range(0, kernel.ndim))
                    t_dim = [dim.pop(-1)]
                    t_dim.extend(dim)
                    channel_size = kernel.shape[-1]
                    kernel_channel = kernel.transpose(t_dim).reshape(channel_size, -1)

                    layer.weight_min_value = np.min(kernel_channel, axis=1).tolist()
                    layer.weight_max_value = np.max(kernel_channel, axis=1).tolist()
                else:
                    # default value, but never expected to be used
                    # cause no kernel weights for this layer
                    layer.weight_min_value = [-10000]
                    layer.weight_max_value = [10000]

        model.save(self.tmp_dir)
        quantized_model = tf.keras.models.load_model(self.tmp_dir)

        return quantized_model

    def query_fw_capability(self, model):
        """The function is used to return framework tuning capability.

        Args:
            model (object): The model to query quantization tuning capability.
        """
        if not isinstance(model, tf.keras.Model):
            model = model.model
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
        self.fp32_model = model
        self.conv_weights = {}
        self.bn_weights = {}
        self.layer_weights = {}
        for layer in self.fp32_model.layers:
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

        self._check_quantize_format(self.fp32_model)
        self.pre_optimized_model = self._fuse_bn(self.fp32_model)

        quantizable_layer_details = OrderedDict()
        for layer in self.fp32_model.layers:
            layer_class = layer.__class__.__name__
            if layer_class == "Conv2D":
                quantizable_layer_details[(layer.name, layer_class)] = [conv_config, bf16_config, fp32_config]
            elif layer_class == "Dense":
                quantizable_layer_details[(layer.name, layer_class)] = [dense_config, bf16_config, fp32_config]
            elif layer_class in {"AveragePooling2D", "AvgPool2D"}:
                quantizable_layer_details[(layer.name, layer_class)] = [avgpool_config, bf16_config, fp32_config]
            elif layer_class in {"MaxPooling2D", "MaxPool2D"}:
                quantizable_layer_details[(layer.name, layer_class)] = [maxpool_config, bf16_config, fp32_config]
            else:
                quantizable_layer_details[(layer.name, layer_class)] = [bf16_config, fp32_config]

        capability = {
            "opwise": copy.deepcopy(quantizable_layer_details),
            "optypewise": self.get_optype_wise_ability(quantizable_layer_details),
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

    def tuning_cfg_to_fw(self, tuning_cfg):
        """Parse tune_config and set framework variables.

        Args:
            tuning_cfg (dict): The dict of tuning config.
        """
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

            if tuning_cfg["op"][each_op_info]["activation"]["dtype"] == "bf16":  # pragma: no cover
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


class KerasConfigConverter:
    """Convert `StaticQuantConfig` to the format used by static quant algo."""

    support_int8_weight = {"Dense", "Conv2D", "DepthwiseConv2D", "SeparableConv2D"}

    def __init__(self, quant_config: StaticQuantConfig, calib_iteration: int):
        """Init parser for keras static quant config.

        Args:
            quant_config: the keras static quant config.
            calib_iteration: the iteration of calibration.
        """
        self.quant_config = quant_config
        self.calib_iteration = calib_iteration

    def update_config(self, quant_config, op_key):
        """Update op-wise config.

        Args:
            quant_config: the keras static quant config.
            op_key: a tuple such as (layer type, layer name).
        """
        op_value = {"activation": {}}
        op_value["activation"].update(
            {
                "dtype": quant_config.act_dtype,
                "quant_mode": "static",
                "scheme": ("sym" if quant_config.act_sym else "asym"),
                "granularity": quant_config.act_granularity,
                "algorithm": "minmax",
            }
        )
        if op_key[1] not in self.support_int8_weight:
            return op_value

        op_value["weight"] = {
            "dtype": quant_config.weight_dtype,
            "scheme": "sym" if quant_config.weight_sym else "asym",
            "granularity": quant_config.weight_granularity,
            "algorithm": "minmax",
        }
        return op_value

    def parse_to_tune_cfg(self) -> Dict:
        """The function that parses StaticQuantConfig to keras tuning config."""
        tune_cfg = {"op": OrderedDict()}
        for op_key, config in self.quant_config.items():
            op_value = self.update_config(config, op_key)
            tune_cfg["op"].update({op_key: op_value})
            tune_cfg["calib_iteration"] = self.calib_iteration

        return tune_cfg


class KerasSurgery:
    """The class that inserts FakeQuant or QDQ layers before the target layers."""

    def __init__(self, model):
        """Init the KerasSurgery class.

        Args:
            model: the model to be modified.
        """
        import shutil

        self.model_outputs = []
        self.keras3 = True if version1_gte_version2(tf.__version__, "2.16.1") else False
        self.tmp_dir = (DEFAULT_WORKSPACE + "tmp_model.keras") if self.keras3 else (DEFAULT_WORKSPACE + "tmp_model")
        model.save(self.tmp_dir)
        self.model = tf.keras.models.load_model(self.tmp_dir)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _parse_inputs(self, BN_fused_layers=None, conv_names=None):
        """Create a input_layer_dict from model.

        Args:
            BN_fused_layers: The layers in which BN layers have been fused.
            conv_names: The name list of conv layers where BNs are fused.

        Returns:
            input_layer_dict: The dict that mapping for layer names to their input layer names.
        """
        input_layer_dict = {}
        layers = BN_fused_layers if BN_fused_layers else self.model.layers
        for layer in layers:
            for node in layer._outbound_nodes:
                out_layer = node.operation if self.keras3 else node.outbound_layer
                out_layer_names = [out_layer.name]
                if conv_names and out_layer.__class__.__name__ in ("BatchNormalization") and layer.name in conv_names:
                    out_layer_names = (
                        [node.operation.name for node in out_layer._outbound_nodes]
                        if self.keras3
                        else [node.outbound_layer.name for node in out_layer._outbound_nodes]
                    )

                for out_layer_name in out_layer_names:
                    if out_layer_name not in input_layer_dict:
                        input_layer_dict[out_layer_name] = [layer.name]
                    else:
                        input_layer_dict[out_layer_name].append(layer.name)

        try:
            model_input = self.model.input
        except ValueError:  # pragma: no cover
            model_input = self.model.inputs[0]

        return input_layer_dict, model_input

    def convert(self, BN_fused_layers=None, conv_names=None, q_layer_dict=None):
        """Generate optimized model by fusing BN layers or replacing Keras layers to custom quantized layers.

        Args:
            BN_fused_layers: The layers in which BN layers have been fused.
            conv_names: The name list of conv layers where BNs are fused.
            q_layer_dict: The dict mapping from keras layers to custom quantized layers.
        """
        input_layer_dict, model_input = self._parse_inputs(BN_fused_layers, conv_names)
        output_tensor_dict = {"keras.Input": model_input}
        layers = BN_fused_layers if BN_fused_layers else self.model.layers

        for idx, layer in enumerate(layers):
            if layer.__class__.__name__ == "InputLayer":
                output_tensor_dict[layer.name] = output_tensor_dict["keras.Input"]
                continue

            input_tensors = (
                output_tensor_dict["keras.Input"]
                if idx == 0
                else [output_tensor_dict[input_layer] for input_layer in input_layer_dict[layer.name]]
            )

            while isinstance(input_tensors, list) and len(input_tensors) == 1:
                input_tensors = input_tensors[0]

            if self.keras3:
                layer._inbound_nodes.clear()

            cur_layer = q_layer_dict[layer.name] if q_layer_dict and layer.name in q_layer_dict else layer
            x = cur_layer(input_tensors)
            output_tensor_dict[layer.name] = x

            if not isinstance(self.model, tf.keras.models.Sequential) and layer.name in self.model.output_names:
                self.model_outputs.append(x)

        if not self.model_outputs:
            self.model_outputs.append(x)

        return tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model_outputs)
