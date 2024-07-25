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
"""The core components of sq."""

from typing import Callable, Dict

import tensorflow as tf

from neural_compressor.common import logger
from neural_compressor.common.utils import DEFAULT_WORKSPACE
from neural_compressor.tensorflow.algorithms.smoother.calibration import (
    SmoothQuantCalibration,
    SmoothQuantCalibrationLLM,
)
from neural_compressor.tensorflow.algorithms.smoother.scaler import SmoothQuantScaler, SmoothQuantScalerLLM
from neural_compressor.tensorflow.quantization.config import SmoothQuantConfig
from neural_compressor.tensorflow.quantization.utils.graph_util import GraphAnalyzer
from neural_compressor.tensorflow.utils import SPR_BASE_VERSIONS, BaseModel, TensorflowLLMModel, TFConfig


class SmoothQuant:
    """The class that performs smooth quantization."""

    def __init__(
        self,
        config: SmoothQuantConfig,
        calib_dataloader: Callable = None,
        calib_iteration: int = 1,
        calib_func: Callable = None,
    ):
        """Convert the model by smooth quant.

        Args:
            config: the SmoothQuantConfig class used to set this class.
            calibdataloader: the calibration dataloader.
            calib_iteration: how many steps of iterations on the dataloader to move forward.
            calib_func: the function used for calibration, should be a substitution for calib_dataloader
            when the built-in calibration function of INC does not work for model inference.

        Returns:
            model: A smoothed Tensorflow model
        """
        assert calib_func is None, "calibration function is not supported for smooth quant."
        self.config = config
        self.calib_dataloader = calib_dataloader
        self.calib_iteration = calib_iteration

        self.new_api = tf.version.VERSION in SPR_BASE_VERSIONS
        self.device = TFConfig.global_config["device"]
        self.itex_mode = TFConfig.global_config["backend"] == "itex"

        for _, value in self.config.items():
            single_config = value
            break

        self.alpha = single_config.alpha
        self.folding = single_config.folding
        self.percentile = single_config.percentile
        self.op_types = single_config.op_types
        self.scales_per_op = single_config.scales_per_op
        self.record_max_info = single_config.record_max_info
        self.weight_clip = single_config.weight_clip
        self.auto_alpha_args = single_config.auto_alpha_args

    def get_weight_from_input_tensor(self, model, input_tensor_names):
        """Extracts weight tensors and their associated nodes from a smooth quant node's input tensor.

        Args:
            model: A TensorFlow model containing a `graph_def` attribute.
            input_tensor_names: A list of input tensor names to search for weight tensors.

        Returns:
            A tuple of two dictionaries:
            - sq_weight_tensors: A dictionary mapping each input tensor name
                to a dict of its associated weight tensors with weight name.
            - sq_weights_nodes: A dictionary mapping each input tensor name
                to a dict of its associated weight nodes with weight name.
        """
        g_analyzer = GraphAnalyzer()
        g_analyzer.graph = model.graph_def
        graph_info = g_analyzer.parse_graph()

        sq_weight_tensors = {}
        sq_weights_nodes = {}

        from tensorflow.python.framework import tensor_util

        for name in input_tensor_names:
            # Use dict rather than list to fix the QKV/VQK misorder issue
            curr_weight_tensors = {}
            curr_weights_nodes = {}
            next_node_names = graph_info[name].outputs
            for node_name in next_node_names:
                curr_node = graph_info[node_name].node
                if curr_node.op not in self.op_types:
                    continue
                if len(curr_node.input) >= 2:
                    weight_name = curr_node.input[1]
                    weight_node = graph_info[weight_name].node
                    weight_tensor = tensor_util.MakeNdarray(weight_node.attr["value"].tensor)
                    curr_weight_tensors[weight_name] = weight_tensor
                    curr_weights_nodes[weight_name] = weight_node
            # {input node -> {xxx_q_proj_matmul: value1, xxx_v_proj_matmul: value2, ...}, ...}
            sq_weight_tensors[name] = curr_weight_tensors
            sq_weights_nodes[name] = curr_weights_nodes
        return sq_weight_tensors, sq_weights_nodes

    def apply_smooth_quant(self, model: BaseModel):
        """Apply smooth quant to the model."""
        logger.info("Start Smoothing process for Smooth Quantization.")

        # Do a pre-optimization before smooth quant
        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.pre_optimize import PreOptimization

        pre_optimizer_handle = PreOptimization(model, self.new_api, self.device)
        pre_optimized_model = pre_optimizer_handle.get_optimized_model(self.itex_mode)
        model.graph_def = pre_optimized_model.graph_def

        # Run calibration to get max values per channel

        calibration = SmoothQuantCalibration(
            model, self.calib_dataloader, self.calib_iteration, self.op_types, self.percentile
        )
        max_vals_per_channel, sq_weight_node_names = calibration()

        # Get weight tensors and weight nodes based on the input tensor
        sq_weight_tensors, sq_weights_nodes = self.get_weight_from_input_tensor(model, max_vals_per_channel.keys())

        # Calculate the smooth quant scaler and insert Mul op into the graph
        scaler = SmoothQuantScaler(model, self.calib_dataloader, self.alpha, self.scales_per_op)
        model, mul_list = scaler.transform(
            max_vals_per_channel, sq_weight_tensors, sq_weights_nodes, sq_weight_node_names
        )

        return model

    def apply_smooth_quant_LLM(self, model: BaseModel):
        """Apply smooth quant to the LLM model."""
        # Do a pre-optimization before smooth quant
        from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.pre_optimize import PreOptimization

        pre_optimizer_handle = PreOptimization(model, self.new_api, self.device)
        pre_optimized_model = pre_optimizer_handle.get_optimized_model(self.itex_mode)
        model.graph_def = pre_optimized_model.graph_def

        llm_temp_dir = DEFAULT_WORKSPACE + "/temp_saved_model"
        # Run calibration to get max values per channel
        calibration = SmoothQuantCalibrationLLM(
            model._model,
            self.calib_dataloader,
            self.calib_iteration,
            self.op_types,
            self.percentile,
            llm_temp_dir,
            model.weight_name_mapping,
        )
        max_vals_per_channel, sq_target_node_names, sq_weight_tensor_dict, sq_graph_def = calibration(
            model.input_node_names, model.output_node_names
        )

        # Calculate the smooth quant scaler and insert Mul op into the graph
        scaler = SmoothQuantScalerLLM(sq_graph_def, self.alpha, self.scales_per_op, self.op_types)
        sq_graph_def, sq_weight_scale_dict, mul_list = scaler.transform(
            max_vals_per_channel, sq_weight_tensor_dict, sq_target_node_names
        )
        model.graph_def = sq_graph_def
        model.model_path = llm_temp_dir
        model.sq_weight_scale_dict = sq_weight_scale_dict
        return model

    def __call__(self, model: BaseModel):
        """Convert the model by smooth quant.

        Args:
            model: original model

        Returns:
            model: A smoothed Tensorflow model
        """
        apply_func = self.apply_smooth_quant_LLM if isinstance(model, TensorflowLLMModel) else self.apply_smooth_quant

        return apply_func(model)
