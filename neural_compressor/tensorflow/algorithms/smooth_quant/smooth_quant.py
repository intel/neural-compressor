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

from typing import Callable, Dict

from neural_compressor.common import logger
from neural_compressor.common.utils import DEFAULT_WORKSPACE
from neural_compressor.tensorflow.utils import BaseModel, TensorflowLLMModel
from neural_compressor.tensorflow.algorithms import TensorFlowAdaptor
from neural_compressor.tensorflow.quantization.config import SmoohQuantConfig
from neural_compressor.tensorflow.algorithms.smooth_quant import (
    SmoothQuantScaler, 
    SmoothQuantScalerLLM,
    SmoothQuantCalibration, 
    SmoothQuantCalibrationLLM,
)

default_alpha_args={
        "alpha_min": 0.0,
        "alpha_max": 1.0,
        "alpha_step": 0.1,
        "shared_criterion": "mean",
        "do_blockwise": False,
    }

class SmoothQuant:
    """The class that performs smooth quantization."""
    def __init__(
        self,
        model: BaseModel,
        config: SmoohQuantConfig,
        adaptor: TensorFlowAdaptor,
        calib_dataloader: Callable,
        calib_iteration: int=1,
    ):
        """Convert the model by smooth quant.

        Args:
            model: original model
            config: the SmoohQuantConfig class used to set this class
            adaptor: the TensorFlowAdaptor class from which this class copy some parameters
            calibdataloader: the calibration dataloader
            calib_iteration: how many steps of iterations on the dataloader to move forward

        Returns:
            model: A smoothed Tensorflow model
        """
        self.model = model
        self.config = config
        self.adaptor = adaptor
        self.calib_dataloader = calib_dataloader
        self.calib_iteration = calib_iteration

        self.new_api = adaptor.new_api
        self.device = adaptor.device
        self.itex_mode = adaptor.itex_mode

        self.alpha = self.config.alpha
        self.folding = self.config.folding
        self.percentile = self.config.percentile
        self.op_types = self.config.op_types
        self.scales_per_op = self.config.scales_per_op
        self.record_max_info = self.config.record_max_info
        self.weight_clip = self.config.weight_clip
        self.auto_alpha_args = self.config.auto_alpha_args

    def apply_smooth_quant(self):
        """Apply smooth quant to the model."""
        logger.info("Start Smoothing process for Smooth Quantization.")

        # Do a pre-optimization before smooth quant
        from neural_compressor.tensorflow.quantization.tf_utils.graph_rewriter.generic.pre_optimize import PreOptimization
        pre_optimizer_handle = PreOptimization(self.model, self.new_api, self.device)
        pre_optimized_model = pre_optimizer_handle.get_optimized_model(self.itex_mode)
        self.model.graph_def = pre_optimized_model.graph_def

        # Run calibration to get max values per channel

        calibration = SmoothQuantCalibration(self.model, self.calib_dataloader, \
                                self.calib_iteration, self.op_types, self.percentile)
        max_vals_per_channel, sq_weight_node_names = calibration()

        # Get weight tensors and weight nodes based on the input tensor
        from neural_compressor.tensorflow.quantization.tf_utils.util import get_weight_from_input_tensor
        sq_weight_tensors, sq_weights_nodes = get_weight_from_input_tensor(self.model, max_vals_per_channel.keys(), self.op_types)

        # Calculate the smooth quant scaler and insert Mul op into the graph
        scaler = SmoothQuantScaler(self.model, self.calib_dataloader, self.alpha, self.scales_per_op)
        model, mul_list = scaler.transform(
            max_vals_per_channel, sq_weight_tensors, sq_weights_nodes, sq_weight_node_names
        )
        
        self.adaptor.smooth_quant_mul_ops.extend(mul_list)
        return self.model

    def apply_smooth_quant_LLM(self):
        """Apply smooth quant to the LLM model."""
        # Do a pre-optimization before smooth quant
        from neural_compressor.tensorflow.quantization.tf_utils.graph_rewriter.generic.pre_optimize import PreOptimization
        pre_optimizer_handle = PreOptimization(self.model, self.new_api, self.device)
        pre_optimized_model = pre_optimizer_handle.get_optimized_model(self.itex_mode)
        self.model.graph_def = pre_optimized_model.graph_def

        llm_temp_dir = DEFAULT_WORKSPACE + "/temp_saved_model"
        # Run calibration to get max values per channel
        calibration = SmoothQuantCalibrationLLM(
            self.model._model,
            self.calib_dataloader,
            self.calib_iteration,
            self.op_types,
            self.percentile,
            llm_temp_dir,
            self.model.weight_name_mapping,
        )
        max_vals_per_channel, sq_target_node_names, sq_weight_tensor_dict, sq_graph_def = calibration(
            self.model.input_node_names, self.model.output_node_names
        )

        # Calculate the smooth quant scaler and insert Mul op into the graph
        scaler = SmoothQuantScalerLLM(sq_graph_def, self.alpha, self.scales_per_op, self.op_types)
        sq_graph_def, sq_weight_scale_dict, mul_list = scaler.transform(
            max_vals_per_channel, sq_weight_tensor_dict, sq_target_node_names
        )
        self.model.graph_def = sq_graph_def
        self.model.model_path = llm_temp_dir
        self.model.sq_weight_scale_dict = sq_weight_scale_dict
        self.adaptor.smooth_quant_mul_ops.extend(mul_list)
        return self.model

    def __call__(self):
        apply_func = self.apply_smooth_quant if isinstance(self.model, TensorflowLLMModel) \
                        else self.apply_smooth_quant_LLM

        return apply_func()