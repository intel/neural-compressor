#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
# pylint: disable=no-member

import copy
import logging
import math
import os
import re
import sys
from collections import OrderedDict
from collections.abc import KeysView
from importlib.util import find_spec
from pathlib import Path
from typing import Dict

import numpy as np
import yaml
from packaging.version import Version

from neural_compressor.adaptor.adaptor import Adaptor, adaptor_registry
from neural_compressor.adaptor.ox_utils.util import ONNXRT_BACKENDS, PROVIDERS, to_numpy
from neural_compressor.adaptor.query import QueryBackendCapability
from neural_compressor.data.dataloaders.base_dataloader import BaseDataLoader
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.utils.utility import GLOBAL_STATE, MODE, CpuInfo, LazyImport, Statistics, dump_elapsed_time

onnx = LazyImport("onnx")
ort = LazyImport("onnxruntime")
ONNXRT152_VERSION = Version("1.5.2")
ONNXRT170_VERSION = Version("1.7.0")
ONNXRT112_VERSION = Version("1.12.0")

logger = logging.getLogger("neural_compressor")


@adaptor_registry
class ONNXRUNTIMEAdaptor(Adaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)
        self.__config_dict = {}
        self.quantizable_ops = []
        self.device = framework_specific_info["device"]
        self.static = framework_specific_info["approach"] == "post_training_static_quant"
        self.dynamic = framework_specific_info["approach"] == "post_training_dynamic_quant"
        self.domain = framework_specific_info.get("domain", "auto")
        self.recipes = framework_specific_info.get("recipes", {})
        self._check_backend_available(framework_specific_info["backend"])
        self.backend = PROVIDERS[framework_specific_info["backend"]]
        self.performance_only = framework_specific_info.get("performance_only", False)
        self.use_bf16 = framework_specific_info.get("use_bf16", False) and self.backend in ort.get_available_providers()
        self.use_fp16 = framework_specific_info.get("use_fp16", False)

        # get quantization format according to framework_specific_info
        if (
            not self.dynamic
            and "format" in framework_specific_info
            and framework_specific_info["format"].lower() == "qdq"
        ) or self.backend == "TensorrtExecutionProvider":
            self.format = "qdq"
        else:
            if not self.dynamic:
                self.format = "qlinearops"
            else:
                self.format = "integerops"
                if "format" in framework_specific_info and framework_specific_info["format"].lower() == "qdq":
                    logger.warning("Dynamic approach doesn't support QDQ format.")

        # do not load TensorRT if backend is not TensorrtExecutionProvider
        if self.backend != "TensorrtExecutionProvider":
            os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"

        # get quantization config file according to backend
        config_file = None
        if self.backend == "CPUExecutionProvider":
            config_file = "onnxrt.yaml"
        elif self.backend == "TensorrtExecutionProvider":
            config_file = "onnxrt_trt.yaml"
        elif self.backend == "CUDAExecutionProvider":
            config_file = "onnxrt_cuda.yaml"
        elif self.backend == "DnnlExecutionProvider":
            config_file = "onnxrt_dnnl.yaml"
        elif self.backend == "DmlExecutionProvider":
            config_file = "onnxrt_dml.yaml"
        else:  # pragma: no cover
            assert False, "{} provider is not supported in current environment, " "supported providers: {}".format(
                self.backend, [provider for provider in PROVIDERS.values()]
            )

        self.query_handler_ext = None
        if framework_specific_info["approach"] == "post_training_auto_quant" and self.format != "integerops":
            # if approach is post_training_auto_quant,
            # both static and dynamic quantization will be performed
            self.query_handler = ONNXRTQuery(
                static=True, format=self.format, local_config_file=os.path.join(os.path.dirname(__file__), config_file)
            )
            self.query_handler_ext = ONNXRTQuery(
                dynamic=True, format=self.format, local_config_file=os.path.join(os.path.dirname(__file__), config_file)
            )
        else:
            self.query_handler = ONNXRTQuery(
                dynamic=self.dynamic,
                static=self.static,
                format=self.format,
                local_config_file=os.path.join(os.path.dirname(__file__), config_file),
            )

        self.work_space = framework_specific_info["workspace_path"]
        self.reduce_range = (
            framework_specific_info["reduce_range"]
            if framework_specific_info.get("reduce_range", None) is not None
            else not CpuInfo().vnni
        )
        self.benchmark = GLOBAL_STATE.STATE == MODE.BENCHMARK
        os.makedirs(self.work_space, exist_ok=True)
        self.pre_optimized_model = None
        self.smooth_quant_model = None
        self.quantizable_op_types = []

        for precision in self.query_handler.get_precisions():
            if precision != "fp32":
                if self.device == "cpu" and precision == "fp16":
                    continue
                self.quantizable_op_types += self.query_handler.get_op_types_by_precision(precision=precision)

        if self.backend == "TensorrtExecutionProvider":
            self.recipes["add_qdq_pair_to_weight"] = True
            self.recipes["dedicated_qdq_pair"] = True
            self.recipes["graph_optimization_level"] = "DISABLE_ALL"
            self.recipes["optypes_to_exclude_output_quant"] = ["Conv", "Gemm", "Add", "MatMul"]
            self.static = True
            self.dynamic = False

        self.evaluate_nums = 0

        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.quantize_config = {}  # adaptor should know current configs at any time
        self.quantize_params = {}  # adaptor should know current params at any time
        self.min_max = None

        self.optype_statistics = None

        # sq algo and args
        self.sq = None
        self.cur_sq_args = {}

    def smooth_quant(
        self,
        model,
        dataloader,
        iterations,
        alpha=0.5,
        folding=True,
        percentile=99.999,
        op_types=["MatMul", "Gemm", "Conv", "FusedConv"],
        scales_per_op=True,
        record_max_info=False,
        weight_clip=True,
        auto_alpha_args={
            "alpha_min": 0.0,
            "alpha_max": 1.0,
            "alpha_step": 0.1,
            "shared_criterion": "mean",
            "do_blockwise": False,
        },
        default_alpha=0.5,
    ):
        """Get augmented model with smooth quant.

        Args:
            model_wrapper (object): origin_model
            dataloader (object): dataloader
            iterations (int): iterations
            alpha (float or str): smooth alpha in SmoothQuant, 1.0 will fallback to SPIQ
            folding (bool): whether fold those foldable Mul which are inserted for SmoothQuant
            percentile (float): percentile of calibration to remove outliers
            op_types (list): The op types whose input tensor will be dumped
            scales_per_op (bool): True, each op will have an individual scale, mainly for accuracy
                                  False, ops with the same input will share a scale, mainly for performance
            record_max_info (bool): False, whether record the scale information
            weight_clip: Whether to clip weight when calculating scales; by default it is on.
            auto_alpha_args: Hyperparameters used to set the alpha search space in SQ auto-tuning.
                            By default the search space is 0.0-1.0 with step_size 0.1.
                            do_blockwise: Whether to do blockwise auto-tuning.
            default_alpha: A hyperparameter that is used in SQ auto-tuning; by default it is 0.5.

        Returns:
            model: A modified onnx model
        """
        if self.smooth_quant_model is not None:
            return self.smooth_quant_model

        from .ox_utils.smooth_quant import ORTSmoothQuant

        # set params to cur_sq_args
        self.cur_sq_args["alpha"] = alpha
        self.cur_sq_args["folding"] = folding
        self.cur_sq_args["percentile"] = percentile
        self.cur_sq_args["op_types"] = op_types
        self.cur_sq_args["scales_per_op"] = scales_per_op
        self.cur_sq_args["calib_iter"] = iterations

        # pre-optimization
        self._pre_optimize(model)

        # assign the algo to the adaptor, so adaptor can call it later when needed
        self.sq = ORTSmoothQuant(self.pre_optimized_model, dataloader, self.reduce_range, self.backend)
        self.sq.record_max_info = record_max_info
        self.smooth_quant_model = self.sq.transform(**self.cur_sq_args)
        if not record_max_info:  # pragma: no cover
            logger.info("Updated the pre-optimized model with smooth quant model.")
        else:
            logger.info("Collected scale information for smooth quant.")
        # TODO double-check the smooth_quant_model and pre_optimized_model to make sure there no two fp32 model replicas
        self.pre_optimized_model = self.smooth_quant_model
        return self.smooth_quant_model

    def _need_smooth_quant(self, tune_cfg) -> bool:
        """Check the model needs smooth quant or not."""
        recipe_cfgs = tune_cfg.get("recipe_cfgs", None)
        if (
            recipe_cfgs
            and recipe_cfgs.get("smooth_quant", False)
            and recipe_cfgs["smooth_quant_args"].get("alpha", None)
        ):
            # update alpha according to tune_cfg
            self.cur_sq_args["alpha"] = tune_cfg["recipe_cfgs"]["smooth_quant_args"]["alpha"]
            return True
        else:
            return False

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """The function is used to do calibration and quanitization in post-training
           quantization.

        Args:
            tune_cfg (dict):     quantization config.
            model (object):      model need to do quantization.
            data_loader (object): calibration dataset.
            q_func (optional):   training function for quantization aware training mode,
                                 unimplement yet for onnx.

        Returns:
            (dict): quantized model
        """
        assert q_func is None, "quantization aware training has not been supported on ONNXRUNTIME"
        if self.smooth_quant_model is not None and model.is_smoothquant_model():
            model = self.smooth_quant_model
        elif self.pre_optimized_model is not None:
            model = self.pre_optimized_model
        ort_version = Version(ort.__version__)
        if ort_version < ONNXRT152_VERSION:  # pragma: no cover
            logger.warning("Quantize input needs onnxruntime 1.5.2 or newer.")
            return model
        if ort_version < ONNXRT170_VERSION and self.format == "qdq":
            logger.error("QDQ mode needs onnxruntime1.7.0 or newer.")
            exit(0)
        if model.model.opset_import[0].version < 11:  # pragma: no cover
            logger.warning("Quantize input needs model opset 11 or newer.")
        if self.backend == "DnnlExecutionProvider" and any(
            [i.domain in ["", "ai.onnx"] and i.version < 15 for i in model.model.opset_import]
        ):  # pragma: no cover
            from onnx import version_converter

            try:
                model = self._rename_node(ONNXModel(version_converter.convert_version(model.model, 15)))
            except:
                logging.warning(
                    "Fail to upgrade model opset_import to >= 15, "
                    "please upgrade it manually to run with bf16 data type"
                )
                exit(0)
        self.quantizable_ops = self._query_quantizable_ops(model.model)
        quantize_config = self._cfg_to_quantize_config(tune_cfg)

        if self.performance_only:
            tmp_model = model
        else:
            try:
                tmp_model = copy.deepcopy(model)
            except Exception as e:  # pragma: no cover
                logger.warning("Fail to deep copy the model due to {}, inplace is used now.".format(repr(e)))
                tmp_model = model

        # smooth quant the model if needed
        if self._need_smooth_quant(tune_cfg) and not tmp_model.is_smoothquant_model():
            self.sq.model = tmp_model
            self.sq.record_max_info = False
            tmp_model = self.sq.transform(**self.cur_sq_args)
            logger.info("Model is smooth quantized.")

        iterations = tune_cfg.get("calib_iteration", 1)
        calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)

        if self.recipes.get("layer_wise_quant", False) and not self.dynamic:
            # layer-wise quantization
            # details refer to docs/source/quantization_weight_only.md#layer-wise-quantization
            _model_to_split = copy.deepcopy(tmp_model)

            split_nodes = _model_to_split.find_split_nodes()
            logger.info(
                "Will split model into {} parts to do layer-wise quantization".format(
                    len([node.name for node in split_nodes]) + 1
                )
            )
            logger.debug(
                "Will split model with these nodes for layer-wise quantization: {}".format(
                    [node.name for node in split_nodes]
                )
            )

            split_idx = 1
            model_to_split = [_model_to_split]
            dataloader_for_split_model = [data_loader]
            quantize_params = {}
            quantized_model_merged = None

            while len(model_to_split) != 0:
                split_model = model_to_split.pop(0)
                split_node = split_nodes.pop(0)
                save_both_split_models = True if len(split_nodes) == 0 else False
                shape_infer = True if split_idx == 1 else False

                # split model with given split_node
                split_model_part_1, split_model_part_2 = split_model.split_model_with_node(
                    split_node.name, tmp_model.model_path, shape_infer, save_both_split_models
                )
                if not save_both_split_models:
                    # append split_model_part_2 to do next split
                    model_to_split.append(split_model_part_2)

                logger.info("Quantize split model {}".format(split_idx))
                # get quantize params of split model
                split_quantize_params, dataloder_for_next_split_model = self._get_split_model_quantize_params(
                    split_model_part_1, dataloader_for_split_model, quantize_config, calib_sampling_size, iterations
                )
                dataloader_for_split_model.append(dataloder_for_next_split_model)
                quantize_params.update(split_quantize_params)

                # quantize split model
                quantized_model_merged = self._quantize_split_model(
                    split_model_part_1, quantize_config, split_quantize_params, quantized_model_merged
                )

                split_idx += 1

                # if this is the last split, then quantize the last split model
                if save_both_split_models:
                    logger.info("Quantize split model {}".format(split_idx))
                    # get quantize params of split model
                    split_quantize_params, dataloder_for_next_split_model = self._get_split_model_quantize_params(
                        split_model_part_2, dataloader_for_split_model, quantize_config, calib_sampling_size, iterations
                    )
                    quantize_params.update(split_quantize_params)

                    # quantize split model
                    quantized_model_merged = self._quantize_split_model(
                        split_model_part_2, quantize_config, split_quantize_params, quantized_model_merged
                    )
                    quantized_model_merged.re_org_output(tmp_model.output())  # re-org output as the origin output

            self.quantize_params = quantize_params
            tmp_model.q_config = self._generate_qconfig(model.model, tune_cfg, quantize_params)
            tmp_model.model = quantized_model_merged.model
            self.quantize_config = quantize_config  # update so other methods can know current configs
            self._dump_model_op_stats(tmp_model)
            tmp_model.topological_sort()
            tmp_model.check_is_large_model()

        else:
            if not self.dynamic:
                calib_iterations = self._reset_calib_iter(data_loader, calib_sampling_size, iterations)
                quantize_params, _ = self._get_quantize_params(
                    tmp_model, data_loader, quantize_config, calib_iterations
                )
            else:
                quantize_params = None
            q_config = self._generate_qconfig(model.model, tune_cfg, quantize_params)
            self.quantize_params = quantize_params
            tmp_model = self._quantize_model(tmp_model, quantize_config, quantize_params)
            tmp_model.q_config = q_config
            self._dump_model_op_stats(tmp_model)

        # if the model is large and acc tuning is required, save it to workspace
        if not self.performance_only and tmp_model.is_large_model:  # pragma: no cover
            from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model

            model_name = os.path.split(tmp_model.model_path)[-1]
            model_path = os.path.join(self.work_space, model_name)
            data_name = model_name + "_quantized_data"
            data_path = os.path.join(self.work_space, data_name)

            load_external_data_for_model(tmp_model.model, os.path.dirname(tmp_model.model_path))

            if os.path.isfile(model_path):
                os.remove(model_path)
            if os.path.isfile(data_path):
                os.remove(data_path)

            # if the model is Tranformer-based, save hf config to workspace
            if tmp_model.hf_config is not None:
                model_type = (
                    "" if not hasattr(tmp_model.hf_config, "model_type") else getattr(tmp_model.hf_config, "model_type")
                )
                setattr(tmp_model.hf_config.__class__, "model_type", model_type)
                output_config_file = Path(self.work_space).joinpath("config.json").as_posix()
                if not os.path.exists(output_config_file):
                    tmp_model.hf_config.to_json_file(output_config_file, use_diff=False)

            # save model and external data
            convert_model_to_external_data(
                tmp_model.model,
                all_tensors_to_one_file=True,
                location=data_name,
                size_threshold=1024,
                convert_attribute=False,
            )
            onnx.save_model(tmp_model.model, model_path)

        return tmp_model

    def _get_split_model_quantize_params(
        self, split_model, split_dataloader, quantize_config, calib_sampling_size, iterations
    ):
        """Get quantize params for current split model and get dataloader for next split model."""
        dataloader = split_dataloader.pop(0)
        calib_iterations = self._reset_calib_iter(dataloader, calib_sampling_size, iterations)
        split_quantize_params, dataloder_for_next_split_model = self._get_quantize_params(
            split_model,
            dataloader,
            quantize_config,
            calib_iterations,
            split_model_input_names=split_model.input(),
        )
        return split_quantize_params, dataloder_for_next_split_model

    def _quantize_model(self, model, quantize_config, quantize_params):
        """Quantize model."""
        from neural_compressor import options
        from neural_compressor.adaptor.ox_utils.quantizer import Quantizer
        from neural_compressor.adaptor.ox_utils.util import QuantizationMode

        if self.format == "qlinearops":
            format = QuantizationMode.QLinearOps
        elif self.format == "qdq":
            format = "qdq"
        else:
            format = QuantizationMode.IntegerOps

        quantizer = Quantizer(
            model,
            quantize_config,
            format,
            self.static,
            quantize_params,
            self.quantizable_op_types,
            self.query_handler.get_fallback_list(),
            self.reduce_range,
            (
                options.onnxrt.qdq_setting.AddQDQPairToWeight
                if "add_qdq_pair_to_weight" not in self.recipes
                else self.recipes.get("add_qdq_pair_to_weight", False)
            ),
            (
                options.onnxrt.qdq_setting.OpTypesToExcludeOutputQuantizatioin
                if "optypes_to_exclude_output_quant" not in self.recipes
                else self.recipes.get("optypes_to_exclude_output_quant", [])
            ),
            (
                options.onnxrt.qdq_setting.DedicatedQDQPair
                if "dedicated_qdq_pair" not in self.recipes
                else self.recipes.get("dedicated_qdq_pair", False)
            ),
            self.backend,
        )
        quantizer.quantize_model()
        model.model = quantizer.model.model
        self.quantize_config = quantize_config  # update so other methods can know current configs
        model.topological_sort()
        return model

    def _quantize_split_model(self, split_model, quantize_config, quantize_params, quantized_model_merged):
        """Quantize split model, and merge the quantized models to generate final model."""
        split_model = self._quantize_model(split_model, quantize_config, quantize_params)
        if quantized_model_merged is None:
            quantized_model_merged = split_model
            quantized_model_merged.write_external_data_to_new_location(overwrite=True)
        else:
            quantized_model_merged.merge_split_models(split_model)

        return quantized_model_merged

    def _check_backend_available(self, backend):
        """Check backend is available or not."""
        if backend not in PROVIDERS:
            assert False, "'{}' backend is not supported, " "supported backends include {}".format(
                backend, [provider for provider in PROVIDERS.keys()]
            )

        if backend in ["onnxrt_trt_ep", "onnxrt_cuda_ep"] and self.device != "gpu":
            logger.warning("Backend `{}` requires a GPU device. Reset device to 'gpu'.".format(backend))
            self.device = "gpu"

        if backend in ["onnxrt_dml_ep"] and self.device != "npu":
            logger.warning("Backend `{}` requires a NPU device. Reset device to 'npu'.".format(backend))
            self.device = "npu"

        ep = PROVIDERS[backend]
        if ep not in ort.get_available_providers():
            logger.warning(
                "Specified provider '{}' is not in available provider names. "
                "Fallback to available providers: '{}'".format(ep, ", ".join(ort.get_available_providers()))
            )

    def _reset_calib_iter(self, data_loader, cfg_calib_sampling_size, cfg_calib_iter):
        """Check and reset calibration iterations according to calib_sampleing_size and dataloader batch_size."""
        if isinstance(data_loader, BaseDataLoader):
            batch_size = data_loader.batch_size
            try:
                for i in range(batch_size):
                    if cfg_calib_sampling_size % (batch_size - i) == 0:
                        calib_batch_size = batch_size - i
                        if i != 0:  # pragma: no cover
                            logger.warning(
                                "Reset `calibration.dataloader.batch_size` field "
                                "to {}".format(calib_batch_size) + " to make sure the sampling_size is "
                                "divisible exactly by batch size"
                            )
                        break
                tmp_iterations = int(math.ceil(cfg_calib_sampling_size / calib_batch_size))
                data_loader.batch(calib_batch_size)
                calib_iterations = tmp_iterations
            except Exception as e:  # pragma: no cover
                if "Got invalid dimensions for input" in str(e):
                    logger.warning(
                        "Please set sampling_size to a multiple of {}".format(
                            str(e).partition("Expected: ")[2].partition("\n")[0]
                        )
                    )
                    exit(0)
                logger.warning("Fail to forward with batch size={}, set to {} now.".format(batch_size, 1))
                data_loader.batch(1)
                calib_iterations = cfg_calib_sampling_size
        else:  # pragma: no cover
            if hasattr(data_loader, "batch_size") and cfg_calib_sampling_size % data_loader.batch_size != 0:
                logger.warning(
                    "Please note that calibration sampling size {} "
                    "isn't divisible exactly by batch size {}. "
                    "So the real sampling size is {}.".format(
                        cfg_calib_sampling_size, data_loader.batch_size, data_loader.batch_size * cfg_calib_iter
                    )
                )
            calib_iterations = cfg_calib_iter
        return calib_iterations

    def _generate_qconfig(self, model, tune_cfg, quantize_params):
        tune_cfg = copy.deepcopy(tune_cfg)
        for node in model.graph.node:
            if (node.name, node.op_type) not in tune_cfg["op"]:
                continue
            scale_info = {}
            if quantize_params:
                for input_name in node.input:
                    if input_name in quantize_params:
                        scale_info[input_name] = quantize_params[input_name]
                for output_name in node.output:
                    if output_name in quantize_params:
                        scale_info[output_name] = quantize_params[output_name]
            tune_cfg["op"][(node.name, node.op_type)]["scale_info"] = scale_info
        fwk_info = {}
        fwk_info["approach"] = "post_training_static_quant" if self.static else "post_training_dynamic_quant"
        fwk_info["format"] = self.format
        fwk_info["backend"] = ONNXRT_BACKENDS[self.backend]
        fwk_info["workspace_path"] = self.work_space
        fwk_info["recipes"] = self.recipes
        fwk_info["domain"] = self.domain
        fwk_info["device"] = self.device
        tune_cfg["framework_specific_info"] = fwk_info
        return tune_cfg

    @dump_elapsed_time("Pass recover model")
    def recover(self, model, q_config):
        """Execute the recover process on the specified model.

        Args:
            model (object):  model need to do quantization.
            q_config (dict): recover configuration

        Returns:
            (dict): quantized model
        """
        self._pre_optimize(model)
        model = self.pre_optimized_model

        ort_version = Version(ort.__version__)
        if ort_version < ONNXRT152_VERSION:  # pragma: no cover
            logger.warning("Quantize input needs onnxruntime 1.5.2 or newer.")
            return model
        if model.model.opset_import[0].version < 11:  # pragma: no cover
            logger.warning("Quantize input needs model opset 11 or newer.")
        if ort_version < ONNXRT170_VERSION and self.format == "qdq":
            logger.error("QDQ mode needs onnxruntime1.7.0 or newer.")
            exit(0)
        if self.backend == "DnnlExecutionProvider" and any(
            [i.domain in ["", "ai.onnx"] and i.version < 15 for i in model.model.opset_import]
        ):  # pragma: no cover
            from onnx import version_converter

            try:
                model = self._rename_node(ONNXModel(version_converter.convert_version(model.model, 15)))
            except:
                logging.warning(
                    "Fail to upgrade model opset_import to >= 15, "
                    "please upgrade it manually to run with bf16 data type"
                )
                exit(0)

        from neural_compressor.adaptor.ox_utils.util import QuantizationMode

        if self.format == "qlinearops":
            format = QuantizationMode.QLinearOps
        elif self.format == "qdq":
            format = "qdq"
        else:
            format = QuantizationMode.IntegerOps

        self.quantizable_ops = self._query_quantizable_ops(model.model)
        quantize_params, tune_cfg = self._parse_qconfig(q_config)
        quantize_config = self._cfg_to_quantize_config(tune_cfg)

        if self._need_smooth_quant(tune_cfg):
            logger.error("Don't support to recover quantized model with smooth quant from original fp32 model.")
            exit(0)

        if self.recipes.get("layer_wise_quant", False) and not self.dynamic:
            # layer-wise quantization
            # details refer to docs/source/quantization_weight_only.md#layer-wise-quantization
            _model_to_split = copy.deepcopy(model)

            split_nodes = _model_to_split.find_split_nodes()
            logger.info(
                "Will split model into {} parts to do layer-wise quantization".format(
                    len([node.name for node in split_nodes]) + 1
                )
            )
            logger.debug(
                "Will split model with these nodes for layer-wise quantization: {}".format(
                    [node.name for node in split_nodes]
                )
            )

            split_idx = 1
            model_to_split = [_model_to_split]
            quantized_model_merged = None

            while len(model_to_split) != 0:
                split_model = model_to_split.pop(0)
                split_node = split_nodes.pop(0)
                save_both_split_models = True if len(split_nodes) == 0 else False
                shape_infer = True if split_idx == 1 else False

                # split model with given split_node
                split_model_part_1, split_model_part_2 = split_model.split_model_with_node(
                    split_node.name, model.model_path, shape_infer, save_both_split_models
                )
                if not save_both_split_models:
                    # append split_model_part_2 to do next split
                    model_to_split.append(split_model_part_2)

                logger.info("Quantize split model {}".format(split_idx))

                # quantize split model
                quantized_model_merged = self._quantize_split_model(
                    split_model_part_1, quantize_config, quantize_params, quantized_model_merged
                )

                split_idx += 1

                # if this is the last split, then quantize the last split model
                if save_both_split_models:
                    logger.info("Quantize split model {}".format(split_idx))

                    # quantize split model
                    quantized_model_merged = self._quantize_split_model(
                        split_model_part_2, quantize_config, quantize_params, quantized_model_merged
                    )
                    quantized_model_merged.re_org_output(model.output())  # re-org output as the origin output

            model.model = quantized_model_merged.model
            self._dump_model_op_stats(model)
            model.check_is_large_model()

        else:
            model = self._quantize_model(model, quantize_config, quantize_params)

        self._dump_model_op_stats(model)
        return model

    def _parse_qconfig(self, q_config):
        quantize_params = {}
        tune_cfg = {}
        for k, v in q_config.items():
            if k == "op":
                tune_cfg["op"] = {}
                for op_name_type, op_info in v.items():
                    node_dict = {}
                    for info_name, info_content in op_info.items():
                        if info_name != "scale_info":
                            node_dict[info_name] = info_content
                        else:
                            for tensor_name, param in info_content.items():
                                quantize_params[tensor_name] = param
                    tune_cfg["op"][op_name_type] = node_dict
            else:
                tune_cfg[k] = v
        if len(quantize_params) == 0:
            quantize_params = None
        return quantize_params, tune_cfg

    def _dump_model_op_stats(self, model):
        fp32_op_list = []
        for precision in self.query_handler.get_precisions():
            if precision != "fp32":
                fp32_op_list += self.query_handler.get_op_types_by_precision(precision=precision)
        qdq_ops = ["QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"]
        res = {}
        for op_type in fp32_op_list:
            res[op_type] = {"INT8": 0, "BF16": 0, "FP16": 0, "FP32": 0}
        for op_type in qdq_ops:
            res[op_type] = {"INT8": 0, "BF16": 0, "FP16": 0, "FP32": 0}

        for node in model.model.graph.node:
            if node.name.endswith("_quant"):
                if node.op_type.startswith("QLinear"):
                    origin_op_type = node.op_type.split("QLinear")[-1]
                else:
                    origin_op_type = node.op_type.split("Integer")[0]

                if origin_op_type in ["QAttention", "QGemm"]:
                    origin_op_type = origin_op_type[1:]
                elif origin_op_type == "DynamicQuantizeLSTM":
                    origin_op_type = "LSTM"
                elif origin_op_type == "QEmbedLayerNormalization":
                    origin_op_type = "EmbedLayerNormalization"
                res[origin_op_type]["INT8"] += 1

            elif node.op_type in qdq_ops:
                res[node.op_type]["INT8"] += 1

            elif node.op_type in fp32_op_list and node.name in self.quantize_config:
                if self.quantize_config[node.name] not in self.query_handler.get_fallback_list():
                    res[node.op_type]["FP32"] += 1
                else:
                    res[node.op_type][self.quantize_config[node.name].upper()] += 1

            elif node.op_type in res:
                res[node.op_type]["FP32"] += 1

        field_names = ["Op Type", "Total", "INT8", "BF16", "FP16", "FP32"]
        output_data = [
            [
                op_type,
                sum(res[op_type].values()),
                res[op_type]["INT8"],
                res[op_type]["BF16"],
                res[op_type]["FP16"],
                res[op_type]["FP32"],
            ]
            for op_type in res.keys()
        ]

        Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()
        self.optype_statistics = field_names, output_data

    def _get_quantize_params(self, model, data_loader, quantize_config, iterations, **kwargs):
        from neural_compressor.adaptor.ox_utils.calibration import ONNXRTAugment
        from neural_compressor.model.onnx_model import ONNXModel

        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)
        black_nodes = [node for node in quantize_config if quantize_config[node] == "fp32"]
        white_nodes = [node for node in quantize_config if quantize_config[node] != "fp32"]

        augment = ONNXRTAugment(
            model,
            data_loader,
            self.quantizable_op_types,
            black_nodes=black_nodes,
            white_nodes=white_nodes,
            iterations=list(range(0, iterations)),
            backend=self.backend,
            reduce_range=self.reduce_range,
            **kwargs,
        )
        self.min_max = augment.dump_minmax(quantize_config)
        quantize_params = augment.dump_calibration(quantize_config, min_max=self.min_max)
        dataloder_for_next_split_model = augment.dataloder_for_next_split_model
        return quantize_params, dataloder_for_next_split_model

    def inspect_tensor(
        self,
        model,
        dataloader,
        op_list=[],
        iteration_list=[],
        inspect_type="activation",
        save_to_disk=False,
        save_path=None,
        quantization_cfg=None,
    ):
        """The function is used by tune strategy class for dumping tensor info."""
        from neural_compressor.adaptor.ox_utils.calibration import ONNXRTAugment
        from neural_compressor.utils.utility import dump_data_to_local

        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)

        if len(op_list) > 0 and isinstance(op_list, KeysView):
            op_list = [item[0] for item in op_list]
        augment = ONNXRTAugment(
            model, dataloader, [], iterations=iteration_list, white_nodes=op_list, backend=self.backend
        )
        tensors = augment.dump_tensor(
            activation=(inspect_type != "weight"), weight=(inspect_type != "activation"), format=self.format
        )
        if save_to_disk:
            if not save_path:
                save_path = self.work_space
            dump_data_to_local(tensors, save_path, "inspect_result.pkl")
        return tensors

    def set_tensor(self, model, tensor_dict):
        from onnx import numpy_helper

        from neural_compressor.adaptor.ox_utils.util import quantize_data_per_channel, quantize_data_with_scale_zero
        from neural_compressor.model.onnx_model import ONNXModel

        if not isinstance(model, ONNXModel):
            model = ONNXModel(model)
        assert "QuantizeLinear" in [
            node.op_type for node in model.model.graph.node
        ], "adaptor.set_tensor only accept int8 model"
        input_name_to_nodes = model.input_name_to_nodes
        for tensor_name, tensor_value in tensor_dict.items():
            if not tensor_name.endswith("_quantized"):
                tensor_name += "_quantized"
            not_filter = False
            scale_tensor, zo_tensor = model.get_scale_zero(tensor_name)
            if scale_tensor is None or zo_tensor is None:
                not_filter = True
            else:
                scale_value = numpy_helper.to_array(scale_tensor)
                zo_value = numpy_helper.to_array(zo_tensor)
            assert (
                len(input_name_to_nodes[tensor_name]) == 1
            ), "quantized filter weight should be input of only one node"
            node = input_name_to_nodes[tensor_name][0]  # TBD only for conv bias
            node_name = node.name.replace("_quant", "")
            assert node_name in self.quantize_config
            q_type = self.quantize_config[node_name]["weight"]["dtype"]
            if not_filter:
                new_tensor_value = self._requantize_bias(model, tensor_name, tensor_value)
            elif self.quantize_config[node_name]["weight"]["granularity"] == "per_tensor":
                new_tensor_value = quantize_data_with_scale_zero(
                    tensor_value, q_type, self.quantize_config[node_name]["weight"]["scheme"], scale_value, zo_value
                )
            elif (Version(ort.__version__) >= ONNXRT112_VERSION and model.model.opset_import[0].version < 13) and len(
                scale_tensor.dims
            ) in [1, 2]:
                logger.warning(
                    "Skip setting per-channel quantized tensor {}, please "
                    "use onnxruntime < 1.12.0 or upgrade model opset version to 13 or "
                    "higher".format(tensor_name)
                )
                return model
            else:
                axis = (
                    tuple(range(1, len(tensor_value.shape)))
                    if tensor_value.shape.index(scale_value.shape[0]) == 0
                    else tuple(range(0, len(tensor_value.shape) - 1))
                )
                new_tensor_value = quantize_data_with_scale_zero(
                    tensor_value,
                    q_type,
                    self.quantize_config[node_name]["weight"]["scheme"],
                    np.expand_dims(scale_value, axis=axis),
                    np.expand_dims(zo_value, axis=axis),
                )
            model.set_initializer(tensor_name, new_tensor_value)
        return model

    def _requantize_bias(self, model, bias_name, bias_data):
        """Helper function to requantize bias, borrowed from onnx_quantizer."""
        from onnx import numpy_helper

        node = model.input_name_to_nodes[bias_name][0]
        input_scale_name = node.input[1]
        input_scale = numpy_helper.to_array(model.get_initializer(input_scale_name))

        weight_scale_name = node.input[4]
        weight_scale = numpy_helper.to_array(model.get_initializer(weight_scale_name))

        bias_scale = input_scale * weight_scale
        new_bias_data = (bias_data / bias_scale).round().astype(np.int32)
        return new_bias_data

    def _detect_domain(self, model):
        """Automatically detect whether the model belongs to NLP domain.

        Args:
            model (ONNXModel): ONNXModel wrapped model

        Returns:
            bool: the model belongs to NLP domain or not
        """
        is_nlp = False
        # 1. according to initializer names
        initializer_names = [init.name for init in model.model.graph.initializer]
        pattern = ".*word.*embedding.*"
        for name in initializer_names:
            obj = re.findall(pattern, name)
            if len(obj) > 0:
                is_nlp = True
                break

        # 2. according to input
        # typically, NLP models have multiple inputs,
        # and the dimension of each input is usually 2 (batch_size, max_seq_len)
        input_shape_lens = [len(inp.type.tensor_type.shape.dim) for inp in model.model.graph.input]
        if len(input_shape_lens) > 1 and all(shape_len == 2 for shape_len in input_shape_lens):
            is_nlp = True

        # 3. according to attention structure
        qkv = model.find_qkv_in_attention()
        if len(qkv) != 0:
            is_nlp = True

        # 4. according to LSTM/Attention optype
        op_types = [node.op_type for node in model.model.graph.node]
        if "LSTM" in op_types or "Attention" in op_types:
            is_nlp = True

        logger.warning(
            "The model is automatically detected as {} model. "
            "You can use 'domain' argument in 'PostTrainingQuantConfig' "
            "to overwrite it".format("an NLP" if is_nlp else "a non-NLP")
        )
        return is_nlp

    def _pre_optimize(self, model, level=1):
        # the pre-optimization may already done at the smoothing process
        # pre_optimize -> sq -> update the pre_optimized_model
        if self.pre_optimized_model:
            logger.info("Pre-optimization already done, return it directly.")
            return self.pre_optimized_model
        from neural_compressor import options
        from neural_compressor.adaptor.ox_utils.util import remove_init_from_model_input, split_shared_bias

        remove_init_from_model_input(model)
        sess_options = ort.SessionOptions()
        optimization_levels = {
            "DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        if not isinstance(self.query_handler.get_graph_optimization(), list):
            level = self.query_handler.get_graph_optimization()
        elif self.recipes.get("layer_wise_quant"):
            level = "ENABLE_BASIC"
            logger.info("Force set graph optimization level to 'ENABLE_BASIC' for layer-wise quantization")
        elif options.onnxrt.graph_optimization.level is not None:
            level = options.onnxrt.graph_optimization.level
        elif self.recipes.get("graph_optimization_level", None) is not None:
            level = self.recipes["graph_optimization_level"]
        else:
            if self.domain == "auto" and self._detect_domain(model):
                self.domain = "nlp"
            level = "ENABLE_EXTENDED" if self.domain == "nlp" else "ENABLE_BASIC"
            logger.warning(
                "Graph optimization level is automatically set to {}. "
                "You can use 'recipe' argument in 'PostTrainingQuantConfig'"
                "to overwrite it".format(level)
            )
        sess_options.graph_optimization_level = optimization_levels[level]
        sess_options.optimized_model_filepath = os.path.join(self.work_space, "Optimized_model.onnx")
        if model.is_large_model and self.recipes.get("layer_wise_quant", False):
            # save the model and external data for layer-wise quantization
            external_data_filename = os.path.basename(sess_options.optimized_model_filepath) + "_data"
            external_data_file_threshold = 1024
            sess_options.add_session_config_entry(
                "session.optimized_model_external_initializers_file_name", external_data_filename
            )
            sess_options.add_session_config_entry(
                "session.optimized_model_external_initializers_min_size_in_bytes", str(external_data_file_threshold)
            )
            logger.info("Saving optimized model for layer-wise quantization. This may take a while...")

        if sys.version_info < (3, 11) and find_spec("onnxruntime_extensions"):  # pragma: no cover
            from onnxruntime_extensions import get_library_path

            sess_options.register_custom_ops_library(get_library_path())

        if not model.is_large_model:
            sess = ort.InferenceSession(model.model.SerializeToString(), sess_options, providers=[self.backend])
        elif model.model_path is not None:  # pragma: no cover
            model.model = onnx.ModelProto()  # clean memory for large model
            sess = ort.InferenceSession(model.model_path, sess_options, providers=[self.backend])
        else:  # pragma: no cover
            logger.warning("Please use model path instead of onnx model object to quantize")
        del sess
        tmp_model = onnx.load(sess_options.optimized_model_filepath, load_external_data=False)

        if model.is_large_model:
            if not self.performance_only:
                # save the large model to workspace if acc tuning is required
                from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model

                # load external data
                load_external_data_for_model(tmp_model, os.path.split(model.model_path)[0])

                # if optimized model exists, remove it
                if os.path.isfile(sess_options.optimized_model_filepath):
                    os.remove(sess_options.optimized_model_filepath)

                # if the model if Tranformer-based, save hf config to workspace
                if model.hf_config is not None:
                    model_type = (
                        "" if not hasattr(model.hf_config, "model_type") else getattr(model.hf_config, "model_type")
                    )
                    setattr(model.hf_config.__class__, "model_type", model_type)
                    output_config_file = Path(self.work_space).joinpath("config.json").as_posix()
                    if not os.path.exists(output_config_file):
                        model.hf_config.to_json_file(output_config_file, use_diff=False)

                # save model and external data
                model_name = os.path.split(model.model_path)[-1]
                model_path = os.path.join(self.work_space, model_name)
                data_name = model_name + "_data"

                convert_model_to_external_data(
                    tmp_model,
                    all_tensors_to_one_file=True,
                    location=data_name,
                    size_threshold=1024,
                    convert_attribute=False,
                )
                onnx.save_model(tmp_model, model_path)
                model.model_path = model_path
            else:
                if not self.recipes.get("layer_wise_quant", False):
                    # load external data if layer-wise quant is False
                    from onnx.external_data_helper import load_external_data_for_model

                    load_external_data_for_model(tmp_model, os.path.split(model.model_path)[0])
                model.model_path = sess_options.optimized_model_filepath
        else:
            model.model_path = sess_options.optimized_model_filepath

        model.model = (
            self._replace_gemm_with_matmul(tmp_model).model
            if options.onnxrt.graph_optimization.gemm2matmul and self.recipes.get("gemm_to_matmul", True)
            else tmp_model
        )
        model = self._rename_node(model)
        model = self._revert_fusedconv(model)
        if self.backend == "TensorrtExecutionProvider":
            model = self._revert_conv_add_fusion(model)
        model = split_shared_bias(model)
        model.topological_sort()
        self.pre_optimized_model = model

    def _revert_conv_add_fusion(self, model):
        from onnx import numpy_helper

        from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg

        add_nodes = []
        remove_nodes = []
        for node in model.model.graph.node:
            if node.op_type == "Conv" and len(node.input) == 3:
                bias_tensor = model.get_initializer(node.input[2])
                bias_array = numpy_helper.to_array(bias_tensor).reshape((-1, 1, 1))
                model.remove_initializer(bias_tensor)
                model.add_initializer(numpy_helper.from_array(bias_array, bias_tensor.name))
                kwargs = {}
                activation_params = None
                for attr in node.attribute:
                    kwargs.update(attribute_to_kwarg(attr))
                conv = onnx.helper.make_node("Conv", node.input[0:2], [node.name + "_revert"], node.name, **kwargs)
                add = onnx.helper.make_node("Add", [conv.output[0], node.input[2]], node.output, node.name + "_add")
                add_nodes.extend([conv, add])

        model.remove_nodes(remove_nodes)
        model.add_nodes(add_nodes)
        model.update()
        return model

    def _revert_fusedconv(self, model):
        from onnx import onnx_pb as onnx_proto

        from neural_compressor.adaptor.ox_utils.util import attribute_to_kwarg

        new_nodes = []
        remove_nodes = []
        for node in model.model.graph.node:
            if node.op_type == "FusedConv":
                kwargs = {}
                activation_params = None
                for attr in node.attribute:
                    if attr.name == "activation":
                        activation_type = attr.s.decode("utf-8")
                    elif attr.name == "activation_params":
                        continue
                    else:
                        kwargs.update(attribute_to_kwarg(attr))
                if activation_type in ["Relu", "Clip"]:
                    continue
                conv = onnx.helper.make_node("Conv", node.input, [node.name], node.name.split("fused ")[-1], **kwargs)
                activation_input = conv.output

                activation = onnx.helper.make_node(
                    activation_type, conv.output, node.output, "_".join((conv.name, activation_type))
                )
                new_nodes.extend([conv, activation])
                remove_nodes.append(node)
        model.model.graph.node.extend(new_nodes)
        for node in remove_nodes:
            model.model.graph.node.remove(node)
        model.update()
        return model

    def _rename_node(self, model_wrapper):
        model = model_wrapper.model
        node_names = [i.name for i in model.graph.node]
        if len(set(node_names)) < len(node_names):
            logger.warning(
                "This model has nodes with the same name, please check"
                "renamed_model.onnx in workspace_path (default is nc_workspace)"
                "for newly generated node name"
            )
            for idx, node in enumerate(model.graph.node):
                if node_names.count(node.name) > 1:
                    node.name = node.op_type + "_nc_rename_" + str(idx)
            if model_wrapper.is_large_model:
                onnx.save(
                    model,
                    os.path.join(self.work_space, "renamed_model.onnx"),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="weights.pb",
                    convert_attribute=False,
                )
            else:
                onnx.save(model, os.path.join(self.work_space, "renamed_model.onnx"))
        return model_wrapper

    @staticmethod
    def _replace_gemm_with_matmul(model):
        new_nodes = []
        from onnx import numpy_helper

        if not isinstance(model, ONNXModel):
            model = ONNXModel(model, ignore_warning=True)

        for node in model.nodes():
            if node.op_type == "Gemm":
                alpha = 1.0
                beta = 1.0
                transA = 0
                transB = 0
                for attr in node.attribute:
                    if attr.name == "alpha":
                        alpha = onnx.helper.get_attribute_value(attr)
                    elif attr.name == "beta":
                        beta = onnx.helper.get_attribute_value(attr)
                    elif attr.name == "transA":
                        transA = onnx.helper.get_attribute_value(attr)
                    elif attr.name == "transB":
                        transB = onnx.helper.get_attribute_value(attr)
                if alpha == 1.0 and beta == 1.0 and transA == 0:
                    inputB = node.input[1]
                    if transB == 1:
                        B = model.get_initializer(node.input[1])
                        if B:
                            # assume B is not used by any other node
                            B_array = numpy_helper.to_array(B)
                            B_trans = numpy_helper.from_array(B_array.T)
                            B_trans.name = B.name
                            model.remove_initializer(B)
                            model.add_initializer(B_trans)

                            # TBD this is for onnx model zoo, which are all in old IR version
                            if model.model.ir_version < 4:
                                for input in model.model.graph.input:
                                    if input.name == B_trans.name:
                                        for i, dim in enumerate(input.type.tensor_type.shape.dim):
                                            dim.dim_value = B_array.T.shape[i]

                        else:
                            inputB += "_Transposed"
                            transpose_node = onnx.helper.make_node(
                                "Transpose", inputs=[node.input[1]], outputs=[inputB], name=node.name + "_Transpose"
                            )
                            new_nodes.append(transpose_node)

                    matmul_node = onnx.helper.make_node(
                        "MatMul",
                        inputs=[node.input[0], inputB],
                        outputs=[node.output[0] + ("_MatMul" if len(node.input) > 2 else "")],
                        name=node.name + "_MatMul",
                    )
                    new_nodes.append(matmul_node)

                    if len(node.input) > 2:
                        add_node = onnx.helper.make_node(
                            "Add",
                            inputs=[node.output[0] + "_MatMul", node.input[2]],
                            outputs=node.output,
                            name=node.name + "_Add",
                        )
                        new_nodes.append(add_node)

                # unsupported
                else:
                    new_nodes.append(node)

            # not GEMM
            else:
                new_nodes.append(node)

        model.graph().ClearField("node")
        model.graph().node.extend(new_nodes)

        return model

    def query_fw_capability(self, model):
        """The function is used to query framework capability.
        TODO: will be replaced by framework query API

        Args:
            model: onnx model

        Returns:
            (dict): quantization capability
        """
        # optype_wise and op_wise capability
        self._pre_optimize(model)
        recipes_ops = {}
        recipes_ops["first_conv_or_matmul_quantization"] = []
        recipes_ops["last_conv_or_matmul_quantization"] = []
        recipes_ops["pre_post_process_quantization"] = []
        exclude_first_quantizable_op = (
            True
            if "first_conv_or_matmul_quantization" in self.recipes
            and not self.recipes["first_conv_or_matmul_quantization"]
            else False
        )
        exclude_last_quantizable_op = (
            True
            if "last_conv_or_matmul_quantization" in self.recipes
            and not self.recipes["last_conv_or_matmul_quantization"]
            else False
        )
        exclude_pre_post_process = (
            True
            if "pre_post_process_quantization" in self.recipes and not self.recipes["pre_post_process_quantization"]
            else False
        )

        quantizable_optype = set([i.op_type for i in self.pre_optimized_model.nodes()])
        optype_wise = OrderedDict()
        op_wise = OrderedDict()
        for query in [self.query_handler, self.query_handler_ext]:
            if query is None:
                continue
            precisions = query.get_precisions()

            for precision in precisions:
                if precision == "fp16" and not self.use_fp16:
                    continue
                if precision == "bf16" and (
                    not self.use_bf16 or (not CpuInfo().bf16 and os.getenv("FORCE_BF16") != "1")
                ):
                    continue
                elif precision == "weight_only_integer":
                    continue
                # get supported optype for target precision
                optypes = (
                    query.get_op_types_by_precision(precision)
                    if query.get_op_types_by_precision(precision) != ["*"]
                    else optype_wise.keys()
                )

                configs = (
                    query.get_quantization_capability()[precision]
                    if precision in query.get_quantization_capability()
                    else {"default": {"weight": {"dtype": precision}, "activation": {"dtype": precision}}}
                )

                if self.backend == "TensorrtExecutionProvider" and precision not in query.get_fallback_list():
                    optypes.append("Add")

                for op in optypes:
                    if op not in quantizable_optype:
                        continue
                    if op not in configs:
                        if "default" in configs:
                            op_capability = copy.deepcopy(configs["default"])
                        else:
                            continue
                    else:
                        op_capability = copy.deepcopy(configs[op])

                    if precision in ["int8", "uint8"]:
                        if self.static:
                            op_capability["activation"]["quant_mode"] = "static"
                        elif self.dynamic:
                            op_capability["activation"]["quant_mode"] = "dynamic"
                        elif query == self.query_handler:  # query static capability for auto
                            op_capability["activation"]["quant_mode"] = "static"
                        elif query == self.query_handler_ext:  # query dynamic capability for auto
                            op_capability["activation"]["quant_mode"] = "dynamic"

                    if op not in optype_wise.keys():
                        optype_wise[op] = [op_capability]
                    elif op_capability not in optype_wise[op]:
                        optype_wise[op].append(op_capability)

        if self.format == "qdq":
            self._optypewise_filter_for_qdq(optype_wise)

        first_quantizable_node = []
        last_quantizable_node = []
        all_conv_matmul = []
        attention_matmul = []
        for _, node in enumerate(self.pre_optimized_model.nodes()):
            if node.op_type in ["Conv", "MatMul", "Attention"]:
                if node.op_type in optype_wise:
                    # get first Conv or MatMul node
                    if len(first_quantizable_node) == 0:
                        first_quantizable_node.append(node)

                    # get last Conv or MatMul node
                    if len(last_quantizable_node) != 0:
                        last_quantizable_node.pop()
                    last_quantizable_node.append(node)

                all_conv_matmul.append(node)
                if node.op_type != "Conv":
                    attention_matmul.append(node)

        if len(first_quantizable_node) != 0:
            recipes_ops["first_conv_or_matmul_quantization"] = [
                (first_quantizable_node[0].name, first_quantizable_node[0].op_type)
            ]
        if len(last_quantizable_node) != 0:
            recipes_ops["last_conv_or_matmul_quantization"] = [
                (last_quantizable_node[0].name, last_quantizable_node[0].op_type)
            ]

        ffn_matmul = []
        attention_matmul_optype = [node.op_type for node in attention_matmul]
        # find matmul ops in feed forward network (FFN) structure which mainly in transformers based NLP models
        if len(attention_matmul) > 0 and "Attention" in attention_matmul_optype:
            # model is optimized and Attention is fused,
            # index of Attention is used as split to find FFN MatMul
            first_attention_index = attention_matmul_optype.index("Attention")
            attention_matmul_optype = attention_matmul_optype[first_attention_index:]
            attention_index = list(np.where(np.array(attention_matmul_optype) == "Attention")[0])
            block_len = attention_index[1] - attention_index[0] if len(attention_index) > 2 else 4
            ffn_matmul = self.pre_optimized_model.find_ffn_matmul(
                attention_index, attention_matmul[first_attention_index:], block_len
            )

            # in case there are unfused Attentions
            qkv = self.pre_optimized_model.find_qkv_in_attention(find_all=True)
            if len(qkv) != 0:
                attention_starts = [nodes[0] for nodes in qkv]
                attention_index = [
                    np.where(np.array([n.name for n in attention_matmul]) == attention_start)[0].tolist()[0]
                    for attention_start in attention_starts
                ]
                block_len = attention_index[1] - attention_index[0] if len(attention_index) > 2 else 4
                for matmul in self.pre_optimized_model.find_ffn_matmul(attention_index, attention_matmul, block_len):
                    if matmul not in ffn_matmul:
                        ffn_matmul.append(matmul)
        else:
            # model is not optimized or Attention isn't fused,
            # query MatMul, key MatMul and value MatMul are used as split to find FFN MatMul
            qkv = self.pre_optimized_model.find_qkv_in_attention(find_all=True)
            if len(qkv) != 0:
                attention_starts = [nodes[0] for nodes in qkv]
                attention_index = [
                    np.where(np.array([n.name for n in attention_matmul]) == attention_start)[0].tolist()[0]
                    for attention_start in attention_starts
                ]
                block_len = attention_index[1] - attention_index[0] if len(attention_index) > 2 else 4
                ffn_matmul = self.pre_optimized_model.find_ffn_matmul(attention_index, attention_matmul, block_len)

        block_wise = []
        for block in reversed(ffn_matmul):
            node_info = []
            for node in block:
                node_info.append((node.name, node.op_type))
            if len(node_info) != 0:
                block_wise.append(node_info)

        for _, node in enumerate(self.pre_optimized_model.nodes()):
            # for TRT EP, only insert Q/DQ to inputs of Add nodes followed by ReduceMean
            if node.op_type == "Add" and self.backend == "TensorrtExecutionProvider":
                children = self.pre_optimized_model.get_children(node)
                if "ReduceMean" not in [i.op_type for i in children]:
                    op_wise.update(
                        {(node.name, node.op_type): [{"weight": {"dtype": "fp32"}, "activation": {"dtype": "fp32"}}]}
                    )
                    continue

            if node.op_type in optype_wise:
                if (exclude_first_quantizable_op and node in first_quantizable_node) or (
                    exclude_last_quantizable_op and node in last_quantizable_node
                ):
                    tmp_cfg = copy.deepcopy(optype_wise[node.op_type])
                    tmp_cfg = list(filter(lambda x: "quant_mode" not in x["activation"], tmp_cfg))
                    op_wise.update({(node.name, node.op_type): tmp_cfg})
                    continue
                op_wise.update({(node.name, node.op_type): copy.deepcopy(optype_wise[node.op_type])})

        # only when first and last quantizable nodes are found and they are not the same,
        # fallback pre/postprocess ops
        if (
            len(first_quantizable_node) != 0
            and len(last_quantizable_node) != 0
            and first_quantizable_node[0].name != last_quantizable_node[0].name
        ):
            # get backbone nodes
            from collections import deque

            # get nodes between first quantizable node and last quantizable node
            backbone_queue = deque(last_quantizable_node)
            backbone_nodes = self.pre_optimized_model.get_nodes_chain(backbone_queue, first_quantizable_node)

            # get extra Conv or MatMul nodes not between first quantizable node and last quantizable node
            backbone_queue_extra = deque()
            for conv_or_matmul in all_conv_matmul:
                if conv_or_matmul.name not in backbone_nodes:
                    backbone_queue_extra.append(conv_or_matmul)
                    backbone_nodes = self.pre_optimized_model.get_nodes_chain(
                        backbone_queue_extra, first_quantizable_node, backbone_nodes
                    )
            backbone_nodes += [i.name for i in first_quantizable_node]

            for _, node in enumerate(self.pre_optimized_model.nodes()):
                if node.name not in backbone_nodes and node.op_type in optype_wise:
                    recipes_ops["pre_post_process_quantization"].append((node.name, node.op_type))
            if exclude_pre_post_process:
                for _, node in enumerate(self.pre_optimized_model.nodes()):
                    if node.op_type in optype_wise:
                        # nodes not in backbone are not quantized
                        if node.name not in backbone_nodes:
                            tmp_cfg = copy.deepcopy(optype_wise[node.op_type])
                            tmp_cfg = list(filter(lambda x: "quant_mode" not in x["activation"], tmp_cfg))
                            op_wise.update({(node.name, node.op_type): tmp_cfg})
                            continue
                        if (node.name, node.op_type) in op_wise:
                            op_wise.update(
                                {(node.name, node.op_type): copy.deepcopy(op_wise[(node.name, node.op_type)])}
                            )
                        else:  # pragma: no cover
                            op_wise.update({(node.name, node.op_type): copy.deepcopy(optype_wise[node.op_type])})

        return {"optypewise": optype_wise, "opwise": op_wise, "recipes_ops": recipes_ops, "block_wise": block_wise}

    def _optypewise_filter_for_qdq(self, optype_wise):
        """Filter optypes that don't support per_channel in QDQ format.

        Args:
            optype_wise (dict): optype and quantization config
        Returns:
            dict: filtered optype and quantization config
        """
        supported_perchannel_optypes = {
            "1.6.0": ["Conv", "Gather"],
            "1.7.0": ["Conv", "Gather"],
            "1.8.0": ["Conv", "Gather"],
            "1.9.0": ["Conv", "Gather"],
            "1.10.0": ["Conv", "Gather", "MatMul"],
            "1.11.0": ["Conv", "Gather", "MatMul", "Gemm"],
            "1.12.0": ["Conv", "Gather", "MatMul", "Gemm"],
        }
        specific_cfg_version = self.query_handler.get_specific_cfg_version()
        if Version(specific_cfg_version) > ONNXRT112_VERSION:
            specific_cfg_version = "1.12.0"
        for optype, caps in optype_wise.items():
            if optype not in supported_perchannel_optypes[specific_cfg_version]:
                for cap in caps:
                    if "mode" in cap and cap["mode"] == "QDQ" and "per_channel" in cap["weight"]["granularity"]:
                        cap["weight"]["granularity"].remove("per_channel")
        return optype_wise

    def _cfg_to_quantize_config(self, tune_cfg):
        quantize_config = {}
        quantize_config["calib_iteration"] = tune_cfg["calib_iteration"]
        granularity = "per_tensor"
        algorithm = "minmax"

        from onnx import onnx_pb as onnx_proto

        for _, op in enumerate(self.quantizable_ops):
            if (op.name, op.op_type) not in tune_cfg["op"]:
                continue
            if tune_cfg["op"][(op.name, op.op_type)]["activation"]["dtype"] in self.query_handler.get_fallback_list():
                quantize_config[op.name] = tune_cfg["op"][(op.name, op.op_type)]["activation"]["dtype"]
            else:
                node_config = copy.deepcopy(tune_cfg["op"][(op.name, op.op_type)])
                for tensor, config in tune_cfg["op"][(op.name, op.op_type)].items():
                    if "granularity" not in config:
                        node_config[tensor]["granularity"] = granularity
                    if "algorithm" not in config:
                        node_config[tensor]["algorithm"] = algorithm
                    if config["dtype"] == "int8":
                        node_config[tensor]["dtype"] = onnx_proto.TensorProto.INT8
                        if "scheme" not in config:
                            node_config[tensor]["scheme"] = "sym"
                    else:
                        node_config[tensor]["dtype"] = onnx_proto.TensorProto.UINT8
                        if "scheme" not in config:
                            node_config[tensor]["scheme"] = "asym"
                quantize_config[op.name] = node_config

        return quantize_config

    def _query_quantizable_ops(self, model):
        for node in model.graph.node:
            if node.op_type in self.quantizable_op_types and node not in self.quantizable_ops:
                self.quantizable_ops.append(node)

        return self.quantizable_ops

    def _query_quantizable_op_types(self):
        quantizable_op_types = self.query_handler.get_op_types_by_precision(precision="int8")
        return quantizable_op_types

    def evaluate(
        self,
        input_graph,
        dataloader,
        postprocess=None,
        metrics=None,
        measurer=None,
        iteration=-1,
        tensorboard=False,
        fp32_baseline=False,
    ):
        """The function is for evaluation if no given eval func.

        Args:
            input_graph      : onnx model for evaluation
            dataloader       : dataloader for evaluation. neural_compressor.data.dataloader.ONNXDataLoader
            postprocess      : post-process for evaluation. neural_compressor.data.transform.ONNXTransforms
            metrics:         : metrics for evaluation. neural_compressor.metric.ONNXMetrics
            measurer         : neural_compressor.objective.Measurer
            iteration(int)   : max iterations of evaluaton.
            tensorboard(bool): whether to use tensorboard for visualization
            fp32_baseline (boolean, optional): only for compare_label=False pipeline

        Returns:
            (float) evaluation results. acc, f1 e.g.
        """
        if input_graph.is_large_model:  # pragma: no cover
            onnx.save_model(
                input_graph.model,
                self.work_space + "eval.onnx",
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="weights.pb",
                convert_attribute=False,
            )
        sess_options = ort.SessionOptions()
        if self.backend == "TensorrtExecutionProvider":
            from neural_compressor.adaptor.ox_utils.util import trt_env_setup

            trt_env_setup(input_graph.model)
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if measurer:
            # https://github.com/microsoft/onnxruntime/issues/7347
            cores_per_instance = int(os.environ.get("CORES_PER_INSTANCE"))
            assert cores_per_instance > 0, "benchmark cores_per_instance should greater than 0"
            sess_options.intra_op_num_threads = cores_per_instance
        if sys.version_info < (3, 11) and find_spec("onnxruntime_extensions"):  # pragma: no cover
            from onnxruntime_extensions import get_library_path

            sess_options.register_custom_ops_library(get_library_path())
        session = (
            ort.InferenceSession(self.work_space + "eval.onnx", sess_options, providers=[self.backend])
            if input_graph.is_large_model
            else ort.InferenceSession(input_graph.model.SerializeToString(), sess_options, providers=[self.backend])
        )
        results = []
        if metrics:
            for metric in metrics:
                metric.reset()
            self.fp32_preds_as_label = any(
                [hasattr(metric, "compare_label") and not metric.compare_label for metric in metrics]
            )

        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]

        def eval_func(dataloader):
            ort_inputs = {}
            for idx, (inputs, labels) in enumerate(dataloader):
                if not isinstance(labels, list):
                    labels = [labels]

                if len_inputs == 1:
                    if isinstance(inputs, dict):
                        for name, input in inputs.items():
                            ort_inputs.update({name: to_numpy(input)})
                    else:
                        ort_inputs.update({inputs_names[0]: to_numpy(inputs)})
                else:
                    assert len_inputs == len(inputs), "number of input tensors must align with graph inputs"

                    if isinstance(inputs, dict):
                        for name, input in inputs.items():
                            ort_inputs.update({name: to_numpy(input)})
                    else:
                        ort_inputs = dict(zip(inputs_names, [to_numpy(i) for i in inputs]))

                if measurer is not None:
                    measurer.start()
                    predictions = session.run(None, ort_inputs)
                    measurer.end()
                else:
                    predictions = session.run(None, ort_inputs)

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

        if isinstance(dataloader, BaseDataLoader) and not self.benchmark:
            try:
                eval_func(dataloader)
            except Exception:  # pragma: no cover
                logger.warning("Fail to forward with batch size={}, set to {} now.".format(dataloader.batch_size, 1))
                dataloader.batch(1)
                eval_func(dataloader)
        else:  # pragma: no cover
            eval_func(dataloader)

        if self.fp32_preds_as_label:
            from neural_compressor.adaptor.ox_utils.util import collate_preds

            if fp32_baseline:
                results = collate_preds(self.fp32_results)
                reference = results
            else:
                reference = collate_preds(self.fp32_results)
                results = collate_preds(results)
            for metric in metrics:
                if hasattr(metric, "compare_label") and not metric.compare_label:
                    metric.update(results, reference)

        acc = 0 if metrics is None else [metric.result() for metric in metrics]
        return acc if not isinstance(acc, list) or len(acc) > 1 else acc[0]

    def save(self, model, path):
        """Save model.

        Args:
            model (ModelProto): model to save
            path (str): save path
        """
        model.save(os.path.join(path, "best_model.onnx"))

    def get_output_op_names(self, qmodel):
        """Get the output ops' names."""
        outputs = qmodel.output()
        output_op_names = []
        for output in outputs:
            output_op_names.append(qmodel.output_name_to_node[output].name)
        logger.debug(f"output op names: {output_op_names}")
        return output_op_names

    def calculate_op_sensitivity(
        self, model, dataloader, tune_cfg, output_op_names, confidence_batches, fallback=True, requantize_cfgs=None
    ):
        """Compute the op sensitivity.

        The sensitivity metric is the mse between the output of the last quantized op of
        the quantized model and the output of its corresponding op in the fp32 model.

          1. Backup the tune cfg
          2. Fallback each int8 op and compute its mse if use fallback (with 'fallback == True'),
            or re-quantize each fp32 op(fallen back in the previous stage) and compute its MSE if not.
          3. Sorted op name list according to its MSE

        Args:
          fp32_model: The fp32 model.
          dataloader: the dataloader with full dataset.
          tune_cfg: tuning config
          fallback: denote fallback stage or re-quantize stage
          requantize_cfgs: the dict of tuning configs for all re-quantizable ops

        Returns:
          A list of op names, sorted by its MSE sensitivity.
        """
        from copy import deepcopy

        fp32_op_cfg = {"activation": {"dtype": "fp32", "quant_mode": "fp32"}, "weight": {"dtype": "fp32"}}

        if fallback:
            ops_list = [
                op
                for op, config in tune_cfg["op"].items()
                if config["activation"]["quant_mode"] in ("static", "dynamic")
            ]
            replace_cfgs = {op: fp32_op_cfg for op in tune_cfg["op"]}
        else:
            ops_list = [
                op
                for op, config in tune_cfg["op"].items()
                if config["activation"]["quant_mode"] == "fp32" and op in requantize_cfgs
            ]
            replace_cfgs = requantize_cfgs

        # Step2. compute mse
        mse_result = self._get_mse_order(
            model, deepcopy(tune_cfg), replace_cfgs, ops_list, dataloader, output_op_names, confidence_batches
        )

        # Step3. sort
        mse_order = [op for op, _ in sorted(mse_result.items(), key=lambda i: i[1])]
        logger.debug("Dump MSE order:")
        for op in mse_order:
            logger.debug(f"{op}: {mse_result[op]}")
        return mse_order

    def _get_mse_order(
        self, fp32_model, tune_cfg, replace_cfgs, ops_lst, dataloader, output_op_names, confidence_batches
    ):
        """Compute MSE."""
        op_cfg = tune_cfg["op"]
        mse_result = {}

        fp32_output = self._inference_model_on_batches(
            fp32_model, tune_cfg, dataloader, output_op_names, confidence_batches
        )

        for op in ops_lst:
            # backup and set replace tuning config
            backup_cfg = op_cfg[op]
            op_cfg[op] = replace_cfgs[op]

            # quantize and inference the model
            q_model = self.quantize(tune_cfg, fp32_model, dataloader)
            q_output = self._inference_model_on_batches(
                q_model, tune_cfg, dataloader, output_op_names, confidence_batches
            )

            mse_result[op] = self._calculate_mse(fp32_output, q_output)

            # recover tune_cfg
            op_cfg[op] = backup_cfg

        return mse_result

    def _calculate_mse(self, fp32_output, q_output):
        """MSE calculation."""
        result = []
        for i, j in zip(fp32_output, q_output):
            result.append(np.square(i - j).mean())
        return np.array(result).mean()

    def _inference_model_on_batches(self, model, tune_cfg, dataloader, output_op_name, iterations):
        """Inference model on batches."""
        ort_inputs = {}
        predictions = []

        session = (
            ort.InferenceSession(self.work_space + "eval.onnx", providers=[self.backend])
            if model.is_large_model
            else ort.InferenceSession(model.model.SerializeToString(), providers=[self.backend])
        )
        inputs_names = [i.name for i in session.get_inputs()]
        len_inputs = len(session.get_inputs())
        for idx, (inputs, _) in enumerate(dataloader):
            if idx + 1 > iterations:
                break

            if len_inputs == 1:
                if isinstance(inputs, dict):
                    for name, input in inputs.items():
                        ort_inputs.update({name: to_numpy(input)})
                else:
                    ort_inputs.update({inputs_names[0]: to_numpy(inputs)})
            else:
                assert len_inputs == len(inputs), "number of input tensors must align with graph inputs"

                if isinstance(inputs, dict):
                    for name, input in inputs.items():
                        ort_inputs.update({name: to_numpy(input)})
                else:
                    ort_inputs = dict(zip(inputs_names, [to_numpy(i) for i in inputs]))

            predictions.extend(session.run(None, ort_inputs))
        return predictions


@adaptor_registry
class ONNXRT_WeightOnlyAdaptor(ONNXRUNTIMEAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """The function is used to do calibration and quanitization in post-training
           quantization.

        Args:
            tune_cfg (dict):     quantization config.
            model (object):      model need to do quantization.
            data_loader (object): calibration dataset.
            q_func (optional):   training function for quantization aware training mode,
                                 unimplement yet for onnx.

        Returns:
            (dict): quantized model
        """
        if self.performance_only:
            tmp_model = model
        else:
            try:
                tmp_model = copy.deepcopy(model)
            except Exception as e:  # pragma: no cover
                logger.warning("Fail to deep copy the model due to {}, inplace is used now.".format(repr(e)))
                tmp_model = model

        assert q_func is None, "quantization aware training has not been supported on ONNXRUNTIME"
        for precision in self.query_handler.get_precisions():
            if precision == "weight_only_integer":
                self.quantizable_op_types += self.query_handler.get_op_types_by_precision(precision=precision)
        self.quantizable_ops = self._query_quantizable_ops(tmp_model.model)

        self._update_tune_cfg(tune_cfg, tmp_model.model)
        quant_config = self._cfg_to_quantize_config(tune_cfg)
        algos = set([item["algorithm"] for key, item in quant_config.items() if isinstance(item, dict)])
        if "GPTQ" in algos:
            from neural_compressor.adaptor.ox_utils.weight_only import gptq_quantize

            assert data_loader is not None, "GPTQ WOQ algorithm needs to pass 'calib_dataloader' to quantization.fit()"
            percdamp = self.recipes.get("gptq_args", {}).get("percdamp", 0.01)
            blocksize = self.recipes.get("gptq_args", {}).get("blocksize", 128)
            actorder = self.recipes.get("gptq_args", {}).get("actorder", False)
            mse = self.recipes.get("gptq_args", {}).get("mse", False)
            perchannel = self.recipes.get("gptq_args", {}).get("perchannel", True)
            accuracy_level = self.recipes.get("gptq_args", {}).get("accuracy_level", 0)
            calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)
            tmp_model = gptq_quantize(
                tmp_model,
                data_loader,
                quant_config,
                n_samples=calib_sampling_size,
                percdamp=percdamp,
                blocksize=blocksize,
                actorder=actorder,
                mse=mse,
                perchannel=perchannel,
                accuracy_level=accuracy_level,
                providers=[self.backend],
            )
        if "AWQ" in algos:
            from neural_compressor.adaptor.ox_utils.weight_only import awq_quantize

            assert data_loader is not None, "AWQ WOQ algorithm needs to pass 'calib_dataloader' to quantization.fit()"
            enable_auto_scale = self.recipes.get("awq_args", {}).get("enable_auto_scale", True)
            enable_mse_search = self.recipes.get("awq_args", {}).get("enable_mse_search", True)
            accuracy_level = self.recipes.get("awq_args", {}).get("accuracy_level", 0)
            calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)
            tmp_model = awq_quantize(
                tmp_model,
                data_loader,
                quant_config,
                n_samples=calib_sampling_size,
                enable_auto_scale=enable_auto_scale,
                enable_mse_search=enable_mse_search,
                accuracy_level=accuracy_level,
                providers=[self.backend],
            )
        elif "RTN" in algos:
            from neural_compressor.adaptor.ox_utils.weight_only import rtn_quantize

            accuracy_level = self.recipes.get("rtn_args", {}).get("accuracy_level", 0)
            tmp_model = rtn_quantize(
                tmp_model,
                quant_config,
                accuracy_level=accuracy_level,
                providers=[self.backend],
            )
        tmp_model.q_config = copy.deepcopy(quant_config)
        self._dump_model_op_stats(tmp_model, tune_cfg)
        tmp_model.topological_sort()

        # if the model is large and acc tuning is required, save it to workspace
        if not self.performance_only and tmp_model.is_large_model:
            from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model

            model_name = os.path.split(tmp_model.model_path)[-1]
            model_path = os.path.join(self.work_space, model_name)
            data_name = model_name + "_quantized_data"
            data_path = os.path.join(self.work_space, data_name)

            load_external_data_for_model(tmp_model.model, os.path.dirname(tmp_model.model_path))

            if os.path.isfile(model_path):
                os.remove(model_path)
            if os.path.isfile(data_path):
                os.remove(data_path)

            # if the model is Tranformer-based, save hf config to workspace
            if tmp_model.hf_config is not None:
                model_type = (
                    "" if not hasattr(tmp_model.hf_config, "model_type") else getattr(tmp_model.hf_config, "model_type")
                )
                setattr(tmp_model.hf_config.__class__, "model_type", model_type)
                output_config_file = Path(self.work_space).joinpath("config.json").as_posix()
                if not os.path.exists(output_config_file):  # pragma: no cover
                    tmp_model.hf_config.to_json_file(output_config_file, use_diff=False)

            # save model and external data
            convert_model_to_external_data(
                tmp_model.model,
                all_tensors_to_one_file=True,
                location=data_name,
                size_threshold=1024,
                convert_attribute=False,
            )
            onnx.save_model(tmp_model.model, model_path)

        return tmp_model

    def _dump_model_op_stats(self, model, tune_cfg):
        import re

        fp32_op_list = self.query_handler.get_op_types_by_precision(precision="weight_only_integer")

        res = {}
        for optype in fp32_op_list:
            res[optype] = {}

        dtype_set = set()
        for node in model.nodes():
            if node.op_type in ["MatMulFpQ4", "MatMulNBits"]:
                optype = "MatMul"
            else:
                optype = node.op_type

            if optype not in res:
                continue
            if re.fullmatch("^.*_Q\d*G\d*", node.input[1]):
                search_out = re.search("_Q\d*", node.input[1])
                dtype = "A32W{}G{}".format(
                    node.input[1][search_out.start() + 2 : search_out.end()], node.input[1][search_out.end() + 1 :]
                )
            else:
                dtype = "FP32"
            dtype_set.add(dtype)

            if dtype in res[optype]:
                res[optype][dtype] += 1
            else:
                res[optype][dtype] = 1

        dtype_list = list(dtype_set)
        for dtype in dtype_list:
            for optype in res.keys():
                if dtype not in res[optype]:
                    res[optype][dtype] = 0

        # update stats format for dump.
        field_names = ["Op Type", "Total"]
        field_names.extend(dtype_list)
        output_data = []
        for op_type in res.keys():
            field_results = [op_type, sum(res[op_type].values())]
            field_results.extend([res[op_type][dtype] for dtype in dtype_list])
            output_data.append(field_results)

        Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()
        self.optype_statistics = field_names, output_data

    def _cfg_to_quantize_config(self, tune_cfg):
        quantize_config = {}
        quantize_config["calib_iteration"] = tune_cfg["calib_iteration"]

        for _, op in enumerate(self.quantizable_ops):
            if (op.name, op.op_type) not in tune_cfg["op"]:
                continue
            if tune_cfg["op"][(op.name, op.op_type)]["weight"]["dtype"] in self.query_handler.get_fallback_list():
                quantize_config[op.name] = tune_cfg["op"][(op.name, op.op_type)]["weight"]["dtype"]
            else:
                quantize_config[op.name] = copy.deepcopy(tune_cfg["op"][(op.name, op.op_type)]["weight"])

        return quantize_config

    def _update_tune_cfg(self, tune_cfg, model):
        """Update tune cfg according to woq_tuning_cfg."""
        if tune_cfg.get("woq_tuning_cfg") is None:
            return tune_cfg

        from neural_compressor.strategy.utils.constant import WOQ_TUNING_ALGOS

        woq_tuning_cfg = tune_cfg.get("woq_tuning_cfg")
        new_woq_cfg = WOQ_TUNING_ALGOS.get(woq_tuning_cfg)

        for node_cfg in tune_cfg["op"].values():
            node_cfg["weight"].update(
                {cfg_name: cfg_value for cfg_name, cfg_value in new_woq_cfg.items() if cfg_name in node_cfg["weight"]}
            )

        # find last matmul and set to fp32
        if "DISABLE_LAST_MATMUL" in woq_tuning_cfg:
            last_matmul = None
            fp32_op_cfg = {"weight": {"dtype": "fp32"}, "activation": {"dtype": "fp32", "quant_mode": "fp32"}}
            for node in model.graph.node:
                if node.op_type in ["MatMul"]:
                    last_matmul = (node.name, node.op_type)
            if last_matmul in tune_cfg["op"]:
                tune_cfg["op"][last_matmul].update(fp32_op_cfg)

    def query_fw_capability(self, model):
        """The function is used to query framework capability.
        TODO: will be replaced by framework query API

        Args:
            model: onnx model

        Returns:
            (dict): quantization capability
        """
        # optype_wise and op_wise capability
        self._pre_optimize(model)

        quantizable_optype = set([i.op_type for i in self.pre_optimized_model.nodes()])
        optype_wise = OrderedDict()
        op_wise = OrderedDict()
        for query in [self.query_handler, self.query_handler_ext]:
            if query is None:
                continue
            precisions = query.get_precisions()

            for precision in precisions:
                if precision not in ["weight_only_integer", "fp32"]:
                    continue
                # get supported optype for target precision
                optypes = (
                    query.get_op_types_by_precision(precision)
                    if query.get_op_types_by_precision(precision) != ["*"]
                    else optype_wise.keys()
                )

                configs = (
                    query.get_quantization_capability()[precision]
                    if precision in query.get_quantization_capability()
                    else {"default": {"weight": {"dtype": precision}, "activation": {"dtype": precision}}}
                )

                for op in optypes:
                    if op not in quantizable_optype:
                        continue
                    if op not in configs:
                        if "default" in configs:
                            op_capability = copy.deepcopy(configs["default"])
                        else:
                            continue
                    else:
                        op_capability = copy.deepcopy(configs[op])
                        op_capability["activation"]["quant_mode"] = "weight_only"
                    if op not in optype_wise.keys():
                        optype_wise[op] = [op_capability]
                    elif op_capability not in optype_wise[op]:
                        optype_wise[op].append(op_capability)

        for node in self.pre_optimized_model.nodes():
            if node.op_type in ["MatMul", "Attention"] and model.get_initializer(node.input[1]) is None:
                op_wise.update(
                    {(node.name, node.op_type): [{"weight": {"dtype": "fp32"}, "activation": {"dtype": "fp32"}}]}
                )
                continue
            if node.op_type in optype_wise:
                op_wise.update({(node.name, node.op_type): copy.deepcopy(optype_wise[node.op_type])})

        return {"optypewise": optype_wise, "opwise": op_wise, "recipes_ops": {}, "block_wise": []}


@adaptor_registry
class ONNXRT_QLinearOpsAdaptor(ONNXRUNTIMEAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)


@adaptor_registry
class ONNXRT_IntegerOpsAdaptor(ONNXRUNTIMEAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)


@adaptor_registry
class ONNXRT_QDQAdaptor(ONNXRUNTIMEAdaptor):
    """The ONNXRT adaptor layer, do onnx-rt quantization, calibration, inspect layer tensors.

    Args:
        framework_specific_info (dict): framework specific configuration for quantization.
    """

    def __init__(self, framework_specific_info):
        super().__init__(framework_specific_info)


class ONNXRTQuery(QueryBackendCapability):
    def __init__(self, dynamic=False, static=False, format=None, local_config_file=None):
        super().__init__()
        self.version = ort.__version__
        self.config_version = "1.6.0"
        self.dynamic = dynamic
        self.static = static
        self.format = format
        self.cfg = local_config_file
        self.cur_config = None
        self._one_shot_query()

    def _one_shot_query(self):
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
            except Exception as e:  # pragma: no cover
                logger.info("Fail to parse {} due to {}.".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError(
                    "Please check if the format of {} follows Neural Compressor yaml schema.".format(self.cfg)
                )

    def _get_specified_version_cfg(self, data):  # pragma: no cover
        """Get the configuration for the current runtime.
        If there's no matched configuration in the input yaml, we'll
        use the `default` field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        from functools import cmp_to_key

        version_config = None

        def _compare(version1, version2):
            if Version(version1[0]) == Version(version2[0]):
                return 0
            elif Version(version1[0]) < Version(version2[0]):
                return -1
            else:
                return 1

        extended_cfgs = []
        for sub_data in data:
            if "default" in sub_data["version"]["name"]:
                assert version_config is None, "Only one default config " "is allowed in framework yaml file."
                version_config = sub_data
            versions = (
                sub_data["version"]["name"]
                if isinstance(sub_data["version"]["name"], list)
                else [sub_data["version"]["name"]]
            )
            for version in versions:
                if version != "default":
                    extended_cfgs.append((version, sub_data))

        extended_cfgs = sorted(extended_cfgs, key=cmp_to_key(_compare), reverse=True)
        for k, v in extended_cfgs:
            if Version(self.version) >= Version(k):
                version_config = v
                self.config_version = k
                break

        # generate specified version config according to quantization approach and format
        config = {}
        config["capabilities"] = {}
        for k, v in version_config.items():
            if k == "version":
                config["version"] = v
            elif k == "recipes":
                config["graph_optimization"] = v["graph_optimization"]
            else:
                if self.static and "static" in v:
                    config["capabilities"].update(
                        {
                            k: {
                                node_op: node_config
                                for node_op, node_config in v["static"].items()
                                if "mode" in node_config
                                and self.format.split("ops")[0].lower()
                                in [mode.lower() for mode in node_config["mode"]]
                            }
                        }
                    )
                elif self.dynamic and "dynamic" in v:
                    config["capabilities"].update({k: v["dynamic"]})
                elif k == "weight_only_integer":
                    config["capabilities"].update({k: v})

        # generate other config content including precisions and ops
        precisions = list(version_config.keys() - {"version", "recipes"})
        if "fp32" not in precisions:
            precisions.append("fp32")
        config["precisions"] = {"names": ",".join(precisions)}

        op_types = {}
        for precision in precisions:
            if precision in config["capabilities"]:
                op_types[precision] = [op_type for op_type in config["capabilities"][precision].keys()]
            elif precision in version_config:
                op_types[precision] = version_config[precision]
        for precision, precision_config in config["capabilities"].items():
            op_types[precision] = [op_type for op_type in precision_config.keys()]
        if "fp32" not in op_types:
            op_types["fp32"] = ["*"]
        config["ops"] = op_types

        return config

    def get_version(self):  # pragma: no cover
        """Get the current backend version information.

        Returns:
            [string]: version string.
        """
        return self.cur_config["version"]["name"]

    def get_precisions(self):  # pragma: no cover
        """Get supported precisions for current backend.

        Returns:
            [string list]: the precisions' name.
        """
        return [i.strip() for i in self.cur_config["precisions"]["names"].split(",")]

    def get_op_types(self):  # pragma: no cover
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
        # assert precision in list(self.cur_config['ops'].keys())
        if precision in list(self.cur_config["ops"].keys()):
            return self.cur_config["ops"][precision]
        else:
            return []

    def get_graph_optimization(self):
        """Get onnxruntime graph optimization level."""
        level = self.cur_config["graph_optimization"]["level"]
        return level

    def get_fallback_list(self):
        """Get fallback list."""
        return list(self.cur_config["ops"].keys() - self.cur_config["capabilities"].keys())

    def get_specific_cfg_version(self):
        """Get version of the specific config."""
        return self.config_version
