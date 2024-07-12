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
"""Tensorflow Adaptor Classes."""

import copy
import math
import os
from collections import OrderedDict, UserDict

import numpy as np
import yaml

from ..data.dataloaders.base_dataloader import BaseDataLoader
from ..utils import logger
from ..utils.utility import (
    GLOBAL_STATE,
    MODE,
    CpuInfo,
    Dequantize,
    LazyImport,
    Statistics,
    deep_get,
    dump_elapsed_time,
    singleton,
    version1_eq_version2,
    version1_gte_version2,
    version1_lt_version2,
)
from .adaptor import Adaptor, adaptor_registry
from .query import QueryBackendCapability

tensorflow = LazyImport("tensorflow")
spr_base_verions = (
    "2.11.0202242",
    "2.11.0202250",
    "2.11.0202317",
    "2.11.0202323",
    "2.14.0202335",
    "2.14.dev202335",
    "2.15.0202341",
)


@adaptor_registry
class TensorFlowAdaptor(Adaptor):
    """Adaptor Layer for stock tensorflow and spr-base."""

    unify_op_type_mapping = {
        "Conv2D": "conv2d",
        "Conv3D": "conv3d",
        "DepthwiseConv2dNative": "conv2d",
        "FusedBatchNormV3": "batchnorm",
        "_MklFusedInstanceNorm": "instancenorm",
        "MaxPool": "pooling",
        "MaxPool3D": "pooling",
        "AvgPool": "pooling",
        "ConcatV2": "concat",
        "MatMul": "matmul",
        "BatchMatMul": "matmul",
        "BatchMatMulV2": "matmul",
        "Pad": "pad",
        "Conv2DBackpropInput": "deconv2d",
        "Conv3DBackpropInputV2": "deconv3d",
    }

    def __init__(self, framework_specific_info):
        """Initialization.

        Args:
            framework_specific_info: framework specific info passed from strategy.
        """
        super().__init__(framework_specific_info)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.quantize_config = {"op_wise_config": {}}
        self.framework_specific_info = framework_specific_info
        self.approach = deep_get(self.framework_specific_info, "approach", False)
        self.device = self.framework_specific_info["device"]
        self.work_dir = os.path.abspath(self.framework_specific_info["workspace_path"])
        self.recipes = deep_get(self.framework_specific_info, "recipes", {})
        self.performance_only = deep_get(self.framework_specific_info, "performance_only", False)
        self.use_bf16 = deep_get(self.framework_specific_info, "use_bf16", False)
        self.backend = self.framework_specific_info["backend"]
        self.format = self.framework_specific_info["format"]
        os.makedirs(self.work_dir, exist_ok=True)

        self.model = None
        self.pre_optimized_model = None
        self.pre_optimizer_handle = None

        self.bf16_ops = []
        self.fp32_ops = []
        self.smooth_quant_mul_ops = []
        self.dump_times = 0  # for tensorboard

        cfg_yaml_name = "{}.yaml".format(self.__class__.__name__[: -len("Adaptor")].lower())
        self.itex_mode = self.backend == "itex" or cfg_yaml_name == "tensorflow_itex.yaml"

        if self.itex_mode:
            self._check_itex()

        self.query_handler = TensorflowQuery(
            local_config_file=os.path.join(os.path.dirname(__file__), cfg_yaml_name),
            performance_only=self.performance_only,
            itex_mode=self.itex_mode,
        )

        import tensorflow as tf
        from pkg_resources import parse_version

        self.new_api = tf.version.VERSION in spr_base_verions
        self.qdq_enabled = self.itex_mode or self.format == "QDQ" or self.new_api
        self.op_wise_sequences = self.query_handler.get_eightbit_patterns(self.qdq_enabled)

        self.fp32_results = []
        self.fp32_preds_as_label = False
        self.benchmark = GLOBAL_STATE.STATE == MODE.BENCHMARK
        self.callbacks = []

        self.optype_statistics = None

        self._last_dequantize_ops = None
        self.smooth_quant_model = None

    def _check_itex(self):
        try:
            import intel_extension_for_tensorflow
        except:
            raise ImportError(
                "The Intel® Extension for TensorFlow is not installed. "
                "Please install it to run models on ITEX backend"
            )

    def _log_histogram(self, writer, tag, values, step=0, bins=1000):
        """Writes a histogram for later analysis."""
        import tensorflow as tf

        # Convert to a numpy array
        values = np.array(values)

        # Create and write Summary
        # update using TF2.X API
        with writer.as_default():
            tf.summary.histogram(tag, values, step)
            writer.flush()

    def _pre_hook_for_hvd(self, dataloader=None):
        """Pre hook for Horovod."""
        import horovod.tensorflow as hvd

        self.hvd = hvd
        self.hvd.init()

    @dump_elapsed_time(customized_msg="Model training")
    def train(self, model, dataloader, optimizer_tuple, criterion_tuple, hooks, postprocess, **kwargs):
        """Model training API.

        Args:
            model ([Graph, GraphDef or Path String]): The model could be the graph,
                        graph_def object, the frozen pb or ckpt/savedmodel folder path.
            dataloader (generator): generate the data and labels.
            optimizer_tuple (tuple): optimizers for model training.
            criterion_tuple (tuple): criterions for model training.
            hooks (callback): on_epoch_begin hook on_epoch_end hook.
            postprocess (object): process the result from the model.

        Returns:
            None.
        """
        # check model is savedmodel or not
        import tensorflow as tf

        from neural_compressor.model.tensorflow_model import get_model_type

        tf.random.set_seed(1)
        self.model_type = get_model_type(model._model)
        optimizer = optimizer_tuple[0](**optimizer_tuple[1])
        criterion = criterion_tuple[0](**criterion_tuple[1])
        start_epochs = kwargs["kwargs"].get("start_epoch", None)
        end_epochs = kwargs["kwargs"].get("end_epoch", None)
        epochs = kwargs["kwargs"].get("epoch", None)
        iters = kwargs["kwargs"].get("iteration", None)
        callbacks = kwargs["kwargs"].get("callbacks", None)
        execution_mode = kwargs["kwargs"].get("execution_mode", None)
        distributed = getattr(dataloader, "distributed", False)
        from neural_compressor.compression.distillation.criterions import TensorflowKnowledgeDistillationLoss

        if isinstance(criterion, TensorflowKnowledgeDistillationLoss):
            input_model = model._model
        else:
            input_model = tf.keras.models.load_model(model._model)
            hooks = callbacks["tf_pruning"](model, input_model, hooks)
        hooks["on_train_begin"]()  # on_train_begin hook
        train_loss_results = []
        if distributed:
            try:
                len_dataloader = len(dataloader)
            except:
                logger.info(
                    "The length of the distributed training dataloader is unknown."
                    "When the iteration of training dataloader in each process is "
                    "inconsistent, an error may occur."
                )
            else:
                list_len_dataloader = self.hvd.allgather_object(len_dataloader)
                if self.hvd.rank() == 0:
                    for i in range(len(list_len_dataloader) - 1):
                        if list_len_dataloader[i] != list_len_dataloader[i + 1]:
                            raise AttributeError(
                                "The training dataloader's iteration is"
                                "different between processes, please reset dataloader's batch_size."
                            )

        def training_step(x, y, first_batch):
            with tf.GradientTape() as tape:
                tape.watch(input_model.trainable_variables)
                y_ = input_model(x, training=True)
                loss_value = criterion(y, y_)
                loss_value = hooks["on_after_compute_loss"](x, y_, loss_value)
            tape = self.hvd.DistributedGradientTape(tape) if distributed else tape
            # Get gradient
            grads = tape.gradient(loss_value, input_model.trainable_variables)  # pylint: disable=no-member
            # Optimize the model
            optimizer.apply_gradients(zip(grads, input_model.trainable_variables))  # pylint: disable=no-member
            if distributed and first_batch:
                self.hvd.broadcast_variables(input_model.variables, root_rank=0)
                self.hvd.broadcast_variables(optimizer.variables(), root_rank=0)
            return loss_value

        training_step = training_step if execution_mode == "eager" else tf.function(training_step)
        if start_epochs is not None and end_epochs is not None:
            epochs = end_epochs - start_epochs
        for epoch in range(epochs):
            cnt = 0
            epoch_loss_avg = tf.keras.metrics.Mean()
            hooks["on_epoch_begin"](epoch)  # on_epoch_begin hook
            # Training loop
            for iter, data in enumerate(dataloader):
                x, y = postprocess(data) if postprocess is not None else data
                hooks["on_step_begin"](iter)  # on_step_begin hook
                cnt += 1
                loss_value = training_step(x, y, iter == 0)
                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                hooks["on_step_end"]()  # on_step_end hook
                if iters is not None and cnt >= iters:
                    break
            model._sess = None
            hooks["on_epoch_end"]()  # on_epoch_end hook
            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            if distributed:
                logger.info(
                    "Epoch-{:03d} training on rank {!s} have been done.".format(
                        epoch + 1, self.hvd.allgather_object(self.hvd.rank())
                    )
                )
            logger.info("Epoch {:03d}: Loss: {:.3f}".format(epoch + 1, epoch_loss_avg.result()))

        hooks["on_train_end"]()  # on_train_end hook
        model._sess = None
        if not isinstance(criterion, TensorflowKnowledgeDistillationLoss):
            if distributed:
                if self.hvd.rank() == 0:
                    # Update the input model with pruned weights manually due to keras API limitation.
                    input_model.save(model._model)
                rank_list = self.hvd.allgather_object(self.hvd.rank())
                logger.info(f"rank 0 has saved the pruned model to '{model._model}'," f"all ranks {rank_list} ready.")
            else:
                input_model.save(model._model)
        else:
            input_model.save("distillation_model")

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
        """Evaluate the model for specified metric on validation dataset.

        Args:
            model ([Graph, GraphDef or Path String]): The model could be the graph,
                        graph_def object, the frozen pb or ckpt/savedmodel folder path.
            dataloader (generator): generate the data and labels.
            postprocess (object, optional): process the result from the model
            metrics (list, optional): Depends on model category. Defaults to None.
            measurer (object, optional): for precise benchmark measurement.
            iteration(int, optional): control steps of mini-batch
            tensorboard (boolean, optional): for tensorboard inspect tensor.
            fp32_baseline (boolean, optional): only for compare_label=False pipeline

        Returns:
            [float]: evaluation result, the larger is better.
        """
        import tensorflow as tf

        from .tf_utils.util import iterator_sess_run

        outputs = model.output_tensor_names

        if getattr(dataloader, "distributed", False):
            import horovod.tensorflow as hvd

            hvd.init()
            # If metric.hvd is not None then run distributed inference
            for metric in metrics:
                metric.hvd = hvd
            try:
                len_dataloader = len(dataloader)
            except:
                logger.info(
                    "The length of the distributed evaluation dataloader is unknown."
                    "When the iteration of evaluation dataloader in each process is "
                    "inconsistent, an error may occur."
                )
            else:
                list_len_dataloader = hvd.allgather_object(len_dataloader)
                if hvd.rank() == 0:
                    for i in range(len(list_len_dataloader) - 1):
                        if list_len_dataloader[i] != list_len_dataloader[i + 1]:
                            raise AttributeError(
                                "The evaluation dataloader's iteration is"
                                "different between processes, please reset dataloader's batch_size."
                            )
            logger.info(
                "Rank {!s} dataloaders' data distribution balance check for evaluation have been finished.".format(
                    hvd.allgather_object(hvd.rank())
                )
            )
        if tensorboard:
            from tensorflow.python.framework import tensor_util

            from .tf_utils.graph_util import GraphAnalyzer

            output_postfix = "_fp32.output"
            inspect_node_types = [
                "Conv2D",
                "DepthwiseConv2dNative",
                "MaxPool",
                "AvgPool",
                "ConcatV2",
                "MatMul",
                "FusedBatchNormV3",
                "FusedBatchNorm",
                "BiasAdd",
                "_MklFusedInstanceNorm",
                "Relu",
                "Relu6",
                "Dequantize",
            ]
            fp32_inspect_node_name = []
            int8_inspect_node_name = []
            q_node_scale = {}
            if self.dump_times == 0:
                temp_dir = "./runs/eval/baseline"
            else:
                temp_dir = "./runs/eval/tune_" + str(self.dump_times)
            if os.path.isdir(temp_dir):
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            # Create the writer using TF2.x APIs to handle eager executions
            writer = tf.summary.create_file_writer(temp_dir)  # pylint: disable=no-member
            with writer.as_default():
                tf.summary.graph(model.graph)  # pylint: disable=no-member

            cur_graph = GraphAnalyzer()
            cur_graph.graph = model.graph_def
            cur_graph.parse_graph()
            graph_info = cur_graph.node_name_details
            for node in model.graph_def.node:
                if node.op in inspect_node_types:
                    fp32_inspect_node_name.append(node.name)
                # Tensor dump supported quantized op including,
                # Requantize, QuantizedConv2DAndRequantize,
                # QuantizedConv2DAndReluAndRequantize,
                # QuantizedConv2DWithBiasAndRequantize,
                # QuantizedConv2DWithBiasAndReluAndRequantize,
                # QuantizedConv2DWithBiasSignedSumAndReluAndRequantize,
                # QuantizedConv2DWithBiasSumAndReluAndRequantize,
                # QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize,
                # QuantizedMatMulWithBiasAndReluAndRequantize,
                # QuantizedMatMulWithBiasAndRequantize
                elif node.op.find("Requantize") != -1:
                    out_min = -2
                    out_max = -1
                    if node.op.find("Sum") != -1:
                        out_min = -5
                        out_max = -4
                    q_out_min = graph_info[node.input[out_min]].node.attr["value"].tensor.float_val[0]
                    q_out_max = graph_info[node.input[out_max]].node.attr["value"].tensor.float_val[0]
                    q_node_scale[node.name] = (node.op, q_out_min, q_out_max)
                    int8_inspect_node_name.append(node.name)
                # Inspect weights, bias. Need further optimize
                if node.op == "Const" and graph_info[graph_info[node.name].outputs[0]].node.op in [
                    "Conv2D",
                    "DepthwiseConv2dNative",
                    "MatMul",
                    "FusedBatchNormV3",
                    "_MklFusedInstanceNorm",
                    "BiasAdd",
                ]:
                    const_value = tensor_util.MakeNdarray(node.attr.get("value").tensor).astype(np.float32)
                    self._log_histogram(writer, node.name, const_value)

            outputs.extend(fp32_inspect_node_name)
            if len(int8_inspect_node_name) > 0:
                output_postfix = "_int8.output"
                outputs.extend(int8_inspect_node_name)

        if metrics:
            for metric in metrics:
                metric.reset()
            self.fp32_preds_as_label = any(
                [hasattr(metric, "compare_label") and not metric.compare_label for metric in metrics]
            )

        origin_output_tensor_names = model.output_tensor_names
        model.output_tensor_names = outputs
        input_tensor = model.input_tensor
        output_tensor = model.output_tensor if len(model.output_tensor) > 1 else model.output_tensor[0]
        logger.info("Start to evaluate the TensorFlow model.")

        def eval_func(dataloader):
            results = []
            for idx, (inputs, labels) in enumerate(dataloader):
                # dataloader should keep the order and len of inputs same with input_tensor
                if len(input_tensor) == 1:
                    feed_dict = {}
                    if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
                        for name in inputs:
                            for tensor in input_tensor:
                                pos = tensor.name.rfind(":")
                                t_name = tensor.name if pos < 0 else tensor.name[:pos]
                                if name == t_name:
                                    feed_dict[tensor] = inputs[name]
                                    break
                    else:
                        feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
                else:
                    assert len(input_tensor) == len(inputs), "inputs len must equal with input_tensor"
                    feed_dict = {}
                    if isinstance(inputs, dict) or isinstance(inputs, OrderedDict) or isinstance(inputs, UserDict):
                        for name in inputs:
                            for tensor in input_tensor:
                                pos = tensor.name.rfind(":")
                                t_name = tensor.name if pos < 0 else tensor.name[:pos]
                                if name == t_name:
                                    feed_dict[tensor] = inputs[name]
                                    break
                    else:
                        feed_dict = dict(zip(input_tensor, inputs))

                if model.iter_op:
                    predictions = iterator_sess_run(
                        model.sess, model.iter_op, feed_dict, output_tensor, iteration, measurer
                    )
                elif measurer is not None:
                    measurer.start()
                    predictions = model.sess.run(output_tensor, feed_dict)
                    measurer.end()
                else:
                    predictions = model.sess.run(output_tensor, feed_dict)

                if self.fp32_preds_as_label:
                    self.fp32_results.append(predictions) if fp32_baseline else results.append(predictions)

                # Inspect node output, just get 1st iteration output tensors for now
                if idx == 0 and tensorboard:
                    for index, node_name in enumerate(outputs):
                        tensor = predictions[index]
                        if node_name in int8_inspect_node_name:
                            tensor = Dequantize(predictions[index], q_node_scale[node_name])
                        self._log_histogram(writer, node_name + output_postfix, tensor.astype(np.float32), idx)
                    writer.close()
                if isinstance(predictions, list):
                    if len(origin_output_tensor_names) == 1:
                        predictions = predictions[0]
                    elif len(origin_output_tensor_names) > 1:
                        predictions = predictions[: len(origin_output_tensor_names)]
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
            return results

        if isinstance(dataloader, BaseDataLoader) and not self.benchmark:
            try:
                results = eval_func(dataloader)
            except Exception:  # pragma: no cover
                logger.warning("Fail to forward with batch size={}, set to {} now.".format(dataloader.batch_size, 1))
                dataloader.batch(1)
                results = eval_func(dataloader)
        else:  # pragma: no cover
            results = eval_func(dataloader)

        if self.fp32_preds_as_label:
            from .tf_utils.util import collate_tf_preds

            if fp32_baseline:
                results = collate_tf_preds(self.fp32_results)
                reference = results
            else:
                reference = collate_tf_preds(self.fp32_results)
                results = collate_tf_preds(results)
            for metric in metrics:
                if hasattr(metric, "compare_label") and not metric.compare_label:
                    metric.update(results, reference)

        acc = 0 if metrics is None else [metric.result() for metric in metrics]
        if tensorboard:
            new_dir = temp_dir + "_acc_" + str(acc)
            writer.close()
            if os.path.isdir(new_dir):
                import shutil

                shutil.rmtree(new_dir, ignore_errors=True)
            os.rename(temp_dir, new_dir)
            self.dump_times += 1
        model.output_tensor_names = origin_output_tensor_names
        return acc if not isinstance(acc, list) or len(acc) > 1 else acc[0]

    def _tuning_cfg_to_fw(self, tuning_cfg):
        """Parse the neural_compressor wrapped configuration to Tensorflow.

        Args:
            tuning_cfg (dict): configuration for quantization.
        """
        self.quantize_config["calib_iteration"] = tuning_cfg["calib_iteration"]
        self.quantize_config["device"] = self.device
        self.quantize_config["advance"] = deep_get(tuning_cfg, "advance")
        fp32_ops = []
        bf16_ops = []
        dispatched_op_names = [j[0] for j in tuning_cfg["op"]]

        invalid_op_names = [i for i in self.quantize_config["op_wise_config"] if i not in dispatched_op_names]

        for op_name in invalid_op_names:
            self.quantize_config["op_wise_config"].pop(op_name)

        for each_op_info in tuning_cfg["op"]:
            op_name = each_op_info[0]

            if tuning_cfg["op"][each_op_info]["activation"]["dtype"] in ["fp32", "bf16"]:
                if op_name in self.quantize_config["op_wise_config"]:
                    self.quantize_config["op_wise_config"].pop(op_name)
                if tuning_cfg["op"][each_op_info]["activation"]["dtype"] == "fp32":
                    fp32_ops.append(op_name)
                if tuning_cfg["op"][each_op_info]["activation"]["dtype"] == "bf16":
                    bf16_ops.append(op_name)
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
        self.fp32_ops = fp32_ops
        self.bf16_ops = bf16_ops

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization configuration
            model (tf.compat.v1.GraphDef): fp32 model
            data_loader (generator): generator the data and labels
            q_func (optional): training function for quantization aware training mode,
                                which not enabled for tensorflow yet.

        Returns:
            tf.compat.v1.GraphDef: the quantized model
        """
        assert (
            self.approach != "post_training_dynamic_quant"
        ), "Dynamic quantization is not supported on TensorFlow framework now!"

        if self.approach == "quant_aware_training":  # pragma: no cover
            assert (
                q_func is not None
            ), "quantization aware training mode \
                is not configured correctly"

            from neural_compressor.model import Model

            qat_model = q_func(model)

            return self.convert(Model(qat_model), "QAT", "default")

        assert q_func is None, "post-training quantization mode is not support calibration function for Tensorflow!"
        self._tuning_cfg_to_fw(tune_cfg)
        self.bf16_ops.extend(self.smooth_quant_mul_ops)
        logger.debug("Dump quantization configurations:")
        logger.debug(self.quantize_config)
        from .tf_utils.graph_converter import GraphConverter

        calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)
        if isinstance(data_loader, BaseDataLoader):
            batch_size = data_loader.batch_size
            try:
                for i in range(batch_size):
                    if calib_sampling_size % (batch_size - i) == 0:
                        calib_batch_size = batch_size - i
                        if i != 0:  # pragma: no cover
                            logger.warning(
                                "Reset `calibration.dataloader.batch_size` field "
                                "to {}".format(calib_batch_size) + " to make sure the sampling_size is "
                                "divisible exactly by batch size"
                            )
                        break
                tmp_iterations = int(math.ceil(calib_sampling_size / calib_batch_size))
                data_loader.batch(calib_batch_size)
                self.quantize_config["calib_iteration"] = tmp_iterations
                converted_model = GraphConverter(
                    model,
                    qt_config=self.quantize_config,
                    recipes=self.recipes,
                    int8_sequences=self.op_wise_sequences,
                    fp32_ops=self.fp32_ops,
                    bf16_ops=self.bf16_ops,
                    data_loader=data_loader,
                    calib_func=q_func,
                    qdq_enabled=self.qdq_enabled,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    use_bf16=self.use_bf16,
                ).convert()
            except Exception:  # pragma: no cover
                from .tf_utils.util import get_model_input_shape

                batch_size = get_model_input_shape(model)
                logger.warning(
                    "Fail to forward with batch size={}, set to {} now.".format(data_loader.batch_size, batch_size)
                )
                data_loader.batch(batch_size)
                self.quantize_config["calib_iteration"] = calib_sampling_size
                converted_model = GraphConverter(
                    model,
                    qt_config=self.quantize_config,
                    recipes=self.recipes,
                    int8_sequences=self.op_wise_sequences,
                    fp32_ops=self.fp32_ops,
                    bf16_ops=self.bf16_ops,
                    data_loader=data_loader,
                    calib_func=q_func,
                    qdq_enabled=self.qdq_enabled,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    use_bf16=self.use_bf16,
                ).convert()
        else:  # pragma: no cover
            if hasattr(data_loader, "batch_size") and calib_sampling_size % data_loader.batch_size != 0:
                iter = self.quantize_config["calib_iteration"]
                logger.warning(
                    "Please note that calibration sampling size {} "
                    "isn't divisible exactly by batch size {}. "
                    "So the real sampling size is {}.".format(
                        calib_sampling_size, data_loader.batch_size, data_loader.batch_size * iter
                    )
                )
            converted_model = GraphConverter(
                model,
                qt_config=self.quantize_config,
                recipes=self.recipes,
                int8_sequences=self.op_wise_sequences,
                fp32_ops=self.fp32_ops,
                bf16_ops=self.bf16_ops,
                data_loader=data_loader,
                calib_func=q_func,
                qdq_enabled=self.qdq_enabled,
                new_api=self.new_api,
                performance_only=self.performance_only,
                use_bf16=self.use_bf16,
            ).convert()
        # just save framework_specific_info feature for recover
        converted_model.q_config.update({"framework_specific_info": self.framework_specific_info})

        self._dump_model_op_stats(converted_model.graph_def)

        return converted_model

    def _dump_model_op_stats(self, model_graphdef):
        """Dump the whole model's OPs statistics information for analysis."""
        fp32_op_list_uint8 = copy.deepcopy(self.query_handler.get_op_types_by_precision(precision="uint8"))
        fp32_op_list_int8 = copy.deepcopy(self.query_handler.get_op_types_by_precision(precision="int8"))
        fp32_op_list = list(set(fp32_op_list_uint8).union(set(fp32_op_list_int8)))

        int8_op_prefix_list = [
            "QuantizedConv2D",
            "_FusedQuantizedConv3D",
            "QuantizedDepthwise",
            "QuantizedMaxPool",
            "QuantizedAvgPool",
            "QuantizedConcatV2",
            "QuantizedMatMul",
            "_QuantizedFusedBatchNorm",
            "_QuantizedMatMul",
            "_QuantizedBatchMatMul",
            "_QuantizedFusedInstanceNorm",
            "_FusedQuantizedDeconv2D",
            "_FusedQuantizedDeconv3D",
        ]
        from tensorflow.python.framework import dtypes

        res = {}
        for op_type in fp32_op_list:
            res[op_type] = {"INT8": 0, "BF16": 0, "FP32": 0}
        res["QuantizeV2"] = {"INT8": 0, "BF16": 0, "FP32": 0}
        res["Dequantize"] = {"INT8": 0, "BF16": 0, "FP32": 0}
        res["Cast"] = {"INT8": 0, "BF16": 0, "FP32": 0}
        fp32_op_list.extend(["QuantizeV2", "Dequantize", "Cast"])
        for i in model_graphdef.node:
            if i.op == "Const":
                continue
            possible_int8_res = [name for name in int8_op_prefix_list if i.op.find(name) != -1]

            if any(possible_int8_res):
                origin_op_type = possible_int8_res[0].split("Quantized")[-1]
                if origin_op_type == "FusedBatchNorm":
                    origin_op_type = "FusedBatchNormV3"
                if origin_op_type == "FusedInstanceNorm":
                    origin_op_type = "_MklFusedInstanceNorm"
                if origin_op_type == "Depthwise":
                    origin_op_type = "DepthwiseConv2dNative"
                if origin_op_type == "BatchMatMul":
                    origin_op_type = "BatchMatMulV2"
                if origin_op_type == "FusedBatchMatMulV2":
                    origin_op_type = "_MklFusedBatchMatMulV2"
                if origin_op_type == "Deconv2D":
                    origin_op_type = "Conv2DBackpropInput"
                if origin_op_type == "Deconv3D":
                    origin_op_type = "Conv3DBackpropInputV2"
                res[origin_op_type]["INT8"] += 1

            if i.op in fp32_op_list:
                if "T" not in i.attr and i.op != "Cast":
                    continue
                if i.op == "Cast":
                    if i.attr["DstT"].type == dtypes.bfloat16:
                        res[i.op]["BF16"] += 1
                    elif i.attr["DstT"].type == dtypes.float32:
                        res[i.op]["FP32"] += 1
                elif i.attr["T"].type == dtypes.bfloat16:
                    res[i.op]["BF16"] += 1
                elif i.attr["T"].type in (dtypes.quint8, dtypes.qint8):
                    res[i.op]["INT8"] += 1
                else:
                    res[i.op]["FP32"] += 1

        field_names = ["Op Type", "Total", "INT8", "BF16", "FP32"]
        output_data = [
            [op_type, sum(res[op_type].values()), res[op_type]["INT8"], res[op_type]["BF16"], res[op_type]["FP32"]]
            for op_type in fp32_op_list
        ]

        Statistics(output_data, header="Mixed Precision Statistics", field_names=field_names).print_stat()
        self.optype_statistics = field_names, output_data

    def _query_bf16_ops(self, matched_nodes):
        """Collect the bf16 OPs configuration for quantization."""
        self.bf16_op_details = OrderedDict()

        valid_precision = self.query_handler.get_mixed_precision_combination()
        if ("bf16" in valid_precision and CpuInfo().bf16) or os.getenv("FORCE_BF16") == "1":
            for details in matched_nodes:
                node_op = details[-1][0]
                node_name = details[0]

                self.bf16_op_details[(node_name, node_op)] = [
                    {"weight": {"dtype": ["bf16"]}, "activation": {"dtype": ["bf16"]}},
                    {"weight": {"dtype": "fp32"}, "activation": {"dtype": "fp32"}},
                ]

    def _query_quantizable_ops(self, matched_nodes):
        """Collect the op-wise configuration for quantization.

        Returns:
            OrderDict: op-wise configuration.
        """
        bf16_common_config = {"weight": {"dtype": "bf16"}, "activation": {"dtype": "bf16"}}
        fp32_common_config = {"weight": {"dtype": "fp32"}, "activation": {"dtype": "fp32"}}
        uint8_type = self.query_handler.get_op_types_by_precision(precision="uint8")
        int8_type = self.query_handler.get_op_types_by_precision(precision="int8")
        bf16_type = self.query_handler.get_op_types_by_precision(precision="bf16")
        tf_quantizable_op_type = list(set(uint8_type).union(set(int8_type)))

        valid_precision = self.query_handler.get_mixed_precision_combination()
        op_capability = self.query_handler.get_quantization_capability()
        conv_config = copy.deepcopy(op_capability["Conv2D"])
        conv3d_config = copy.deepcopy(op_capability["Conv3D"]) if "Conv3D" in op_capability else None
        matmul_config = copy.deepcopy(op_capability["MatMul"])
        other_config = copy.deepcopy(op_capability["default"])

        self.quantizable_op_details = OrderedDict()
        self.recipes_ops = {}

        self._init_op_stat = {i: [] for i in tf_quantizable_op_type}

        exclude_first_quantizable_op = (
            True
            if "first_conv_or_matmul_quantization" in self.recipes
            and not self.recipes["first_conv_or_matmul_quantization"]
            else False
        )
        for details in matched_nodes:
            node_op = details[-1][0]
            node_name = details[0]
            patterns = details[-1]
            pat_length = len(patterns)
            pattern_info = {
                "sequence": [[",".join(patterns[: pat_length - i]) for i in range(pat_length)][0]],
                "precision": ["int8"],
            }
            first_conv_or_matmul_node = []
            if (
                node_op in tf_quantizable_op_type
                and node_name not in self.exclude_node_names
                and (node_name, self.unify_op_type_mapping[node_op]) not in self.quantizable_op_details
            ):
                if (
                    self.unify_op_type_mapping[node_op].find("conv2d") != -1
                    or self.unify_op_type_mapping[node_op].find("matmul") != -1
                ) and len(first_conv_or_matmul_node) == 0:
                    first_conv_or_matmul_node.append((node_name, self.unify_op_type_mapping[node_op]))
                    self.recipes_ops["first_conv_or_matmul_quantization"] = first_conv_or_matmul_node
                if exclude_first_quantizable_op and (
                    self.unify_op_type_mapping[node_op].find("conv2d") != -1
                    or self.unify_op_type_mapping[node_op].find("matmul") != -1
                ):
                    exclude_first_quantizable_op = False
                    self.exclude_node_names.append(node_name)
                    continue
                self._init_op_stat[node_op].append(node_name)
                if self.unify_op_type_mapping[node_op].find("conv2d") != -1:
                    conv2d_int8_config = copy.deepcopy(conv_config)
                    conv2d_int8_config["pattern"] = pattern_info
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])] = [
                        conv2d_int8_config,
                        fp32_common_config,
                    ]
                elif self.unify_op_type_mapping[node_op].find("conv3d") != -1:
                    conv3d_int8_config = copy.deepcopy(conv3d_config)
                    conv3d_int8_config["pattern"] = pattern_info
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])] = [
                        conv3d_int8_config,
                        fp32_common_config,
                    ]
                elif self.unify_op_type_mapping[node_op].find("matmul") != -1:
                    matmul_int8_config = copy.deepcopy(matmul_config)
                    matmul_int8_config["pattern"] = pattern_info
                    # TODO enable the sym mode once the tf fixed the mkldequantize_op.cc bug.
                    # is_positive_input = self.pre_optimizer_handle.has_positive_input(node_name)
                    # matmul_scheme = 'sym' if is_positive_input else 'asym'
                    matmul_scheme = ["asym"]
                    matmul_int8_config["activation"]["scheme"] = matmul_scheme
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])] = [
                        matmul_int8_config,
                        fp32_common_config,
                    ]
                else:
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])] = [
                        copy.deepcopy(other_config),
                        fp32_common_config,
                    ]
                if node_op in bf16_type and (
                    ("bf16" in valid_precision and CpuInfo().bf16) or os.getenv("FORCE_BF16") == "1"
                ):
                    self.quantizable_op_details[(node_name, self.unify_op_type_mapping[node_op])].insert(
                        1, bf16_common_config
                    )

                self.quantize_config["op_wise_config"][node_name] = (False, "minmax", False)
        return self.quantizable_op_details

    def _filter_unquantizable_concat(self, matched_nodes):
        """Filter out unquantizable ConcatV2 Ops based on the positive input rule."""
        target_concat_nodes = [i[0] for i in matched_nodes if i[-1][0] == "ConcatV2"]
        from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper
        from neural_compressor.adaptor.tf_utils.util import GraphAnalyzer

        g = GraphAnalyzer()
        g.graph = self.pre_optimized_model.graph_def
        graph_info = g.parse_graph()
        concat_nodes = g.query_fusion_pattern_nodes([["ConcatV2"]])
        for i in concat_nodes:
            concat_node_name = i[0]
            if concat_node_name not in target_concat_nodes:
                continue
            input_positive_status = []
            for index in range(graph_info[concat_node_name].node.attr["N"].i):
                each_input_name = GraphRewriterHelper.node_name_from_input(
                    graph_info[concat_node_name].node.input[index]
                )
                each_input_node = graph_info[each_input_name].node
                positive_input = False
                if each_input_node.op in ("Relu", "Relu6"):
                    positive_input = True
                else:
                    positive_input = g.has_positive_input(each_input_node.name)
                input_positive_status.append(positive_input)
            if not any(input_positive_status):
                matched_nodes.remove(i)

    def _filter_unquantizable_concat_performance_only(self, matched_nodes):
        """OOB filter out unquantizable ConcatV2 OPs by checking the control flow rule."""
        target_concat_nodes = [i[0] for i in matched_nodes if i[-1][0] == "ConcatV2"]
        from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper
        from neural_compressor.adaptor.tf_utils.util import GraphAnalyzer

        g = GraphAnalyzer()
        g.graph = self.pre_optimized_model.graph_def
        graph_info = g.parse_graph()
        concat_nodes = g.query_fusion_pattern_nodes([["ConcatV2"]])
        for i in concat_nodes:
            concat_node_name = i[0]
            if concat_node_name not in target_concat_nodes:
                continue
            input_positive_status = []
            control_flow = False
            for index in range(graph_info[concat_node_name].node.attr["N"].i):
                each_input_name = GraphRewriterHelper.node_name_from_input(
                    graph_info[concat_node_name].node.input[index]
                )
                each_input_node = graph_info[each_input_name].node
                if each_input_node.op in ("Switch"):
                    control_flow = True
            if control_flow:
                matched_nodes.remove(i)

    def query_fw_capability(self, model):
        """Collect the model-wise and op-wise configuration for quantization.

        Args:
            model (tf.compat.v1.GraphDef): model definition.

        Returns:
            [dict]: model-wise & op-wise configuration for quantization.
        """
        if self.pre_optimized_model is None:
            from .tf_utils.graph_rewriter.generic.pre_optimize import PreOptimization

            self.pre_optimizer_handle = PreOptimization(model, self.new_api, self.device)
            self.pre_optimized_model = self.pre_optimizer_handle.get_optimized_model(self.itex_mode)
            model.graph_def = self.pre_optimized_model.graph_def

        self.exclude_node_names = self.pre_optimizer_handle.get_excluded_node_names()
        patterns = self.query_handler.generate_internal_patterns()
        bf16_patterns = self.query_handler.get_bf16_patterns()
        matched_nodes = self.pre_optimizer_handle.get_matched_nodes(patterns)
        matched_bf16_nodes = self.pre_optimizer_handle.get_matched_nodes(bf16_patterns)
        original_graph_node_name = [i.name for i in model.graph_def.node]
        matched_nodes = sorted(
            matched_nodes, reverse=True, key=lambda i: (original_graph_node_name.index(i[0]), len(i[-1]))
        )

        def check_match(patterns, input_pattern):
            for i in patterns:
                if input_pattern == [i for i in i.replace("+", " ").strip().split(" ") if i]:
                    return True
            return False

        if (self.new_api and self.performance_only) or self.itex_mode or os.getenv("TF_FORCE_CONCAT_OPTS") == "1":
            self._filter_unquantizable_concat_performance_only(matched_nodes)
        else:
            self._filter_unquantizable_concat(matched_nodes)
        copied_matched_nodes = copy.deepcopy(matched_nodes)
        for i in copied_matched_nodes:
            if i[-1][0] in self.query_handler.get_op_types()["int8"]:
                continue

            if not self.pre_optimizer_handle.has_positive_input(i[0]) and not check_match(
                self.query_handler.get_fuse_patterns()["int8"], i[-1]
            ):
                matched_nodes.remove(i)

        del copied_matched_nodes

        copied_matched_nodes = copy.deepcopy(matched_bf16_nodes)
        for i in copied_matched_nodes:
            for j in matched_nodes:
                if i[0] == j[0] and i in matched_bf16_nodes:
                    matched_bf16_nodes.remove(i)

        del copied_matched_nodes

        self._query_quantizable_ops(matched_nodes)
        self._query_bf16_ops(matched_bf16_nodes)
        capability = {"optypewise": self.get_optype_wise_ability(), "recipes_ops": self.recipes_ops}
        capability["opwise"] = copy.deepcopy(self.quantizable_op_details)
        capability["opwise"].update(self.bf16_op_details)
        logger.debug("Dump framework quantization capability:")
        logger.debug(capability)

        return capability

    def set_tensor(self, model, tensor_dict):
        """Quantize the bias and weight tensors in tensor_dict."""
        from .tf_utils.graph_util import GraphAnalyzer

        g = GraphAnalyzer()
        g.graph = model.graph_def
        graph_info = g.parse_graph()

        def _get_fp32_op_name(model, tensor_name):
            is_weight = False
            is_biasadd = False
            last_node_name = None
            current_node_name = None
            for each_node in model.graph_def.node:
                if tensor_name in each_node.input:
                    tensor_index = list(each_node.input).index(tensor_name)
                    if each_node.op.find("Quantized") != -1 and tensor_index == 2:
                        is_biasadd = True
                        last_node_name = each_node.input[0]
                        current_node_name = each_node.name

                if tensor_name + "_qint8_const" in each_node.input:
                    pass

            return is_weight, is_biasadd, current_node_name, last_node_name

        from tensorflow.core.framework import attr_value_pb2
        from tensorflow.python.framework import dtypes, tensor_util

        from neural_compressor.adaptor.tf_utils.graph_util import GraphRewriterHelper as Helper

        qint32_type = dtypes.qint32.as_datatype_enum

        for tensor_name, tensor_content in tensor_dict.items():
            is_weight, is_biasadd, current_node_name, last_node_name = _get_fp32_op_name(model, tensor_name)

            if is_biasadd:
                is_biasadd_dtype_is_fp32 = graph_info[current_node_name].node.attr["Tbias"] == attr_value_pb2.AttrValue(
                    type=dtypes.float32.as_datatype_enum
                )
                current_node = graph_info[current_node_name].node
                bias_add_node = graph_info[current_node.input[2]].node
                if is_biasadd_dtype_is_fp32:
                    bias_add_node.attr["value"].CopyFrom(
                        attr_value_pb2.AttrValue(
                            tensor=tensor_util.make_tensor_proto(tensor_content, dtypes.float32, tensor_content.shape)
                        )
                    )
                else:
                    last_node = graph_info[last_node_name].node
                    min_input = graph_info[last_node.input[-2]].node.attr["value"].tensor.float_val[0]
                    max_input = graph_info[last_node.input[-1]].node.attr["value"].tensor.float_val[0]
                    channel_size = tensor_content.shape[0]
                    max_filter_node = graph_info[current_node.input[6]].node
                    min_filter_node = graph_info[current_node.input[5]].node
                    if max_filter_node.attr["value"].tensor.float_val:
                        max_filter_tensor = []
                        min_filter_tensor = []
                        max_filter_tensor.append((max_filter_node.attr["value"].tensor.float_val)[0])
                        min_filter_tensor.append((min_filter_node.attr["value"].tensor.float_val)[0])
                    else:
                        max_filter_tensor = tensor_util.MakeNdarray(min_filter_node.attr["value"].tensor)
                        min_filter_tensor = tensor_util.MakeNdarray(min_filter_node.attr["value"].tensor)
                    activation_range = 127.0 if current_node.attr["Tinput"].type == dtypes.qint8 else 255.0
                    updated_bias = Helper.generate_int32_bias_for_conv(
                        tensor_content,
                        channel_size,
                        max_input,
                        min_input,
                        max_filter_tensor,
                        min_filter_tensor,
                        activation_range,
                    )

                    bias_add_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=qint32_type))
                    bias_add_node.attr["value"].CopyFrom(
                        attr_value_pb2.AttrValue(
                            tensor=tensor_util.make_tensor_proto(updated_bias, dtypes.int32, tensor_content.shape)
                        )
                    )
                    bias_add_node.attr["value"].tensor.dtype = qint32_type
                    current_node.attr["Tbias"].CopyFrom(attr_value_pb2.AttrValue(type=qint32_type))

            if is_weight:
                tmp_const_node = Helper.create_constant_node(
                    current_node.name + "_weights_tmp", tensor_content.transpose(2, 3, 1, 0), dtypes.float32
                )
                min_filter_node = graph_info[current_node.input[5]].node
                per_channel = True if min_filter_node.attr["value"].tensor.tensor_shape else False
                from .tf_utils.quantize_graph_common import QuantizeGraphHelper

                original_fp32_op = current_node.op.split("With")[0].split("Quantized")[-1]
                if original_fp32_op.find("Depthwise") != -1:
                    original_fp32_op = "DepthwiseConv2dNative"
                qint8_const_node, min_node, max_node = QuantizeGraphHelper.generate_quantized_weight_node(
                    original_fp32_op, tmp_const_node, per_channel
                )
                g.add_node(qint8_const_node, [], [current_node.name])
                g.add_node(min_node, [], [current_node.name])
                g.add_node(max_node, [], [current_node.name])
                g.replace_constant_graph_with_constant_node(qint8_const_node, tensor_name)
                g.replace_constant_graph_with_constant_node(min_node, current_node.input[5])
                g.replace_constant_graph_with_constant_node(max_node, current_node.input[6])

    def inspect_weight_and_bias(self, node_list, graph_def, graph_info, graph_node_name_mapping):
        """Inspect the weights and biases."""
        import tensorflow as tf

        from neural_compressor.adaptor.tf_utils.util import get_tensor_val_from_graph_node
        from neural_compressor.utils.utility import dequantize_weight

        from .tf_utils.util import int8_node_name_reverse

        weights_result = {}
        inspect_nodes = []
        node_set = set(node_list)
        for node in graph_def.node:
            node_name = node.name
            if "Quantized" in node.op:
                node_name = int8_node_name_reverse(node)
            if node_name in node_set and ("Conv" in node.op or "Mul" in node.op):
                inspect_nodes.append(node)
        logger.debug(f"Start to inspect weight and bias for: {[node.name for node in inspect_nodes]}.")
        for node in inspect_nodes:
            # inspect weights and bias
            node_name = node.name
            weight_node_name = node.input[1]
            weight_node = graph_node_name_mapping[weight_node_name]
            if weight_node.op != "Const":  # skip the matmul whose two inputs are previous output
                continue
            weight_node_val = get_tensor_val_from_graph_node(graph_node_name_mapping, weight_node_name)
            weight_node_val = weight_node_val.astype("float32")
            # dequantize the weight for quantized model
            if "Quantized" in node.op:
                node_name = int8_node_name_reverse(node)
                weight_node_name_pre = weight_node_name.split("_qint8_const")[0]
                min_filter_node = weight_node_name_pre + "_min"
                max_filter_node = weight_node_name_pre + "_max"
                if graph_info[min_filter_node].node.attr["value"].tensor.float_val:
                    min_filter_val = graph_info[min_filter_node].node.attr["value"].tensor.float_val
                    max_filter_val = graph_info[max_filter_node].node.attr["value"].tensor.float_val
                else:
                    min_filter_val = get_tensor_val_from_graph_node(graph_node_name_mapping, min_filter_node)
                    max_filter_val = get_tensor_val_from_graph_node(graph_node_name_mapping, max_filter_node)
                weight_node_val = dequantize_weight(weight_node_val, min_filter_val, max_filter_val)
            weights_result[node_name] = {weight_node_name: weight_node_val}
        return weights_result

    def fused_node_mapping(self, node_list, pattern_mapping, graph_info, graph_node_name_mapping):
        """Create the mapping between first node and last node in fused sequence.

        Args:
            node_list: node name list
            pattern_mapping:  key: node name, val: node pattern mapping
            graph_info: key: node name, val: node details
            graph_node_name_mapping: key: node name, val: node
        Returns:
            fused_mapping: key: first node name in fused seq, val: last node in fused seq
            fused_mapping_reverse: key: last node in fused seq, val: first node name in fused seq
        """
        fused_mapping = {}
        fused_mapping_reverse = {}
        for node_name in node_list:
            fused_seq = pattern_mapping[node_name]["sequence"].split(",")
            # for the node not fused with others
            if len(fused_seq) == 1:
                fused_mapping[node_name] = node_name
                fused_mapping_reverse[node_name] = node_name
                continue
            _next_node_name = node_name
            for _next_node_op_type in fused_seq[1:]:
                node_details = graph_info[_next_node_name]
                for node_output_name in node_details.outputs:
                    if graph_node_name_mapping[node_output_name].op == "Cast":
                        cast_node = graph_node_name_mapping[node_output_name]
                        node_output_name = graph_info[cast_node.name].outputs[0]
                    if graph_node_name_mapping[node_output_name].op in [_next_node_op_type, "Cast"]:
                        _next_node_name = node_output_name
            fused_mapping[node_name] = _next_node_name
            fused_mapping_reverse[_next_node_name] = node_name
        return fused_mapping, fused_mapping_reverse

    def _inspect_tensor_inference(self, inspect_node_dict, model, dataloader, iteration_list):
        """Do inference for inspect activation."""
        out_tensor_lst = []
        out_tensor_lst += [{n: [n + ":" + str(i) for i in range(3)]} for n in inspect_node_dict["qreq_node"]]
        out_tensor_lst += [{n: n + ":0"} for n in inspect_node_dict["qdq_node"]]
        out_tensor_lst += [{n: n + ":0"} for n in inspect_node_dict["f_node"]]
        out_cnt = len(out_tensor_lst)
        iteration_list = set(iteration_list)
        input_tensor = model.input_tensor
        logger.info("Start to do inference for inspect activation.")
        activation_result = []
        for idx, (inputs, labels) in enumerate(dataloader):
            model_out = []
            if idx + 1 > max(iteration_list):
                break
            if idx + 1 not in iteration_list:
                continue
            if len(input_tensor) == 1:
                feed_dict = {input_tensor[0]: inputs}  # get raw tensor using index [0]
            else:
                assert len(input_tensor) == len(inputs), "inputs len must equal with input_tensor"
                feed_dict = dict(zip(input_tensor, inputs))
            # TODO find an optimized method to avoid multiple runs
            for i, out_t in enumerate(out_tensor_lst):
                logger.debug(f"Finished inspect {i}/{out_cnt} nodes, current inspect node {out_t.keys()}.")
                model_out.append(model.sess.run(out_t, feed_dict))
            activation_result.append(model_out)
        return activation_result

    def inspect_activation(
        self, node_list, graph_def, graph_node_name_mapping, quantization_cfg, dataloader, iteration_list, graph_info
    ):
        """Inspect the activation."""
        from neural_compressor.model import Model

        original_graph_node_mapping = {}
        for node in graph_def.node:
            original_graph_node_mapping[node.name] = node
        inspect_node_dict = {"qdq_node": [], "qreq_node": [], "f_node": []}
        for node_name in node_list:
            node = graph_node_name_mapping[node_name]
            if "Quantized" in node.op and "Dequantize" in node.op:
                inspect_node_dict["qdq_node"].append(node.name)
            elif "Quantized" in node.op or "_Quantized" in node.op or "Requantize" in node.op:
                inspect_node_dict["qreq_node"].append(node.name)
            else:
                inspect_node_dict["f_node"].append(node_name)
        pattern_mapping = {}
        node_dict = quantization_cfg["op"]
        for node_name_and_type in node_dict.keys():
            node_name, _ = node_name_and_type
            if "pattern" in node_dict[node_name_and_type]:
                pattern_mapping[node_name] = node_dict[node_name_and_type]["pattern"]
            else:
                pattern_mapping[node_name] = {"sequence": node_name}
        if inspect_node_dict["f_node"]:
            fuse_map, fuse_map_reverse = self.fused_node_mapping(
                inspect_node_dict["f_node"], pattern_mapping, graph_info, graph_node_name_mapping
            )
            inspect_node_dict["f_node"] = [fuse_map[n] for n in inspect_node_dict["f_node"]]
        # build model and do inference
        model = Model(graph_def)
        activation_result = self._inspect_tensor_inference(inspect_node_dict, model, dataloader, iteration_list)
        final_result = []
        int8_postfix = "_eightbit"
        for iter_res in activation_result:
            tmp_iter_result = {}
            for res in iter_res:
                node_name, val = list(res.keys())[0], list(res.values())[0]
                val = Dequantize(val[0], (node_name, val[1], val[2])) if len(val) == 3 else val
                val = val.astype(np.float32)
                index_postfix = node_name.find(int8_postfix)
                if index_postfix != -1:
                    node_name = node_name[:index_postfix]
                    tmp_iter_result[node_name] = {node_name: val}
                else:
                    tmp_iter_result[fuse_map_reverse[node_name]] = {fuse_map_reverse[node_name]: val}
            final_result.append(tmp_iter_result)
        return final_result

    def inspect_tensor(
        self,
        model,
        dataloader=None,
        op_list=[],
        iteration_list=[],
        inspect_type="activation",
        save_to_disk=False,
        save_path=None,
        quantization_cfg=None,
    ):
        """Dump the weight and activation(output) to local disk.

        1. create the correspondence between query node name and the actually output node name in graph_def
        2. get the weight and bias for the given node
        3. get the activation for the given node
        4. save the tensor to disk
        Args:
            model: int8/fp32 graph_def/TensorflowBaseModel
            dataloader: dataloader used during inspect activation
            op_list: op list to inspect
            iteration_list: iteration list to inspect, start from 1
            inspect_type: activation/weight/all
            save_to_disk: dump to disk or not
            save_path: the dump path for inspect tensor
            quantization_cfg: quantization configuration for fused fp32 model and quantized model
        Returns:
            Dict
               {
                 'weight': {
                   'node0_name': {'weight0_name': numpy.array, 'bias0_name': numpy.array, ...},
                   'node1_name': {'weight1_name': numpy.array, 'bias1_name': numpy.array, ...},
                   ...
                 },
                 'activation': [
                   # iter 1:
                       {
                         'node0_name': {'output0_name': numpy.array, 'output1_name': numpy.array, ...}
                         'node1_name': {'output1_name': numpy.array, 'output1_name': numpy.array, ...}
                         ...
                       },
                   # iter 2:
                        {
                       ...
                       }
                 ]
               }
        """
        import tensorflow as tf

        from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
        from neural_compressor.model.tensorflow_model import TensorflowBaseModel
        from neural_compressor.utils.utility import dump_data_to_local, load_data_from_pkl

        from .tf_utils.util import int8_node_name_reverse

        if isinstance(model, TensorflowBaseModel):
            model = model.graph_def
        if not quantization_cfg:
            # TODO get config from graph if config is None
            quantization_cfg = load_data_from_pkl("./nc_workspace/", "cfg.pkl")
        node_list = op_list
        # create the mapping between node name and node, key: node_name, val: node
        graph_node_name_mapping = {}
        quan_model_flag = False
        for node in model.node:
            node_name = int8_node_name_reverse(node)
            if "Quantized" in node.op:
                quan_model_flag = True
                node_name = int8_node_name_reverse(node)
            if node.attr["value"].tensor.dtype == tf.dtypes.bfloat16.as_datatype_enum:
                quan_model_flag = True
            graph_node_name_mapping[node_name] = node
        if quan_model_flag:
            logger.info("Dump the tensor for quantized model.")

        # create the mapping between node name and node detail
        g = GraphAnalyzer()
        g.graph = model
        graph_info = g.parse_graph()
        inspect_result = {}

        # inspect weight
        if inspect_type == "weight" or inspect_type == "all":
            logger.info("Start to inspect weight and bias.")
            weights_result = self.inspect_weight_and_bias(node_list, model, graph_info, graph_node_name_mapping)
            inspect_result["weight"] = weights_result

        # inspect activation
        if inspect_type == "activation" or inspect_type == "all":
            logger.info("Start to inspect activation.")
            activation_result = self.inspect_activation(
                node_list, model, graph_node_name_mapping, quantization_cfg, dataloader, iteration_list, graph_info
            )
            inspect_result["activation"] = activation_result

        # save to disk
        if save_to_disk:
            if not save_path:
                save_path = "./nc_workspace/tmp/"
            dump_data_to_local(inspect_result, save_path, "inspect_result.pkl")
            logger.info(f"Dumped the inspect tensor to {save_path}")
        return inspect_result

    def quantize_input(self, model):
        """Quantize the model to be able to take quantized input.

        Remove graph QuantizedV2 op and move its input tensor to QuantizedConv2d
        and calculate the min-max scale.

        Args:
            model (tf.compat.v1.GraphDef): The model to quantize input

        Return:
            model (tf.compat.v1.GraphDef): The quantized input model
            scale (float): The scale for dataloader to generate quantized input
        """
        scale = None
        # quantize input only support tensorflow version > 2.1.0
        import tensorflow as tf

        if version1_lt_version2(tf.version.VERSION, "2.1.0"):
            logger.warning("Quantize input needs tensorflow 2.1.0 and newer.")
            return model, scale

        graph_def = model.as_graph_def()
        node_name_mapping = {}
        quantize_nodes = []
        for node in graph_def.node:
            node_name_mapping[node.name] = node
            if node.op == "QuantizeV2":
                quantize_nodes.append(node)

        target_quantize_nodes = []
        for node in quantize_nodes:
            # only support Quantizev2 input op Pad and Placeholder
            if (
                node_name_mapping[node.input[0]].op == "Pad"
                and node_name_mapping[node_name_mapping[node.input[0]].input[0]].op == "Placeholder"
            ) or node_name_mapping[node.input[0]].op == "Placeholder":
                target_quantize_nodes.append(node)
        assert len(target_quantize_nodes) == 1, "only support 1 QuantizeV2 from Placeholder"
        quantize_node = target_quantize_nodes[0]

        quantize_node_input = node_name_mapping[quantize_node.input[0]]
        quantize_node_outputs = [node for node in graph_def.node if quantize_node.name in node.input]

        from .tf_utils.graph_util import GraphRewriterHelper

        if quantize_node_input.op == "Pad":
            pad_node_input = node_name_mapping[quantize_node_input.input[0]]
            assert pad_node_input.op == "Placeholder", "only support Pad between QuantizeV2 and Placeholder"
            from tensorflow.python.framework import tensor_util

            paddings_tensor = tensor_util.MakeNdarray(
                node_name_mapping[quantize_node_input.input[1]].attr["value"].tensor
            ).flatten()

            quantize_node.input[0] = quantize_node_input.input[0]
            for conv_node in quantize_node_outputs:
                assert "Conv2D" in conv_node.op, "only support QuantizeV2 to Conv2D"

                GraphRewriterHelper.set_attr_int_list(conv_node, "padding_list", paddings_tensor)
            graph_def.node.remove(quantize_node_input)

        from tensorflow.python.framework import dtypes

        GraphRewriterHelper.set_attr_dtype(node_name_mapping[quantize_node.input[0]], "dtype", dtypes.qint8)

        for conv_node in quantize_node_outputs:
            for index, conv_input in enumerate(conv_node.input):
                if conv_input == quantize_node.name:
                    conv_node.input[index] = quantize_node.input[0]
                elif conv_input == quantize_node.name + ":1":
                    conv_node.input[index] = quantize_node.input[1]
                elif conv_input == quantize_node.name + ":2":
                    conv_node.input[index] = quantize_node.input[2]

        # get the input's min-max value and calculate scale
        max_node = node_name_mapping[quantize_node.input[2]]
        min_node = node_name_mapping[quantize_node.input[1]]
        max_value = max_node.attr["value"].tensor.float_val[0]
        min_value = min_node.attr["value"].tensor.float_val[0]
        scale = 127.0 / max(abs(max_value), abs(min_value))
        # remove QuantizeV2 node
        graph_def.node.remove(quantize_node)

        graph = tensorflow.Graph()
        with graph.as_default():
            # use name='' to avoid 'import/' to name scope
            tensorflow.import_graph_def(graph_def, name="")
        return graph, scale

    def get_optype_wise_ability(self):
        """Get the op type wise capability by generating the union value of each op type.

        Returns:
            [string dict]: the key is op type while the value is the
                           detail configurations of activation and weight for this op type.
        """
        res = OrderedDict()
        for op in self.quantizable_op_details:
            if op[1] not in res:
                res[op[1]] = {"activation": self.quantizable_op_details[op][0]["activation"]}
                if "weight" in self.quantizable_op_details[op][0]:
                    res[op[1]]["weight"] = self.quantizable_op_details[op][0]["weight"]
        for op in self.bf16_op_details:
            if op[1] not in res:
                res[op[1]] = {"activation": {"dtype": ["bf16"]}, "weight": {"dtype": ["bf16"]}}
        return res

    def _pre_hook_for_qat(self, dataloader=None):
        """Pre hook for QAT."""
        self.model.model = self.qat_convert(self.model.model)

    def _post_hook_for_qat(self):
        """Post hook for QAT."""
        pass

    def _pre_eval_hook(self, model):
        """Pre evaluation hook."""
        return model

    # Add keyword arguments unpacking
    def _post_eval_hook(self, model, **kwargs):
        """Post evaluation hook."""
        pass

    def save(self, model, path):
        """Save model to the path."""
        pass

    # this function is used to convert keras QAT model to pb in old QAT implementation,
    # and it's not used in refactored QAT
    def convert(self, model, source, destination):  # pragma: no cover
        """The function is used to convert a source model format to another.

        Args:
            model (neural_compressor.model): base model to be converted.
            source (string): The source model format.
            destination (string): The destination model format.
        """
        assert source.lower() == "qat" and destination.lower() == "default"
        capability = self.query_fw_capability(model)

        quantize_config = {"op_wise_config": {}}
        for each_op_info in capability["opwise"]:
            is_perchannel = False
            weight_bit = 7.0
            for op_cap in capability["opwise"][each_op_info]:
                if "activation" in op_cap and "quant_mode" in op_cap["activation"]:
                    activation = op_cap["activation"]
                    if "weight" in op_cap:
                        weight = op_cap["weight"]
                        is_perchannel = True if weight["granularity"][0] == "per_channel" else False
                    algorithm = activation["algorithm"][0]
                    is_asymmetric = False
                    if "activation" in op_cap:
                        is_asymmetric = True if activation["scheme"][0] == "asym" else False

                    quantize_config["op_wise_config"][each_op_info[0]] = (
                        is_perchannel,
                        algorithm,
                        is_asymmetric,
                        weight_bit,
                    )
        from .tf_utils.graph_converter import GraphConverter

        tmp_graphdef = copy.deepcopy(model.graph_def)
        for i in tmp_graphdef.node:
            if i.op == "Const" and i.input:
                i.ClearField("input")
        model.graph_def = tmp_graphdef
        converter = GraphConverter(
            model,
            qt_config=quantize_config,
            int8_sequences=self.op_wise_sequences,
            fake_quant=True,
            new_api=self.new_api,
            performance_only=self.performance_only,
            use_bf16=self.use_bf16,
        )

        return converter.convert()

    def qat_convert(self, model, quantize_recipe=None):
        """Convert a fp32 'tf.keras' model to be a int8 one with quantization aware training implementation.

        Args:
            model (tf.keras.Model): The model to be quantized, expected to be a Keras Functional or Sequential model.
            quantize_recipe (dict): A dict that decide whether given layers should be quantized.

        Returns:
            converted_model (tf.keras.Model): Quantized model with fake quant nodes inserted.
        """
        import tensorflow as tf

        assert isinstance(model, tf.keras.Model), (
            "The model to be converted is expected to be "
            "a `tf.keras.Model` instance. You should not pass an instance of type: {input}.".format(
                input=model.__class__.__name__
            )
        )

        assert model.__class__.__name__ in [
            "Functional",
            "Sequential",
        ], "Only `Functional` or `Sequential` keras model is supported for QAT."

        from .tf_utils.quantize_graph.qat.quantize_config import global_config
        from .tf_utils.quantize_graph.qat.quantize_helper import init_quantize_config, qat_clone_function

        config = init_quantize_config(model, quantize_recipe)
        q_model = tf.keras.models.clone_model(model, input_tensors=None, clone_function=qat_clone_function)
        global_config.clear()

        return q_model

    @dump_elapsed_time("Pass recover model")
    def recover_tuned_model(self, model, q_config):
        """Execute the recover process on the specified model.

        Args:
            tune_cfg (dict): quantization configuration
            model (tf.compat.v1.GraphDef): fp32 model
            q_config (dict): recover configuration

        Returns:
            tf.compat.v1.GraphDef: the quantized model
        """
        from .tf_utils.graph_rewriter.generic.pre_optimize import PreOptimization

        self.pre_optimizer_handle = PreOptimization(model, self.new_api, self.device)
        self.pre_optimized_model = self.pre_optimizer_handle.get_optimized_model(self.itex_mode)
        model.graph_def = self.pre_optimized_model.graph_def

        from .tf_utils.graph_converter_without_calib import GraphConverterWithoutCalib

        converter = GraphConverterWithoutCalib(
            model,
            recover_config=q_config,
            new_api=self.new_api,
            performance_only=self.performance_only,
            use_bf16=self.use_bf16,
        )

        return converter.convert_without_calib()

    def get_output_op_names(self, qmodel):
        """Get the oupur OPs's names."""
        from .tf_utils.graph_util import GraphAnalyzer

        graph_def = GraphAnalyzer().parse_graph(qmodel.graph_def)
        output_op_names = set()

        def _add_output_op_name(opname):
            if opname.endswith("_dequantize"):
                output_op_names.add(opname[: -len("_dequantize")])  # pylint: disable=no-member
            elif opname.endswith("__dequant"):
                pass
            else:
                output_op_names.add(opname)  # pylint: disable=no-member

        for output_opname in qmodel.output_node_names:
            op_count = 0
            stack = [output_opname]
            while stack:
                opname = stack.pop()
                while True:
                    op_count += 1
                    if opname not in graph_def:
                        break
                    op = graph_def[opname]
                    if op.node.op == "Dequantize":
                        _add_output_op_name(opname)
                        break
                    next_opnames = op.node.input
                    if not next_opnames:
                        break
                    elif len(next_opnames) > 1:
                        stack += next_opnames[1:]

                    opname = next_opnames[0]

        output_op_names = list(output_op_names)
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
        partial_dataloader = self._partial_dataloader(dataloader, confidence_batches)

        fp32_output = self._inference_model_on_batches(fp32_model, tune_cfg, partial_dataloader, output_op_names)

        for op in ops_lst:
            # backup and set replace tuning config
            backup_cfg = op_cfg[op]
            op_cfg[op] = replace_cfgs[op]

            # quantize and inference the model
            q_model = self.quantize(tune_cfg, fp32_model, partial_dataloader)
            q_output = self._inference_model_on_batches(q_model, tune_cfg, partial_dataloader, output_op_names)

            mse_result[op] = self._calculate_mse(fp32_output, q_output)

            # recover tune_cfg
            op_cfg[op] = backup_cfg

        return mse_result

    def _partial_dataset_of(self, dataloader, confidence_batches):
        """Partial dataset."""
        from neural_compressor.data.datasets.dummy_dataset import DummyDataset
        from neural_compressor.data.datasets.dummy_dataset import DummyDataset as DummyDataset_v2_x

        if isinstance(dataloader.dataset, DummyDataset) or isinstance(dataloader.dataset, DummyDataset_v2_x):
            assert isinstance(confidence_batches, int)
            ds = copy.deepcopy(dataloader.dataset)
            ds.dataset = ds.dataset[:confidence_batches]
            return ds
        else:
            return dataloader.dataset.take(confidence_batches)

    def _partial_dataloader(self, dataloader, confidence_batches):
        """Partial dataloader."""
        return type(dataloader)(
            dataset=self._partial_dataset_of(dataloader, confidence_batches),
            batch_size=dataloader.batch_size,
            last_batch=dataloader.last_batch,
            collate_fn=dataloader.collate_fn,
            sampler=dataloader.sampler,
            batch_sampler=dataloader.batch_sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            shuffle=dataloader.shuffle,
            distributed=dataloader.distributed,
        )

    def _calculate_mse(self, fp32_output, q_output):
        """MSE calculation."""
        result = []
        for i, j in zip(fp32_output, q_output):
            result.append(np.square(i - j).mean())
        return np.array(result).mean()

    def _inference_model_on_batches(self, model, tune_cfg, dataloader, output_op_names):
        """Inference model on batches."""
        from .tf_utils.util import generate_feed_dict

        input_tensors = model.input_tensor
        output_tensors = []
        for op in output_op_names:
            for tensor in model.graph.get_operation_by_name(op).outputs:
                output_tensors.append(tensor)

        predictions = []
        for index, (inputs, _) in enumerate(dataloader):
            feed_dict = generate_feed_dict(input_tensors, inputs)

            pred = model.sess.run(output_tensors, feed_dict)
            for item in pred:
                predictions.append(item)

        return predictions

    def smooth_quant(
        self,
        model,
        dataloader,
        calib_iter=1,
        alpha=0.5,
        folding=False,
        percentile=99.999,
        op_types=["MatMul", "Conv2D"],
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
        """Convert the model by smooth quant.

        Args:
            model: original model
            dataloader: the calibration dataloader
            calib_iter: how many steps of iterations on the dataloader to move forward
            alpha: smooth alpha in SmoothQuant, 1.0 will fallback to SPIQ
            folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant
            percentile: percentile of calibration to remove outliers
            op_types: The op types whose input tensor will be dumped
            scales_per_op: True, each op will have an individual scale, mainly for accuracy
                           False, ops with the same input will share a scale, mainly for performance
            record_max_info: whether record the max info in model for alpha tuning.
            weight_clip: Whether to clip weight when calculating scales; by default it is on.
            auto_alpha_args: Hyperparameters used to set the alpha search space in SQ auto-tuning.
                            By default the search space is 0.0-1.0 with step_size 0.1.
                            do_blockwise: Whether to do blockwise auto-tuning.
            default_alpha: A hyperparameter that is used in SQ auto-tuning; by default it is 0.5.

        Returns:
            model: A smoothed Tensorflow model
        """
        logger.info("Start Smoothing process for Smooth Quantization.")
        if self.smooth_quant_model is not None:
            return self.smooth_quant_model

        if model.model_type == "llm_saved_model":
            return self.smooth_quant_LLM(
                model, dataloader, calib_iter, alpha, folding, percentile, op_types, scales_per_op
            )

        # Do a pre-optimization before smooth quant
        from .tf_utils.graph_rewriter.generic.pre_optimize import PreOptimization

        self.pre_optimizer_handle = PreOptimization(model, self.new_api, self.device)
        self.pre_optimized_model = self.pre_optimizer_handle.get_optimized_model(self.itex_mode)
        model.graph_def = self.pre_optimized_model.graph_def

        # Run calibration to get max values per channel
        from .tf_utils.smooth_quant_calibration import SmoothQuantCalibration

        calibration = SmoothQuantCalibration(model, dataloader, calib_iter, op_types, percentile)
        max_vals_per_channel, sq_weight_node_names = calibration()

        # Get weight tensors and weight nodes based on the input tensor
        from neural_compressor.adaptor.tf_utils.util import get_weight_from_input_tensor

        sq_weight_tensors, sq_weights_nodes = get_weight_from_input_tensor(model, max_vals_per_channel.keys(), op_types)

        # Calculate the smooth quant scaler and insert Mul op into the graph
        from .tf_utils.smooth_quant_scaler import SmoothQuantScaler

        scaler = SmoothQuantScaler(model, dataloader, alpha, scales_per_op)
        model, mul_list = scaler.transform(
            max_vals_per_channel, sq_weight_tensors, sq_weights_nodes, sq_weight_node_names
        )
        self.smooth_quant_mul_ops.extend(mul_list)
        self.smooth_quant_model = model
        return self.smooth_quant_model

    def smooth_quant_LLM(
        self,
        model,
        dataloader,
        calib_iter=1,
        alpha=0.5,
        folding=False,
        percentile=99.999,
        op_types=["MatMul", "Conv2D"],
        scales_per_op=True,
    ):
        """Convert the model by smooth quant.

        Args:
            model: original model of TensorflowLLMModel object.
            calib_iter: how many steps of iterations on the dataloader to move forward.
            tune_cfg: quantization config.
            alpha: smooth alpha in SmoothQuant, 1.0 will fallback to SPIQ.
            folding: whether insert mul(False) or just allow foldable layers(True) for SmoothQuant.
            percentile: percentile of calibration to remove outliers.
            op_types: The op types whose input tensor will be dumped.
            scales_per_op: True, each op will have an individual scale, mainly for accuracy.
                           False, ops with the same input will share a scale, mainly for performance.

        Returns:
            model: A smoothed Tensorflow model.
        """
        # Do a pre-optimization before smooth quant
        from .tf_utils.graph_rewriter.generic.pre_optimize import PreOptimization

        self.pre_optimizer_handle = PreOptimization(model, self.new_api, self.device)
        self.pre_optimized_model = self.pre_optimizer_handle.get_optimized_model(self.itex_mode)
        model.graph_def = self.pre_optimized_model.graph_def

        # only support per-tensor MatMul now
        op_types = ["MatMul"]
        llm_temp_dir = self.work_dir + "/temp_saved_model"
        # Run calibration to get max values per channel
        from .tf_utils.smooth_quant_calibration import SmoothQuantCalibrationLLM

        calibration = SmoothQuantCalibrationLLM(
            model._model,
            dataloader,
            calib_iter,
            op_types,
            percentile,
            llm_temp_dir,
            model.weight_name_mapping,
        )
        max_vals_per_channel, sq_target_node_names, sq_weight_tensor_dict, sq_graph_def = calibration(
            model.input_node_names, model.output_node_names
        )

        # Calculate the smooth quant scaler and insert Mul op into the graph
        from .tf_utils.smooth_quant_scaler import SmoothQuantScalerLLM

        scaler = SmoothQuantScalerLLM(sq_graph_def, alpha, scales_per_op, op_types)
        sq_graph_def, sq_weight_scale_dict, mul_list = scaler.transform(
            max_vals_per_channel, sq_weight_tensor_dict, sq_target_node_names
        )
        model.graph_def = sq_graph_def
        model.model_path = llm_temp_dir
        model.sq_weight_scale_dict = sq_weight_scale_dict
        self.smooth_quant_mul_ops.extend(mul_list)
        self.smooth_quant_model = model
        return self.smooth_quant_model


@adaptor_registry
class Tensorflow_ITEXAdaptor(TensorFlowAdaptor):
    """Tensorflow ITEX Adaptor Class."""

    def __init__(self, framework_specific_info):
        """Initialization.

        Args:
            framework_specific_info: framework specific information.
        """
        super().__init__(framework_specific_info)

    @dump_elapsed_time("Pass quantize model")
    def quantize(self, tune_cfg, model, data_loader, q_func=None):
        """Execute the quantize process on the specified model.

        Args:
            tune_cfg (dict): quantization configuration
            model (tf.compat.v1.GraphDef): fp32 model
            data_loader (generator): generator the data and labels
            q_func (optional): training function for quantization aware training mode,
                                which not enabled for tensorflow yet.

        Returns:
            tf.compat.v1.GraphDef: the quantized model
        """
        assert q_func is None, "quantization aware training mode is not support on tensorflow"
        self._tuning_cfg_to_fw(tune_cfg)
        logger.debug("Dump quantization configurations:")
        logger.debug(self.quantize_config)
        from .tf_utils.graph_converter import GraphConverter

        calib_sampling_size = tune_cfg.get("calib_sampling_size", 1)
        if isinstance(data_loader, BaseDataLoader):
            batch_size = data_loader.batch_size
            try:
                for i in range(batch_size):
                    if calib_sampling_size % (batch_size - i) == 0:
                        calib_batch_size = batch_size - i
                        if i != 0:  # pragma: no cover
                            logger.warning(
                                "Reset `calibration.dataloader.batch_size` field "
                                "to {}".format(calib_batch_size) + " to make sure the sampling_size is "
                                "divisible exactly by batch size"
                            )
                        break
                tmp_iterations = int(math.ceil(calib_sampling_size / calib_batch_size))
                data_loader.batch(calib_batch_size)
                self.quantize_config["calib_iteration"] = tmp_iterations
                converted_model = GraphConverter(
                    model,
                    qt_config=self.quantize_config,
                    recipes=self.recipes,
                    int8_sequences=self.op_wise_sequences,
                    fp32_ops=self.fp32_ops,
                    bf16_ops=self.bf16_ops,
                    data_loader=data_loader,
                    calib_func=q_func,
                    itex_mode=self.itex_mode,
                    qdq_enabled=self.qdq_enabled,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    use_bf16=self.use_bf16,
                ).convert()
            except Exception:  # pragma: no cover
                from .tf_utils.util import get_model_input_shape

                batch_size = get_model_input_shape(model)
                logger.warning(
                    "Fail to forward with batch size={}, set to {} now.".format(data_loader.batch_size, batch_size)
                )
                data_loader.batch(batch_size)
                self.quantize_config["calib_iteration"] = calib_sampling_size
                converted_model = GraphConverter(
                    model,
                    qt_config=self.quantize_config,
                    recipes=self.recipes,
                    int8_sequences=self.op_wise_sequences,
                    fp32_ops=self.fp32_ops,
                    bf16_ops=self.bf16_ops,
                    data_loader=data_loader,
                    itex_mode=self.itex_mode,
                    qdq_enabled=self.qdq_enabled,
                    new_api=self.new_api,
                    performance_only=self.performance_only,
                    use_bf16=self.use_bf16,
                ).convert()
        else:  # pragma: no cover
            if hasattr(data_loader, "batch_size") and calib_sampling_size % data_loader.batch_size != 0:
                iter = self.quantize_config["calib_iteration"]
                logger.warning(
                    "Please note that calibration sampling size {} "
                    "isn't divisible exactly by batch size {}. "
                    "So the real sampling size is {}.".format(
                        calib_sampling_size, data_loader.batch_size, data_loader.batch_size * iter
                    )
                )
            converted_model = GraphConverter(
                model,
                qt_config=self.quantize_config,
                recipes=self.recipes,
                int8_sequences=self.op_wise_sequences,
                fp32_ops=self.fp32_ops,
                bf16_ops=self.bf16_ops,
                data_loader=data_loader,
                calib_func=q_func,
                itex_mode=self.itex_mode,
                qdq_enabled=self.qdq_enabled,
                new_api=self.new_api,
                performance_only=self.performance_only,
                use_bf16=self.use_bf16,
            ).convert()

        self._dump_model_op_stats(converted_model.graph_def)

        return converted_model


class TensorflowQuery(QueryBackendCapability):
    """Tensorflow Query Capability Class."""

    def __init__(self, local_config_file=None, performance_only=False, itex_mode=False, quant_mode="static"):
        """Initialization.

        Args:
            local_config_file: local configuration file name.
            performance_only: oob performance only mode.
            itex_mode: check if itex mode.
            quant_mode: quantization mode, static or dynamic.
        """
        import tensorflow as tf

        super().__init__()
        self.version = tf.version.VERSION
        self.cfg = local_config_file
        self.cur_config = None
        self.performance_only = performance_only
        self.quant_mode = quant_mode
        self.itex_mode = itex_mode
        self._one_shot_query()

    def _get_specified_version_cfg(self, data):
        """Get the configuration for the current runtime.

        If there's no matched configuration in the input yaml, we'll
        use the configuration of the nearest framework version field of yaml.

        Args:
            data (Yaml content): input yaml file.

        Returns:
            [dictionary]: the content for specific version.
        """
        from functools import cmp_to_key

        from pkg_resources import parse_version

        config = None

        def _compare(version1, version2):
            if parse_version(version1) == parse_version(version2):
                return 0
            elif parse_version(version1) < parse_version(version2):
                return -1
            else:
                return 1

        fallback_list = []
        for sub_data in data:
            if "default" in sub_data["version"]["name"]:
                assert config is None, "Only one default config " "is allowed in framework yaml file."
                config = sub_data

            if self.version in sub_data["version"]["name"]:
                return sub_data
            else:
                if sub_data["version"]["name"] == [
                    "2.11.0202242",
                    "2.11.0202250",
                    "2.11.0202317",
                    "2.11.0202323",
                    "2.14.0202335",
                    "2.14.dev202335",
                    "2.15.0202341",
                ]:
                    continue
                sorted_list = copy.deepcopy(sub_data["version"]["name"])
                sorted_list.remove("default") if "default" in sorted_list else None
                if isinstance(sorted_list, list):
                    # TensorFlow 1.15.0-up1/up2/up3 release versions are abnoraml release naming
                    # convention. Replacing them with dot for version comparison.
                    sorted_list = [i.replace("-up", ".") for i in sorted_list]
                    sorted_list = sorted(sorted_list, key=cmp_to_key(_compare), reverse=True)
                else:
                    assert isinstance(sorted_list, str)
                    sorted_list = list(sorted_list.replace("-up", ".").split())
                for i in sorted_list:
                    if parse_version(self.version) >= parse_version(i):
                        fallback_list.append([i, sub_data])
                        break

        assert config is not None, "The default config in framework yaml must exist."
        nearest_version = str(0)
        for fallback in fallback_list:
            if parse_version(fallback[0]) > parse_version(nearest_version):
                nearest_version = fallback[0]
                config = fallback[1]

        return config

    def _one_shot_query(self):
        """One short query for some patterns."""
        # pylint: disable=E1136
        with open(self.cfg) as f:
            content = yaml.safe_load(f)
            try:
                self.cur_config = self._get_specified_version_cfg(content)
                if not self.performance_only:
                    remove_int8_ops = [
                        "FusedBatchNorm",
                        "FusedBatchNormV2",
                        "FusedBatchNormV3",
                        "_MklFusedInstanceNorm",
                    ]
                    for op_type in remove_int8_ops:
                        while op_type in self.cur_config["int8"][self.quant_mode].keys():
                            self.cur_config["int8"][self.quant_mode].pop(op_type, None)

            except Exception as e:
                logger.info("Fail to parse {} due to {}.".format(self.cfg, str(e)))
                self.cur_config = None
                raise ValueError(
                    "Please check if the format of {} follows Neural Compressor yaml schema.".format(self.cfg)
                )

    def get_version(self):
        """Get the current backend version information.

        Returns:
            [string]: version string.
        """
        return self.cur_config["version"]["name"]

    def get_op_types(self):
        """Get the supported op types by all precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the op types.
        """
        return {
            "int8": self.get_op_types_by_precision("int8"),
            "uint8": self.get_op_types_by_precision("uint8"),
            "bf16": self.get_op_types_by_precision("bf16"),
        }

    def get_fuse_patterns(self):
        """Get supported patterns by low precisions.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is the supported patterns.
        """
        spr_int8_pattern_list = [
            "Conv2D + BiasAdd",
            "Conv2D + BiasAdd + Add + Relu6 + Mul + Mul",
            "Conv2D + Add + Relu6 + Mul + Mul",
            "Conv2D + BiasAdd + swish_f32",
            "Conv2D + Add + swish_f32",
            "Conv2D + AddV2 + swish_f32",
            "Conv2D + swish_f32",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + Relu",
            "Conv2D + BiasAdd + Elu",
            "Conv2D + Elu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd + LeakyRelu",
            "Conv2D + BiasAdd + Add + LeakyRelu",
            "Conv2D + BiasAdd + AddV2 + LeakyRelu",
            "Conv2D + Add + LeakyRelu",
            "Conv2D + AddV2 + LeakyRelu",
            "Conv2D + LeakyRelu",
            "Conv2D + BiasAdd + Sigmoid",
            "Conv2D + Sigmoid",
            "Conv2D + BiasAdd + LeakyRelu + AddV2",
            "Conv2D + BiasAdd + LeakyRelu + Add",
            "Conv2D + LeakyRelu + AddV2",
            "Conv2D + LeakyRelu + Add",
            "Conv2D + BiasAdd + Relu + AddV2",
            "Conv2D + BiasAdd + Relu + Add",
            "Conv2D + Relu + AddV2",
            "Conv2D + Relu + Add",
            "Conv2D + Add",
            "Conv2D + AddV2",
            "Conv2D + AddV2 + Add",
            "Conv2D + Add + Add",
            "Conv2D + BiasAdd + Add",
            "Conv3D + Add",
            "Conv3D + AddV2",
            "Conv3D + BiasAdd",
            "Conv3D + BiasAdd + Add",
            "Conv3D + BiasAdd + AddV2",
            "Conv3D + AddV2 + AddV2",
            "DepthwiseConv2dNative + BiasAdd + Add + Relu6 + Mul + Mul",
            "DepthwiseConv2dNative + Add + Relu6 + Mul + Mul",
            "DepthwiseConv2dNative + BiasAdd + swish_f32",
            "DepthwiseConv2dNative + Add + swish_f32",
            "DepthwiseConv2dNative + AddV2 + swish_f32",
            "DepthwiseConv2dNative + swish_f32",
            "DepthwiseConv2dNative + BiasAdd + LeakyRelu",
            "DepthwiseConv2dNative + LeakyRelu",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + Relu6",
            "DepthwiseConv2dNative + BiasAdd + Relu",
            "DepthwiseConv2dNative + Relu",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "FusedBatchNormV3 + Relu",
            "FusedBatchNormV3 + LeakyRelu",
            "_MklFusedInstanceNorm + Relu",
            "_MklFusedInstanceNorm + LeakyRelu",
            "Conv2DBackpropInput + BiasAdd",
            "Conv3DBackpropInputV2 + BiasAdd",
        ]

        spr_uint8_pattern_list = [
            "Conv2D + BiasAdd + AddN + Relu",
            "Conv2D + AddN + Relu",
            "Conv2D + BiasAdd + AddN + Relu6",
            "Conv2D + AddN + Relu6",
            "Conv2D + BiasAdd + AddV2 + Relu",
            "Conv2D + AddV2 + Relu",
            "Conv2D + BiasAdd + AddV2 + Relu6",
            "Conv2D + AddV2 + Relu6",
            "Conv2D + BiasAdd + Add + Relu",
            "Conv2D + Add + Relu",
            "Conv2D + BiasAdd + Add + Relu6",
            "Conv2D + Add + Relu6",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Relu",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd",
            "Conv2D + Add + Add + Relu",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + Relu6",
            "DepthwiseConv2dNative + BiasAdd + Relu",
            "DepthwiseConv2dNative + Relu",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "MatMul + BiasAdd",
            "MatMul + BiasAdd + Add",
            "MatMul + BiasAdd + AddV2",
            "MatMul + BiasAdd + Relu",
            "MatMul + BiasAdd + Relu6",
            "MatMul + BiasAdd + LeakyRelu",
            "MatMul + BiasAdd + Gelu",
            "MatMul + BiasAdd + Elu",
            "MatMul + BiasAdd + Tanh",
            "MatMul + BiasAdd + Sigmoid",
            "MatMul + Add",
            "MatMul + AddV2",
            "MatMul + Relu",
            "MatMul + Relu6",
            "MatMul + LeakyRelu",
            "MatMul + Gelu",
            "MatMul + Elu",
            "MatMul + Tanh",
            "MatMul + Sigmoid",
            "BatchMatMul + Mul",
            "BatchMatMulV2 + Mul",
            "BatchMatMul + Add",
            "BatchMatMulV2 + Add",
            "BatchMatMul + AddV2",
            "BatchMatMulV2 + AddV2",
            "BatchMatMul + Mul + Add",
            "BatchMatMulV2 + Mul + Add",
            "BatchMatMul + Mul + AddV2",
            "BatchMatMulV2 + Mul + AddV2",
            "Conv3D + AddV2 + AddV2 + Relu",
            "Conv3D + Add + Relu",
            "Conv3D + AddV2 + Relu",
            "Conv3D + Relu",
            "Conv3D + Relu6",
            "Conv3D + Add + Relu6",
            "Conv3D + AddV2 + Relu6",
            "Conv3D + Elu",
            "Conv3D + LeakyRelu",
            "Conv3D + BiasAdd + Relu",
            "Conv3D + BiasAdd + Relu6",
            "Conv3D + BiasAdd + Elu",
            "Conv3D + BiasAdd + LeakyRelu",
            "Conv3D + Add + Elu",
            "Conv3D + Add + LeakyRelu",
            "Conv2DBackpropInput + BiasAdd",
            "Conv3DBackpropInputV2 + BiasAdd",
        ]

        tf_int8_pattern_list = ["Conv2D + BiasAdd", "Conv2D + BiasAdd + Relu", "Conv2D + BiasAdd + Relu6"]
        tf_uint8_pattern_list = [
            "Conv2D + BiasAdd + AddN + Relu",
            "Conv2D + BiasAdd + AddN + Relu6",
            "Conv2D + BiasAdd + AddV2 + Relu",
            "Conv2D + BiasAdd + AddV2 + Relu6",
            "Conv2D + BiasAdd + Add + Relu",
            "Conv2D + BiasAdd + Add + Relu6",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Add + Relu",
            "Conv2D + Add + Relu6",
            "Conv2D + Relu",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + BiasAdd + Relu",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "MatMul + BiasAdd + Relu",
            "MatMul + BiasAdd",
        ]
        tf1_15_up3_int8_pattern_list = [
            "Conv2D + BiasAdd",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + LeakyRelu",
            "Conv2D + BiasAdd + LeakyRelu + AddV2",
            "Conv2D + BiasAdd + Relu6",
        ]
        tf1_15_up3_uint8_pattern_list = [
            "Conv2D + BiasAdd + AddN + Relu",
            "Conv2D + BiasAdd + AddN + Relu6",
            "Conv2D + BiasAdd + AddV2 + Relu",
            "Conv2D + BiasAdd + AddV2 + Relu6",
            "Conv2D + BiasAdd + Add + Relu",
            "Conv2D + BiasAdd + Add + Relu6",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Add + Relu",
            "Conv2D + Add + Relu6",
            "Conv2D + Relu",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "MatMul + BiasAdd + Relu",
            "MatMul + BiasAdd",
        ]
        old_tf_int8_pattern_list = ["MatMul + BiasAdd + Relu", "MatMul + BiasAdd"]
        old_tf_uint8_pattern_list = [
            "Conv2D + BiasAdd + AddN + Relu",
            "Conv2D + BiasAdd + AddN + Relu6",
            "Conv2D + BiasAdd + AddV2 + Relu",
            "Conv2D + BiasAdd + AddV2 + Relu6",
            "Conv2D + BiasAdd + Add + Relu",
            "Conv2D + BiasAdd + Add + Relu6",
            "Conv2D + BiasAdd + Relu",
            "Conv2D + BiasAdd + Relu6",
            "Conv2D + Add + Relu",
            "Conv2D + Add + Relu6",
            "Conv2D + Relu",
            "Conv2D + Relu6",
            "Conv2D + BiasAdd",
            "DepthwiseConv2dNative + BiasAdd + Relu6",
            "DepthwiseConv2dNative + Add + Relu6",
            "DepthwiseConv2dNative + BiasAdd",
            "MatMul + BiasAdd + Relu",
            "MatMul + BiasAdd",
        ]

        for index, pattern in enumerate(spr_int8_pattern_list):
            spr_int8_pattern_list[index] = "Dequantize + " + pattern + " + QuantizeV2"
        for index, pattern in enumerate(spr_uint8_pattern_list):
            spr_uint8_pattern_list[index] = "Dequantize + " + pattern + " + QuantizeV2"

        if not self.performance_only:
            remove_int8_ops = ["FusedBatchNorm", "FusedBatchNormV2", "FusedBatchNormV3", "_MklFusedInstanceNorm"]
            for op_type in remove_int8_ops:
                patterns = [
                    f"Dequantize + {op_type} + Relu + QuantizeV2",
                    f"Dequantize + {op_type} + LeakyRelu + QuantizeV2",
                ]
                for pattern in patterns:
                    while pattern in spr_int8_pattern_list:
                        spr_int8_pattern_list.remove(pattern)
                    while pattern in spr_uint8_pattern_list:
                        spr_uint8_pattern_list.remove(pattern)

        patterns = {}
        import tensorflow as tf

        if tf.version.VERSION in spr_base_verions or self.itex_mode:
            patterns["int8"] = spr_int8_pattern_list
            patterns["uint8"] = spr_uint8_pattern_list
        elif version1_gte_version2(tf.version.VERSION, "2.1.0"):
            patterns["int8"] = tf_int8_pattern_list
            patterns["uint8"] = tf_uint8_pattern_list
            if self.itex_mode:
                patterns["int8"].append("FusedBatchNormV3 + Relu")
                patterns["int8"].append("FusedBatchNormV3 + LeakyRelu")
        elif version1_eq_version2(tf.version.VERSION, "1.15.0-up3"):
            patterns["int8"] = tf1_15_up3_int8_pattern_list
            patterns["uint8"] = tf1_15_up3_uint8_pattern_list
        else:
            patterns["int8"] = old_tf_int8_pattern_list
            patterns["uint8"] = old_tf_uint8_pattern_list

        return patterns

    def get_quantization_capability(self):
        """Get the supported op types' quantization capability.

        Returns:
            [dictionary list]: A list composed of dictionary which key is precision
            and value is a dict that describes all op types' quantization capability.
        """
        for op_type, _ in self.cur_config["int8"][self.quant_mode].items():
            self.cur_config["int8"][self.quant_mode][op_type]["activation"]["quant_mode"] = self.quant_mode
        return self.cur_config["int8"][self.quant_mode]

    def get_op_types_by_precision(self, precision):
        """Get op types per precision.

        Args:
            precision (string): precision name

        Returns:
            [string list]: A list composed of op type.
        """
        assert precision in ("bf16", "uint8", "int8")

        import tensorflow as tf

        if precision == "int8":
            if tf.version.VERSION in spr_base_verions or self.itex_mode:
                op_type_list = [key for key in self.cur_config["int8"][self.quant_mode].keys()]
                if not self.performance_only and not self.itex_mode:
                    remove_int8_ops = [
                        "FusedBatchNorm",
                        "FusedBatchNormV2",
                        "FusedBatchNormV3",
                        "_MklFusedInstanceNorm",
                    ]
                    for op_type in remove_int8_ops:
                        while op_type in op_type_list:
                            op_type_list.remove(op_type)
                return op_type_list
            if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(
                tf.version.VERSION, "1.15.0-up3"
            ):
                return ["Conv2D", "MatMul", "ConcatV2", "MaxPool", "AvgPool"]
            return ["MatMul", "ConcatV2", "MaxPool", "AvgPool"]
        if precision == "uint8":
            if tf.version.VERSION in spr_base_verions:
                return [key for key in self.cur_config["int8"][self.quant_mode].keys() if "Norm" not in key]
            if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(
                tf.version.VERSION, "1.15.0-up3"
            ):
                return ["Conv2D", "MatMul", "ConcatV2", "MaxPool", "AvgPool", "DepthwiseConv2dNative"]
            return ["Conv2D", "MatMul", "ConcatV2", "MaxPool", "AvgPool"]
        if precision == "bf16":
            if tf.version.VERSION in spr_base_verions:
                return self.cur_config[precision]
            if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(
                tf.version.VERSION, "1.15.0-up3"
            ):
                return self.cur_config[precision]
            return []

    def get_mixed_precision_combination(self):
        """Get the valid mixed precisions.

        Returns:
            [string list]: valid precision list.
        """
        import tensorflow as tf

        if version1_gte_version2(tf.version.VERSION, "2.1.0") or version1_eq_version2(tf.version.VERSION, "1.15.0-up3"):
            return ["int8", "uint8", "bf16", "fp32"]
        return ["uint8", "fp32"]

    def get_bf16_patterns(self):
        """Get BF16 pattern list.

        Returns:
            [List]: bf16 pattern list.
        """
        bf16_op_types = [i for i in self.get_op_types_by_precision("bf16")]
        res = []
        for i in bf16_op_types:
            res.append([[i]])

        return res

    def get_eightbit_patterns(self, qdq_enabled=False):
        """Get eightbit op wise sequences information.

        Returns:
            [dictionary]: key is the op type while value is the list of sequences start
                        with the op type same as key value.
        """
        quantizable_op_types = self.get_op_types_by_precision("int8") + self.get_op_types_by_precision("uint8")
        int8_patterns = [
            i.replace("+", " ").split()
            for i in list(set(self.get_fuse_patterns()["int8"] + self.get_fuse_patterns()["uint8"]))
        ]
        res = {}
        for i in quantizable_op_types:
            if qdq_enabled:
                res[i] = [["Dequantize", i, "QuantizeV2"]]
            else:
                res[i] = [[i]]

        for pattern in int8_patterns:
            if qdq_enabled:
                op_type = pattern[1]
            else:
                op_type = pattern[0]
            if op_type in res:
                res[op_type].append(pattern)

        return res

    def generate_internal_patterns(self):
        """Translate the patterns defined in the yaml to internal pattern expression."""

        def _generate_pattern(data):
            length = [len(i) for i in data]
            res = []
            for index in range(max(length)):
                if index <= min(length) - 1:
                    tmp = [i[index] for i in data]
                    if len(set(tmp)) == 1:
                        res.append([tmp[0]])
                    else:
                        res.append(tuple(set(tmp)))
                else:
                    tmp1 = [i[index] for i in data if len(i) > index]
                    res.append(tuple(set(tmp1)))

            return res

        op_level_sequences = {}

        for k, op_level_all_sequences in self.get_eightbit_patterns().items():
            op_level_sequences[k] = []
            sorted_sequences = sorted(op_level_all_sequences)
            last_len = 1
            each_combination = []
            for index, value in enumerate(sorted_sequences):
                if len(value) >= last_len:
                    last_len = len(value)
                    each_combination.append(value)
                else:
                    op_level_sequences[k].append(copy.deepcopy(each_combination))
                    each_combination.clear()
                    each_combination.append(value)
                    last_len = len(value)

                if index == len(sorted_sequences) - 1:
                    op_level_sequences[k].append(copy.deepcopy(each_combination))

        final_out = []
        for _, op_level_sequences in op_level_sequences.items():
            for similar_sequences in op_level_sequences:
                final_out.append(_generate_pattern(similar_sequences))

        return final_out
