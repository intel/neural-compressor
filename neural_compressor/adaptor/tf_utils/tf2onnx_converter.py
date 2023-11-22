#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2022 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Tensorflow QDQ model convert to ONNX QDQ model."""

import logging

import numpy as np
import tensorflow as tf
from onnx import helper
from packaging.version import Version
from tensorflow.core.framework import node_def_pb2, tensor_pb2

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.utils.utility import LazyImport, dump_elapsed_time

from .graph_rewriter.onnx import tf2onnx_utils as utils
from .graph_rewriter.onnx.onnx_graph import OnnxGraph

t2o = LazyImport("tf2onnx")

logger = logging.getLogger("neural_compressor")


class TensorflowQDQToOnnxQDQConverter:
    """Convert tensorflow QDQ graph to ONNX QDQ graph."""

    def __init__(
        self,
        model,
        input_names,
        output_names,
        shape_override,
        inputs_as_nchw=None,
        opset_version=utils.DEFAULT_OPSET_VERSION,
    ):
        """Constructor, initialization.

        Args:
            model (graphdef): tensorflow QDQ graphdef
            input_names (list, optional): input names. Defaults to None.
            output_names (list, optional): output names. Defaults to None.
            shape_override: dict with inputs that override the shapes given by tensorflow.
            opset_version (int, optional): opset version. Defaults to 14.
            inputs_as_nchw (list, optional): transpose the input. Defaults to None.
        """
        graph_def = self.tf_graph_optimize(model)

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.graph = graph
        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names
        self.shape_override = shape_override
        self.inputs_as_nchw = inputs_as_nchw

        if self.shape_override:
            logger.info("Apply shape override:")
            for name, shape in self.shape_override.items():
                logger.info("\tSet %s shape to %s", name, shape)
                self.graph.get_tensor_by_name(name).set_shape(shape)
                graph_def = self.graph.as_graph_def(add_shapes=True)
                with tf.Graph().as_default() as inferred_graph:
                    tf.import_graph_def(graph_def, name="")
                self.graph = inferred_graph

    def duplicate_tf_quantizev2_nodes(self, model):
        """Duplicate QuantizeV2 nodes if the Dequantize nodes share the same QuantizeV2."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = model
        graph_info = cur_graph.parse_graph()

        # Scan the QDQ pairs
        patterns = [["QuantizeV2"], ["Dequantize"]]
        matched_nodes = cur_graph.query_fusion_pattern_nodes(patterns)

        # Append the QDQ pairs to QuantizeV2 nodes map and Dequantize nodes map
        quantizev2_map = {}
        dequantize_map = {}

        for i in matched_nodes:
            quantizev2_input_node_name = graph_info[i[0]].node.input[0]
            if quantizev2_input_node_name in quantizev2_map:
                quantizev2_map[quantizev2_input_node_name].append(graph_info[i[0]].node)
                dequantize_map[quantizev2_input_node_name].append(graph_info[i[1]].node)
            else:
                quantizev2_map[quantizev2_input_node_name] = [graph_info[i[0]].node]
                dequantize_map[quantizev2_input_node_name] = [graph_info[i[1]].node]

        # Find out the QuantizeV2 nodes which needs to be duplicated
        for input_map_node_name, quantizev2_nodes in quantizev2_map.items():
            if input_map_node_name not in cur_graph.node_name_details:
                continue

            dequantize_nodes = dequantize_map[input_map_node_name]
            if len(dequantize_nodes) == 1:
                continue

            do_duplicate = True
            quantizev2_node_name = quantizev2_nodes[0].name
            for index, node in enumerate(quantizev2_nodes):
                if index == 0:
                    continue
                if node.name != quantizev2_node_name:
                    do_duplicate = False

            # Duplicate the QuantizeV2 nodes
            if do_duplicate:
                for index in range(len(dequantize_nodes) - 1):
                    dequantize_node = dequantize_nodes[index + 1]
                    new_quantizev2_node = node_def_pb2.NodeDef()
                    new_quantizev2_node.CopyFrom(quantizev2_nodes[0])
                    new_quantizev2_node.name = quantizev2_nodes[0].name + "_copy_" + str(index + 1)
                    cur_graph.add_node(new_quantizev2_node, input_map_node_name, [dequantize_node.name])
                    cur_graph.node_name_details[dequantize_node.name].node.ClearField("input")
                    cur_graph.node_name_details[dequantize_node.name].node.input.extend(
                        [new_quantizev2_node.name, new_quantizev2_node.name + ":1", new_quantizev2_node.name + ":2"]
                    )

        return cur_graph.dump_graph()

    def tf_graph_optimize(self, model):
        """Pre optimize the tensorflow graphdef to make ONNX QDQ model convert more easier."""
        # Convert HostConst to Const
        for node in model.node:
            if node.op == "HostConst":
                node.op = "Const"

        # Duplicate the QuantizeV2 node if it has multi Dequantize nodes
        model = self.duplicate_tf_quantizev2_nodes(model)
        return model

    def transpose_inputs(self, ctx, inputs_as_nchw):
        """Insert a transpose from NHWC to NCHW on model input on users request."""
        ops = []
        for node in ctx.get_nodes():
            for _, output_name in enumerate(node.output):
                if output_name in inputs_as_nchw:
                    shape = ctx.get_shape(output_name)
                    if len(shape) != len(utils.NCHW_TO_NHWC):
                        logger.warning("transpose_input for %s: shape must be rank 4, ignored" % output_name)
                        ops.append(node)
                        continue
                    # insert transpose
                    op_name = utils.set_name(node.name)
                    transpose = ctx.insert_new_node_on_output("Transpose", output_name, name=op_name)
                    transpose.set_attr("perm", utils.NCHW_TO_NHWC)
                    ctx.copy_shape(output_name, transpose.output[0])
                    ctx.set_shape(output_name, np.array(shape)[utils.NHWC_TO_NCHW])
                    ops.append(transpose)
                    ops.append(node)
                    continue
            ops.append(node)
        ctx.reset_nodes(ops)

    @dump_elapsed_time("Pass TensorflowQDQToOnnxQDQConverter")
    def convert(self, save_path):
        """Convert tensorflow QDQ model to onnx QDQ model.

        Args:
          save_path (str): save path of ONNX QDQ model.
        """
        onnx_nodes = []
        output_shapes = {}
        dtypes = {}
        functions = {}
        logger.info("Using ONNX opset %s", self.opset_version)

        self.graph = t2o.shape_inference.infer_shape_for_graph(self.graph)

        op_outputs_with_none_shape = t2o.shape_inference.check_shape_for_tf_graph(self.graph)
        if op_outputs_with_none_shape:
            if Version(tf.__version__) > Version("1.5.0"):
                for op, outs in op_outputs_with_none_shape.items():
                    logger.warning("Cannot infer shape for %s: %s", op, ",".join(outs))
            self.graph = t2o.shape_inference.infer_shape_for_graph_legacy(self.graph)

        node_list = self.graph.get_operations()

        outputs_to_values, _ = utils.compute_const_folding_using_tf(self.graph, None, self.output_names)

        # Create dict with output to shape mappings
        for node in node_list:
            for out in node.outputs:
                shape = None
                if self.shape_override:
                    shape = self.shape_override.get(out.name)
                if shape is None:
                    shape = utils.get_tensorflow_tensor_shape(out)

                dtypes[out.name] = utils.map_tensorflow_dtype(out.dtype)
                output_shapes[out.name] = shape
                if output_shapes[out.name] is None:
                    output_shapes[out.name] = []

        # Convert the TF FP32 node to ONNX FP32 node
        for node in node_list:
            attr_dict = utils.read_tensorflow_node_attrs(node)
            convert_to_onnx = True
            for each_attr in node.node_def.attr:
                value = utils.get_tensorflow_node_attr(node, each_attr)
                if each_attr == "T":
                    if value and not isinstance(value, list):
                        dtypes[node.name] = utils.map_tensorflow_dtype(value)
                elif each_attr in utils.TF2ONNX_SUBGRAPH_ATTRS:
                    input_shapes = [input.get_shape() for input in node.inputs]
                    nattr = utils.get_tensorflow_node_attr(node, each_attr)
                    attr_dict[each_attr] = nattr.name
                    functions[nattr.name] = input_shapes
                elif isinstance(value, tensor_pb2.TensorProto):
                    onnx_tensor = utils.convert_tensorflow_tensor_to_onnx(value, name=utils.add_port_to_name(node.name))
                    attr_dict[each_attr] = onnx_tensor
            node_type = node.type
            node_input_names = [i.name for i in node.inputs]
            node_output_names = [i.name for i in node.outputs]

            if convert_to_onnx:
                try:
                    onnx_node = helper.make_node(
                        node_type, node_input_names, node_output_names, name=node.name, **attr_dict
                    )
                    onnx_nodes.append(onnx_node)
                except Exception as ex:
                    logger.error("tensorflow node convert to onnx failed for %s, ex=%s", node.name, ex)
                    raise

        # Build ONNX Graph using onnx_nodes, output_shapes and dtypes
        onnx_graph = OnnxGraph(
            onnx_nodes, output_shapes, dtypes, input_names=self.input_names, output_names=self.output_names
        )
        t2o.tfonnx.fold_constants_using_tf(onnx_graph, outputs_to_values)

        if self.inputs_as_nchw:
            self.transpose_inputs(onnx_graph, self.inputs_as_nchw)

        # Convert TF QDQ pattern to ONNX QDQ format
        for node in onnx_graph.get_nodes():
            if node.type == "Dequantize":
                parent_node = onnx_graph.get_node_by_name(node.input[0].rsplit(":", 1)[0])
                if parent_node:
                    if parent_node.type == "QuantizeV2":
                        onnx_graph.convert_qdq_nodes(parent_node, node)

        # Create ops mapping for the desired opsets
        ops_mapping = t2o.handler.tf_op.create_mapping(onnx_graph.opset, onnx_graph.extra_opset)

        # Run tf2onnx rewriters
        rewriters = [
            # single directional
            t2o.tfonnx.rewrite_constant_fold,
            t2o.rewriter.rewrite_transpose,
            t2o.rewriter.rewrite_flatten,
            t2o.rewriter.rewrite_random_uniform,
            t2o.rewriter.rewrite_random_uniform_fold_const,
            t2o.rewriter.rewrite_random_normal,
            t2o.rewriter.rewrite_dropout,
            t2o.rewriter.rewrite_conv_dilations,
            t2o.rewriter.rewrite_eye,
            t2o.rewriter.rewrite_leakyrelu,
            t2o.rewriter.rewrite_thresholded_relu,
            t2o.rewriter.rewrite_conv2d_with_pad,
            t2o.rewriter.rewriter_lstm_tf2,
            t2o.rewriter.rewrite_gru_tf2,
            t2o.rewriter.rewrite_single_direction_lstm,
            # bi-directional
            t2o.rewriter.rewrite_bi_direction_lstm,
            t2o.rewriter.rewrite_single_direction_gru,
            t2o.rewriter.rewrite_bi_direction_gru,
            t2o.rewriter.rewrite_custom_rnn_cell,
            t2o.rewriter.rewrite_generic_loop,
            t2o.rewriter.rewrite_cond,
            # rewrite_biasadd_with_conv2d introduces accuracy issue
            # t2o.rewriter.rewrite_biasadd_with_conv2d,
            t2o.rewriter.rewrite_layer_normalization,
            t2o.rewriter.rewrite_gemm,
            t2o.rewriter.rewrite_ragged_variant_shape,
        ]

        t2o.tfonnx.run_rewriters(onnx_graph, rewriters, False)

        # Some nodes may already copied into inner Graph, so remove them from main Graph.
        onnx_graph.delete_unused_nodes(onnx_graph.outputs)
        t2o.tfonnx.topological_sort(onnx_graph, False)

        mapped_op, unmapped_op, exceptions = t2o.tfonnx.tensorflow_onnx_mapping(onnx_graph, ops_mapping)
        if unmapped_op:
            logger.error("Unsupported ops: %s", unmapped_op)
        if exceptions:
            raise exceptions[0]

        # onnx requires topological sorting
        t2o.tfonnx.topological_sort(onnx_graph, False)

        onnx_graph.update_proto()

        op_cnt, attr_cnt = onnx_graph.dump_node_statistics(include_attrs=True, include_subgraphs=False)
        logger.info(
            "Summary Stats:\n"
            "\ttensorflow ops: {}\n"
            "\ttensorflow attr: {}\n"
            "\tonnx mapped: {}\n"
            "\tonnx unmapped: {}".format(op_cnt, attr_cnt, mapped_op, unmapped_op)
        )

        onnx_graph = t2o.optimizer.optimize_graph(onnx_graph)

        # some nodes may already copied into inner Graph, so remove them from main Graph.
        onnx_graph.delete_unused_nodes(onnx_graph.outputs)
        t2o.tfonnx.topological_sort(onnx_graph, False)

        onnx_graph = onnx_graph.apply_onnx_fusion()

        # Build ONNX model
        model_proto = onnx_graph.make_model("converted from neural compressor")
        # Save ONNX model
        utils.save_protobuf(save_path, model_proto)

        logger.info("Model inputs: %s", [n.name for n in model_proto.graph.input])
        logger.info("Model outputs: %s", [n.name for n in model_proto.graph.output])
