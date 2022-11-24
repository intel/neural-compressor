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
import logging
import tensorflow as tf
from onnx import helper
from tensorflow.core.framework import tensor_pb2, node_def_pb2

from neural_compressor.adaptor.tf_utils.graph_util import GraphAnalyzer
from neural_compressor.utils.utility import dump_elapsed_time
from .graph_rewriter.onnx import tf2onnx_utils as utils
from .graph_rewriter.onnx.onnx_graph import OnnxGraph


logger = logging.getLogger("neural_compressor")

class TensorflowQDQToOnnxQDQConverter:
    """Convert tensorflow QDQ graph to ONNX QDQ graph."""
    def __init__(self, model, input_names, output_names, opset_version=utils.DEFAULT_OPSET_VERSION):
        """constructor

        Args:
            model (graphdef): tensorflow QDQ graphdef
        """
        graph_def = self.pre_optimize(model)
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def)

        self.graph = graph
        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names

    def duplicate_quantizev2_nodes(self, model):
        """Duplicate QuantizeV2 nodes if the Dequantize nodes share the same QuantizeV2."""
        cur_graph = GraphAnalyzer()
        cur_graph.graph = model
        graph_info = cur_graph.parse_graph()

        # Scan the QDQ pairs
        patterns = [['QuantizeV2'], ['Dequantize']]
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
                    new_quantizev2_node.name = quantizev2_nodes[0].name + '_copy_' + str(index+1)
                    cur_graph.add_node(
                        new_quantizev2_node, input_map_node_name, [dequantize_node.name])
                    cur_graph.node_name_details[dequantize_node.name].node.ClearField('input')
                    cur_graph.node_name_details[dequantize_node.name].node.input.extend([
                        new_quantizev2_node.name, new_quantizev2_node.name + ':1',
                        new_quantizev2_node.name + ':2'])

        return cur_graph.dump_graph()

    def pre_optimize(self, model):
        """Pre optimize the tensorflow graphdef to make ONNX QDQ model convert more easier."""
        # Convert HostConst to Const
        for node in model.node:
            if node.op == 'HostConst':
                node.op = 'Const'

        # Duplicat the QuantizeV2 node if it has multi Dequantize nodes
        model = self.duplicate_quantizev2_nodes(model)
        return model

    @dump_elapsed_time("Pass TensorflowQDQToOnnxQDQConverter")
    def convert(self, save_path):
        """ convert tensorflow QDQ model to onnx QDQ model

        Args:
          input_graph_def (graphdef): tensorflow QDQ graphdef object

        Returns:
           onnx QDQ graph
        """
        onnx_nodes = []
        output_shapes = {}
        dtypes = {}
        functions = {}
        logger.info("Using ONNX opset %s", self.opset_version)

        node_list = self.graph.get_operations()

        # create dict with output to shape mappings
        for node in node_list:
            for out in node.outputs:
                shape = utils.get_tensorflow_tensor_shape(out)
                dtypes[out.name] = utils.map_tensorflow_dtype(out.dtype)
                output_shapes[out.name] = shape

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
                    onnx_tensor = utils.convert_tensorflow_tensor_to_onnx(
                        value, name=utils.add_port_to_name(node.name))
                    attr_dict[each_attr] = onnx_tensor
            node_type = node.type
            node_input_names = [i.name for i in node.inputs]
            node_output_names = [i.name for i in node.outputs]

            if convert_to_onnx:
                try:
                    onnx_node = helper.make_node(node_type, node_input_names, node_output_names,
                                                name=node.name, **attr_dict)
                    onnx_nodes.append(onnx_node)
                except Exception as ex:
                    logger.error("tf2onnx node convert failed for %s, ex=%s", node.name, ex)
                    raise

        # Build ONNX Graph using onnx_nodes, output_shapes and dtypes
        onnx_graph = OnnxGraph(onnx_nodes, output_shapes, dtypes)

        # Convert TF QDQ pattern to ONNX QDQ format
        for node in onnx_graph.get_nodes():
            if node.type == 'Dequantize':
                parent_node = onnx_graph.get_node_by_name(node.input[0].rsplit(':', 1)[0])
                if parent_node:
                    if parent_node.type == 'QuantizeV2':
                        onnx_graph.convert_qdq_nodes(parent_node, node)

        # Build ONNX model
        model_proto = onnx_graph.make_model("converted from neural compressor")
        # Save ONNX model
        utils.save_protobuf(save_path, model_proto)

        logger.info("Model inputs: %s", [n.name for n in model_proto.graph.input])
        logger.info("Model outputs: %s", [n.name for n in model_proto.graph.output])
