
import argparse
import os
import sys
import re

from utils import *


unlikely_output_types = ['Const', 'Assign', 'NoOp', 'Parameter', 'Assert', 'Postprocessor', 'Batch', 'Preprocessor', 'save', \
    'global_step', 'Conv2d', 'read', 'switch', 'gradient', 'cond', 'train', 'detection_masks', 'detection_classes', 'xmax', 'xmin', 'ymax', 'ymin', \
        'init_op', 'Merge', 'batch', 'SparseToDense', 'init_ops', 'RMSProp', 'transpose', 'ApplyAdam', ]

def summarize_graph(graph_def, fix_dynamic_shape):
    placeholders = dict()
    outputs = list()
    graph = tf_v1.Graph()
    with graph.as_default():  # pylint: disable=not-context-manager
        tf_v1.import_graph_def(graph_def, name='')
    for node in graph.as_graph_def().node:  # pylint: disable=no-member
        if node.op == 'Placeholder':
            node_dict = dict()
            node_dict['type'] = tf_v1.DType(node.attr['dtype'].type).name

            if node_dict['type'] != 'bool':
                # convert shape to list
                try:
                    _shape = list(tf_v1.TensorShape(node.attr['shape'].shape))
                    if tf_v1.__version__ >= '2.0.0':
                        node_dict['shape'] = [item if item != None else fix_dynamic_shape for item in _shape]
                    else:
                        node_dict['shape'] = [item.value if item.value != None else fix_dynamic_shape for item in _shape]
                    # if shape dimension > 1, suppose first dimension is batch-size
                    if len(node_dict['shape']) > 1: node_dict['shape'] = node_dict['shape'][1:]
                except ValueError as e:
                    print(str(e))
                    _shape = [fix_dynamic_shape, fix_dynamic_shape, 3]
                    node_dict['shape'] = _shape

            else:   # deal with bool dtype inputs, now assign bool dtype input False value
                node_dict['shape'] = None
                node_dict['value'] = False

            print("********** Find input node: {}".format(node_dict))
            placeholders[node.name] = node_dict

        if len(children(node.name, graph)) == 0:
            is_output = True
            if node.op in unlikely_output_types or node.name.split('/')[-1] in unlikely_output_types:
                is_output = False
            for item in node.name.split('/'):
                for unlikely_pattern in unlikely_output_types:
                    if unlikely_pattern in item:
                        is_output = False
            if is_output:
                print("********** Find output node: {}".format(node.name))
                outputs.append(node.name)

    result = dict()
    result['inputs'] = placeholders
    result['outputs'] = outputs
    return result

def _load_pb(graph_def: [tf_v1.GraphDef, tf_v1.MetaGraphDef], graph_file_name: str = "",
                           is_binary: bool = True):
    """
    Reads file to protobuf
    :param graph_def: GraphDef orr MetaGraphDef object to store the network
    :param graph_file_name: path to file with graph
    :param is_binary: flag to switch between binary and test protobuf format of graph file
    :return: GraphDef or MetaGaphDef containing the network with cleared device info.
    """
    try:
        if is_binary:
            with open(graph_file_name, "rb") as f:
                graph_def.ParseFromString(f.read())
        else:
            with open(graph_file_name, "r") as f:
                text_format.Merge(f.read(), graph_def)
        nodes_to_clear_device = graph_def.node if isinstance(graph_def, tf_v1.GraphDef) else graph_def.graph_def.node
        for node in nodes_to_clear_device:
            node.device = ""
    except Exception as e:
        print(str(e))
    graph_def = delete_assign(graph_def)
    return graph_def

def children(op_name: str, graph: tf_v1.Graph):
    op = graph.get_operation_by_name(op_name)
    return set(op for out in op.outputs for op in out.consumers())

def _load_protobuf_from_file(container, filename):
    with open(filename, 'rb') as fin:
        file_content = fin.read()

    # First try to read it as a binary file.
    try:
        container.ParseFromString(file_content)
        print("Parse file [%s] with binary format successfully." % (filename))
        return container

    except Exception as e:  # pylint: disable=broad-except
        print ("Info: Trying to parse file [%s] with binary format but failed with error [%s]." % (filename, str(e)))

    # Next try to read it as a text file.
    try:
        from google.protobuf import text_format
        text_format.Parse(file_content.decode('UTF-8'), container, allow_unknown_extension=True)
        print("Parse file [%s] with text format successfully." % (filename))
    except text_format.ParseError as e:
        raise IOError("Cannot parse file %s: %s." % (filename, str(e)))

    return container

def _load_meta(model_network_path):
        """Load a tensorflow meta file from disk
        Parameters
        ----------
        model_network_path: str
            Path where the model network path is (protobuf meta file)
        Returns
        -------
        model: A tensorflow protobuf file
        """
        from tensorflow.core.protobuf import meta_graph_pb2

        meta_graph = meta_graph_pb2.MetaGraphDef()
        _load_protobuf_from_file(meta_graph, model_network_path)
        graph = meta_graph.graph_def

        print ("Tensorflow model file [%s] loaded successfully." % model_network_path)
        return graph

def get_input_output(graph_path, input_meta=False):
    # give a fix shape if not get input shape 
    fix_dynamic_shape = 300

    if input_meta:
        # meta file
        graph_def = _load_meta(graph_path)
        output_nodes = summarize_graph(graph_def, fix_dynamic_shape)
        graph_prefix = graph_path[:-5]
        output_freeze_model_dir = graph_prefix + "_freeze.pb"
        output_graph_def = freeze_graph(input_checkpoint=graph_prefix, output_graph=output_freeze_model_dir, output_node_names=output_nodes['outputs'])
        print("****** {} is a ckpt model, now save freezed model at {}".format(graph_path, output_freeze_model_dir))
        # output_nodes = summarize_graph(output_graph_def, fix_dynamic_shape)
    else:
        graph_def = tf_v1.GraphDef()
        load_graph = _load_pb(graph_def, graph_file_name=graph_path)
        output_nodes = summarize_graph(load_graph, fix_dynamic_shape)

    return output_nodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", type=str, help="Path to tensorflow model", default="")
    parser.add_argument('--input_meta', action='store_true',
                        help='TensorFlow*: treat the input model file as a meta graph def format', default=False)
    args = parser.parse_args()

    graph_path = args.input_model
    res = get_input_output(graph_path, input_meta=args.input_meta)

    print("output nodes name is: {}".format(res['outputs']))
    print("input nodes info is: {}".format(res['inputs']))