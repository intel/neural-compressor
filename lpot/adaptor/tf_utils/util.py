#
#  -*- coding: utf-8 -*-
#
#  Copyright (c) 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

from google.protobuf import text_format
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from lpot.utils import logger


def read_graph(in_graph, in_graph_is_binary=True):
    """Reads input graph file as GraphDef.

    :param in_graph: input graph file.
    :param in_graph_is_binary: whether input graph is binary, default True.
    :return: input graphDef.
    """
    if not gfile.Exists(in_graph):
        raise ValueError('Input graph pb file %s does not exist.' % in_graph)

    input_graph_def = graph_pb2.GraphDef()
    mode = "rb" if in_graph_is_binary else "r"
    with gfile.Open(in_graph, mode) as f:
        data = f.read()
        if in_graph_is_binary:
            input_graph_def.ParseFromString(data)
        else:
            text_format.Merge(data, input_graph_def)

    return input_graph_def


def write_graph(out_graph_def, out_graph_file):
    """Write output graphDef to file.

    :param out_graph_def: output graphDef.
    :param out_graph_file: path to output graph file.
    :return: None.
    """
    if not isinstance(out_graph_def, tf.compat.v1.GraphDef):
        raise ValueError(
            'out_graph_def is not instance of TensorFlow GraphDef.')
    if out_graph_file and not os.path.exists(os.path.dirname(out_graph_file)):
        raise ValueError('"output_graph" directory does not exists.')
    f = gfile.GFile(out_graph_file, 'wb')
    f.write(out_graph_def.SerializeToString())


def is_ckpt_format(model_path):
    """check the model_path format is ckpt or not.

    Args:
        model_path (string): the model folder path

    Returns:
        string: return the ckpt prefix if the model_path contains ckpt format data else None.
    """
    file_list = [os.path.splitext(i)[-1] for i in os.listdir(model_path)]
    if file_list.count('.meta') == 1 and file_list.count('.index') == 1:
        return [os.path.splitext(i)[0] for i in os.listdir(model_path) if i.endswith(".meta")][0]

    return None


def is_keras_savedmodel_format(model_path):
    """check the model_path format is keras saved model or not.

    Args:
        model_path (string): the model folder path

    Returns:
        bool: return the keras model if the model is keras model else None.
    """
    if is_saved_model_format(model_path):
        model = tf.keras.models.load_model(model_path)
        if isinstance(model, tf.keras.Model):
            return model
    return None

def parse_ckpt_model(ckpt_prefix, outputs):
    """Parse the ckpt model

    Args:
        ckpt_prefix (string): the ckpt prefix for parsing
    """
    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.import_meta_graph(ckpt_prefix + '.meta',
                                                     clear_devices=True)
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, ckpt_prefix)
        graph_def = sess.graph.as_graph_def()
        _parse_ckpt_bn_input(graph_def)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=graph_def,
            output_node_names=outputs)

        return output_graph_def


def _parse_ckpt_bn_input(graph_def):
    """parse ckpt batch norm inputs to match correct moving mean and variance
    Args:
        graph_def (graph_def): original graph_def
    Returns:
        graph_def: well linked graph_def
    """
    for node in graph_def.node:
        if node.op == 'FusedBatchNorm':
            moving_mean_op_name = node.input[3]
            moving_var_op_name = node.input[4]
            moving_mean_op = _get_nodes_from_name(moving_mean_op_name, graph_def)[0]
            moving_var_op = _get_nodes_from_name(moving_var_op_name, graph_def)[0]

            if moving_mean_op.op == 'Const':
                name_part = moving_mean_op_name.rsplit('/', 1)[0]
                real_moving_mean_op_name = name_part + '/moving_mean'
                if len(_get_nodes_from_name(real_moving_mean_op_name, graph_def)) > 0:
                    # replace the real moving mean op name
                    node.input[3] = real_moving_mean_op_name

            if moving_var_op.op == 'Const':
                name_part = moving_var_op_name.rsplit('/', 1)[0]
                real_moving_var_op_name = name_part + '/moving_variance'
                if len(_get_nodes_from_name(real_moving_var_op_name, graph_def)) > 0:
                    # replace the real moving mean op name
                    node.input[4] = real_moving_var_op_name

    return graph_def


def _get_nodes_from_name(node_name, graph_def):
    """get nodes from graph_def using node name
    Args:
        graph_def (graph_def): graph_def
        node_name (str): node name

    Returns:
        node (NodeDef): graph node
    """
    return [node for node in graph_def.node if node.name == node_name]


def is_saved_model_format(model_path):
    """check the model_path format is saved_model or not

    Args:
        model_path (string): the model folder path

    Returns:
        bool: return True if the model_path contains saved_model format else False.
    """
    file_list = [os.path.splitext(i)[-1] for i in os.listdir(model_path)]
    return bool(file_list.count('.pb') == 1 and ('variables') in os.listdir(model_path))


def parse_kerasmodel_model(model):
    """Convert Keras Model to graphdef

    Args:
        model (keras.Model): Keras model object

    Returns:
        graph_def: the parsed graph_def object.
        input_names: input node names
        output_names: output node name
    """

    kwargs = dict(zip(model.input_names, model.inputs))
    full_model = tf.function(lambda **kwargs: model(kwargs.values()))
    concrete_function = full_model.get_concrete_function(**kwargs)
    frozen_model = convert_variables_to_constants_v2(concrete_function)
    graph_def = frozen_model.graph.as_graph_def()
    input_names = [node.name for node in graph_def.node if node.op == 'Placeholder']
    output_names = [output.split(':')[0] for output in model.output_names]
    # replace the output name with squential
    for output_name in output_names:
        for node in graph_def.node[::-1]:
            if node.op == 'Identity' and output_name in node.input[0]:
                node.name = output_name
                break

    return graph_def, input_names, output_names


def parse_savedmodel_model(model_path):
    """Convert SavedModel to graphdef

    Args:
        model_path (string): the model folder path

    Returns:
        graphdef: the parsed graphdef object.
        input_names: input node names
        output_names: output node name
    """

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        meta_graph = tf.compat.v1.saved_model.loader.load(
            sess, ["serve"], model_path)

        model_graph_signature = list(
            meta_graph.signature_def.items())[0][1]

        input_names = [input_item[1].name
                       for input_item in model_graph_signature.inputs.items()]

        output_names = [output_item[1].name
                        for output_item in model_graph_signature.outputs.items()]

        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=[output_item[0]
                               for output_item in model_graph_signature.outputs.items()])

        return output_graph_def, input_names, output_names


def convert_pb_to_savedmodel(graph_def, input_tensor_names, output_tensor_names, output_dir):
    """Convert the graphdef to SavedModel

    Args:
        graph_def (graphdef): parsed graphdef object.
        input_tensor_names (list): input tensor names list.
        output_tensor_names (list): output tensor names list.
        output_dir (string): Converted SavedModel store path.
    """
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(output_dir)

    sigs = {}
    with tf.compat.v1.Session() as sess:
        tf.import_graph_def(graph_def, name="")
        g = tf.compat.v1.get_default_graph()

        input_tensors = {}
        for input_tensor_name in output_tensor_names:
            input_tensors[input_tensor_name.split(':')[0]] = g.get_tensor_by_name(
                "{}".format(input_tensor_name))

        output_tensors = {}
        for output_tensor_name in input_tensor_names:
            output_tensors[output_tensor_name.split(':')[0]] = g.get_tensor_by_name(
                "{}".format(output_tensor_name))

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            output_tensors, input_tensors)

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)

    builder.save()


def get_slim_graph(model, model_func, arg_scope, images, outputs=None, **kwargs):
    assert tf.version.VERSION < '2.0.0', 'slim model only used in tensorflow 1.x'
    import tf_slim as slim
    with tf.compat.v1.Session() as sess:
        with slim.arg_scope(arg_scope) as scope:  # pylint: disable=not-context-manager
            model_func(images, is_training=False, **kwargs)
        graph_def = sess.graph.as_graph_def()

        if outputs is None:
            outputs = graph_def.node[-1].name

        from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos
        graph_def = freeze_graph_with_def_protos(
            input_graph_def=graph_def,
            input_saver_def=None,
            input_checkpoint=model,
            output_node_names=outputs,
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='',
            clear_devices=True,
            initializer_nodes='')

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph

def get_estimator_graph(estimator, input_fn):
    with tf.Graph().as_default() as g:
      features, input_hooks = estimator._get_features_from_input_fn(
          input_fn, tf.estimator.ModeKeys.PREDICT)
      estimator_spec = estimator._call_model_fn(features, None,
          tf.estimator.ModeKeys.PREDICT, estimator.config)

      outputs = [tensor.name for tensor in estimator_spec.predictions.values()] if\
          isinstance(estimator_spec.predictions, dict) else \
              [estimator_spec.predictions.name]
      logger.info('estimator output tensor names is {}'.format(outputs))
      with tf.compat.v1.Session(graph=g) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # Freezing a graph requires output_node_names, which can be found in
        # estimator_spec.predictions that contains prediction tensors as a
        # dictionary
        # When a model uses Iterator, we need to have 'MakeIterator' (default
        # name used by TF) in the output_node_names as well.
        output_nodes = list(set([output.split(':')[0] for output in outputs]))
        if 'MakeIterator' in [node.op for node in g.as_graph_def().node]:
            output_nodes.append('MakeIterator')
                
        graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess,
          g.as_graph_def(), output_nodes)

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name='') 
    return graph
    
def get_tensor_by_name(graph, name, try_cnt=3):
    """Get the tensor by name considering the 'import' scope when model
       may be imported more then once, handle naming format like both name:0 and name
    
    Args:
        graph (tf.compat.v1.GraphDef): the model to get name from
        name (string): tensor of tensor_name:0 or tensor_name without suffixes
        try_cnt: the times to add 'import/' to find  tensor
    
    Returns:
        tensor: tensor got by name.
    """
    if name.find(':') == -1:
        name = name + ':0'
    for _ in range(try_cnt):
        try:
            return graph.get_tensor_by_name(name)
        except BaseException:
            name = 'import/' + name
    raise ValueError('can not find tensor by name')

def iterator_sess_run(sess, iter_op, feed_dict, output_tensor, iteration=-1):
    """Run the graph that have iterator integrated in the graph
    
    Args:
        sess (tf.compat.v1.Session): the model sess to run the graph
        iter_op (Operator): the MakeIterator op
        feed_dict(dict): the feeds to initialize a new iterator
        output_tensor(list): the output tensors 
        iteration(int): iterations to run, when -1 set, run to end of iterator
    
    Returns:
        preds: the results of the predictions
    """
    sess.run(iter_op, feed_dict)
    preds = []
    idx = 0 
    while idx < iteration or iteration == -1:
        try:
            prediction = sess.run(output_tensor)
            preds.append(prediction)
            idx += 1
        except tf.errors.OutOfRangeError:
            break
        except:
            logger.warning('not run out of the preds...')
            break
    preds = list(zip(*preds))
    return preds

def validate_graph_input(graph_def, input_node_names):
    """Check input node existence, have 3 conditions:
       1. input node names empty, return False
       2. input node names in the graph node list, return True
       3. input node names not in the graph node list, raise Error

    Args:
        graph_def (GraphDef):GraphDef
        input_node_namess ([String]): input node names list.

    Returns:
        status (bool): the validation status
    """
    if len(input_node_names) == 0:
        return False
    all_node_name = [node.name for node in graph_def.node]
    for user_input_name in input_node_names:
        assert user_input_name in all_node_name, \
            "Input node name {} doesn't exist in the model, please check the yaml.".\
                format(user_input_name)
    return True

def validate_graph_output(graph_def, output_node_names):
    """Check output node existence, have 3 conditions:
       1. output node names empty, return False
       2. output node names in the graph node list, return True
       3. output node names not in the graph node list, raise Error

    Args:
        graph_def (GraphDef):GraphDef
        output_node_namess ([String]): output node names list.

    Returns:
        status (bool): the validation status
    """
    if len(output_node_names) == 0:
        return False
    all_node_name = [node.name for node in graph_def.node]
    for user_output_name in output_node_names:
        assert user_output_name in all_node_name,\
             "Output node name {} doesn't exist in the model, please check the yaml.".\
                 format(user_output_name)
    return True

def get_input_node_names(graph_def):
    from .graph_rewriter.graph_util import GraphAnalyzer
    g = GraphAnalyzer()
    g.graph = graph_def
    g.parse_graph()
    return g.get_graph_input_output()[0]

def get_output_node_names(graph_def):
    from .graph_rewriter.graph_util import GraphAnalyzer
    g = GraphAnalyzer()
    g.graph = graph_def
    g.parse_graph()
    return g.get_graph_input_output()[1]

def get_graph_def(model, outputs=[], auto_input_output=False):
    """Get the input model graphdef

    Args:
        model ([Graph, GraphDef or Path String]): support Graph, GraphDef, keras.Model,
                                                  frozen pb or ckpt/savedmodel path.
        outputs ([String]): output node names list.

    Returns:
        graph_def (graphdef): parsed graphdef object.
    """
    graph_def = None
    if isinstance(model, Graph):
        graph_def = model.as_graph_def()
    elif isinstance(model, tf.compat.v1.GraphDef):
        graph_def = model
    elif isinstance(model, tf.keras.Model):
        graph_def, _, _ = parse_kerasmodel_model(model)
    elif isinstance(model, str):
        graph_def = tf.compat.v1.GraphDef()
        model = os.path.expanduser(model)
        if model.endswith('.pb') and os.path.isfile(model):
            with open(model, 'rb') as f:
                graph_def.ParseFromString(f.read())
        elif model.endswith('.ckpt') and os.path.isfile(model):
            raise ValueError('use get_slim_graph to get the graph first')
        elif model.endswith('.h5') and os.path.isfile(model):
            # (TODO) support h5 saved model, notice there is also h5 weights
            raise ValueError('saved model h5 format not supported yet, soon')
        elif os.path.isdir(model):
            # tf2.x checkpoint only save weight and do not contain any
            # description of the computation, so we drop tf2.x checkpoint support
            ckpt_prefix = is_ckpt_format(model)
            assert outputs
            if ckpt_prefix is not None:
                graph_def = parse_ckpt_model(
                    os.path.join(model, ckpt_prefix), outputs)
            # (TODO) support tf2.x saved model
            # tf1.x saved model is out of date and few examples, drop
            if is_saved_model_format(model):
                keras_model = is_keras_savedmodel_format(model)
                if keras_model is not None:
                    graph_def, _, _ = parse_kerasmodel_model(keras_model)
                else:
                    raise ValueError('tf saved model format not supported yet, soon')
            if graph_def is None:
                raise ValueError('only support tf1.x checkpoint or tf2.x keras saved model')
        else:
            raise ValueError('only support frozen pb file or model path')
    else:
        raise ValueError(
            'only support Graph, GraghDef, keras.Model, tf1.x checkpoint, keras saved model')

    return graph_def
