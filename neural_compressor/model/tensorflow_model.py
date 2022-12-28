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

"""Class for Tensorflow model."""

import copy
import os
import shutil
import importlib
from abc import abstractmethod
import tempfile
import sys
from neural_compressor.utils.utility import LazyImport, compute_sparsity
from neural_compressor.utils.utility import version1_lt_version2, version1_gt_version2, version1_gte_version2
from neural_compressor.utils import logger
from neural_compressor.conf import config as cfg
from neural_compressor.model.base_model import BaseModel

tf = LazyImport('tensorflow')
np = LazyImport('numpy')

tensor_to_node = lambda s: list(set([x.split(':')[0] for x in s]))

def get_model_type(model):
    """Get Tensorflow mode type.

    Args:
        model (string or model object): model path or model object.

    Returns:
        string: model type
    """
    from neural_compressor.adaptor.tf_utils.util import is_saved_model_format, is_ckpt_format
    if isinstance(model, str):
        model = os.path.abspath(os.path.expanduser(model))
        if (model.endswith('.h5') and os.path.isfile(model)) or \
          is_saved_model_format(os.path.dirname(model)) or \
          (os.path.isdir(model) and is_saved_model_format(model)):
            if version1_lt_version2(tf.version.VERSION, '2.10.0'):
                logger.warn("keras model running on tensorflow 2.10.0 and"
                            " lower not support intel ITEX.")
            try:
                model = tf.keras.models.load_model(model)
                if isinstance(model, tf.keras.Model) and hasattr(model, 'to_json'):
                    return 'keras'
                return 'saved_model'
            except:
                pass
    if isinstance(model, tf.keras.Model) and hasattr(model, 'to_json'):
        return 'keras'
    if isinstance(model, tf.Graph):
        return 'graph'
    elif isinstance(model, tf.compat.v1.GraphDef):
        return 'graph_def'
    elif isinstance(model, tf.compat.v1.estimator.Estimator):
        return 'estimator'
    elif isinstance(model, str):
        model = os.path.abspath(os.path.expanduser(model))
        if (model.endswith('.pb') and os.path.isfile(model)):
            if is_saved_model_format(os.path.dirname(model)):
                return 'saved_model'
            else:
                return 'frozen_pb'
        elif model.endswith('.ckpt') and os.path.isfile(model):
            return 'slim'
        elif os.path.isdir(model):
            if is_ckpt_format(model):
                return 'checkpoint'
            elif is_saved_model_format(model):
                return 'saved_model'
        elif os.path.isfile(model + '.pb'):
            return 'frozen_pb'

    raise ValueError('model {} has not recognized model type....'.format(model))



def validate_graph_node(graph_def, node_names):
    """Validate nodes exist in the graph_def.

    Args:
        graph_def (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object.
        node_names (list of string): node names to be validated.
    """
    if len(node_names) == 0:
        return False
    all_node_name = [node.name for node in graph_def.node]
    for user_name in node_names:
        if user_name not in all_node_name:
            logger.warn(
                str("Node name {} specified in yaml doesn't exist in the model.").
                format(user_name))
            return False
    return True

def validate_and_inference_input_output(graph_def, \
    input_tensor_names, output_tensor_names):
    """Validate and inference the input and output tensor names of graph_def.

    Args:
        graph_def (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object.
        input_tensor_names (list of string): input_tensor_names of graph_def.
        output_tensor_names (list of string): output_tensor_names of graph_def.

    Returns:
        input_tensor_names (list of string): validated input_tensor_names.
        output_tensor_names (list of string): validated output_tensor_names.
    """
    from neural_compressor.adaptor.tf_utils.util import get_input_output_node_names
    temp_output_tensor_names = []
    if validate_graph_node(graph_def, tensor_to_node(input_tensor_names)):
        input_tensor_names = input_tensor_names
    else:
        input_tensor_names, temp_output_tensor_names = get_input_output_node_names(graph_def)

    if validate_graph_node(graph_def, tensor_to_node(output_tensor_names)):
        output_tensor_names = output_tensor_names
    elif temp_output_tensor_names:
        output_tensor_names = temp_output_tensor_names
    else:
        _, output_tensor_names = get_input_output_node_names(graph_def)

    return input_tensor_names, output_tensor_names

def graph_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Helper to build session with tf.compat.v1.Graph.

    Args:
        model (tf.compat.v1.Graph): tf.compat.v1.Graph object.
        input_tensor_names (list of string): input_tensor_names of model.
        output_tensor_names (list of string): output_tensor_names of model.

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object.
        input_tensor_names (list of string): validated input_tensor_names.
        output_tensor_names (list of string): validated output_tensor_names.
    """
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    sess = tf.compat.v1.Session(graph=model, config=config)

    input_tensor_names, output_tensor_names = validate_and_inference_input_output(\
        model.as_graph_def(), input_tensor_names, output_tensor_names)

    return sess, input_tensor_names, output_tensor_names

def graph_def_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with tf.compat.v1.GraphDef.

    Args:
        model (tf.compat.v1.GraphDef): tf.compat.v1.GraphDef object.
        input_tensor_names (list of string): input_tensor_names of model.
        output_tensor_names (list of string): output_tensor_names of model.

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object
        input_tensor_names (list of string): validated input_tensor_names
        output_tensor_names (list of string): validated output_tensor_names
    """
    graph = tf.Graph()
    try:
        with graph.as_default():
            tf.import_graph_def(model, name='')
    except:
        input_tensor_names, output_tensor_names = validate_and_inference_input_output(\
            model, input_tensor_names, output_tensor_names)
        from neural_compressor.adaptor.tf_utils.util import fix_ref_type_of_graph_def
        from neural_compressor.adaptor.tf_utils.util import strip_unused_nodes
        model = fix_ref_type_of_graph_def(model)
        input_node_names = tensor_to_node(input_tensor_names)
        output_node_names = tensor_to_node(output_tensor_names)
        model = strip_unused_nodes(model, input_node_names, output_node_names)
        with graph.as_default():
            tf.import_graph_def(model, name='')

    return graph_session(graph, input_tensor_names, output_tensor_names, **kwargs)

def frozen_pb_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with frozen pb.

    Args:
        model (string): model path.
        input_tensor_names (list of string): input_tensor_names of model.
        output_tensor_names (list of string): output_tensor_names of model.

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object.
        input_tensor_names (list of string): validated input_tensor_names.
        output_tensor_names (list of string): validated output_tensor_names.
    """
    graph_def = tf.compat.v1.GraphDef()
    model = model if model.endswith('.pb') else model + '.pb'
    with open(model, 'rb') as f:
        graph_def.ParseFromString(f.read())
    return graph_def_session(graph_def, input_tensor_names, \
        output_tensor_names, **kwargs)

def _contains_function_with_implements_attr(saved_model_proto):
    meta_graph = saved_model_proto.meta_graphs[0]
    for function in meta_graph.graph_def.library.function:
        if function.attr.get("_implements", None) or function.attr.get(
            "api_implements", None):
            return True
    return False

def load_saved_model(model, saved_model_tags, input_tensor_names, output_tensor_names):
    """Load graph_def from saved model with the default serving signature key.

    Args:
        saved_model_dir: Directory of the SavedModel.
        saved_model_tags: Set of tags identifying the MetaGraphDef within the
            SavedModel to analyze.

    Returns:
        graph_def: The loaded GraphDef.
        input_tensors: List of input tensors.
        output_tensors: List of output tensors.
    """
    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    if not os.listdir(os.path.join(model,'variables')):
        sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)
        loader = tf.compat.v1.saved_model.loader.load(sess, ["serve"], model)
        if len(input_tensor_names) == 0:
            input_tensor_names = [i.name for _, i in \
                loader.signature_def['serving_default'].inputs.items()]
        else:
            assert validate_graph_node(\
                sess.graph.as_graph_def(), tensor_to_node(input_tensor_names)), \
                    'tensor names {} not in the graph'.format(input_tensor_names)

        if len(output_tensor_names) == 0:
            output_tensor_names = [i.name for _, i in \
                loader.signature_def['serving_default'].outputs.items()]
        else:
            assert validate_graph_node(\
                sess.graph.as_graph_def(), tensor_to_node(output_tensor_names)), \
                    'tensor names {} not in the graph'.format(output_tensor_names)

        return sess.graph.as_graph_def(), input_tensor_names, output_tensor_names
    else:
        from tensorflow.python.eager import context
        from tensorflow.python.saved_model import load
        from tensorflow.python.saved_model import tag_constants
        from tensorflow.python.saved_model import signature_constants
        from tensorflow.python.framework.convert_to_constants import \
        convert_variables_to_constants_v2
        from tensorflow.python.training import saver
        from tensorflow.core.protobuf import config_pb2
        from tensorflow.python.grappler import tf_optimizer
        from tensorflow.core.protobuf import meta_graph_pb2
        _saved_model = load.load(model, [tag_constants.SERVING])
        func = _saved_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        frozen_func = convert_variables_to_constants_v2(func)
        grappler_meta_graph_def = saver.export_meta_graph(
        graph_def=frozen_func.graph.as_graph_def(), graph=frozen_func.graph)
        if len(input_tensor_names) == 0:
            input_tensor_names = [i.name.split(':')[0] for i in frozen_func.inputs]
        if len(output_tensor_names) == 0:
            output_tensor_names = [i.name.split(':')[0] for i in frozen_func.outputs]
        # Add a collection 'train_op' so that Grappler knows the outputs.
        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in frozen_func.inputs + frozen_func.outputs:
            fetch_collection.node_list.value.append(array.name)
            grappler_meta_graph_def.collection_def["train_op"].CopyFrom(
            fetch_collection)
        from tensorflow.python.eager import context
        grappler_session_config = config_pb2.ConfigProto()
        rewrite_options = grappler_session_config.graph_options.rewrite_options
        rewrite_options.min_graph_nodes = -1
        opt = tf_optimizer.OptimizeGraph(grappler_session_config,
                                         grappler_meta_graph_def, graph_id=b"tf_graph")
        return opt, input_tensor_names, output_tensor_names

def _get_graph_from_saved_model_v2(saved_model_dir,
        input_tensor_names, output_tensor_names):
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.saved_model import signature_constants
    saved_model_exported_names = [
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    saved_model_tags = set([tag_constants.SERVING])
    return load_saved_model(saved_model_dir, saved_model_tags,
                            input_tensor_names, output_tensor_names)

def _get_graph_from_original_keras_v2(model, output_dir):
    from tensorflow.python.eager import def_function
    from tensorflow.lite.python.util import trace_model_call
    from tensorflow.lite.python.util import model_input_signature
    from tensorflow.python.framework import convert_to_constants
    from tensorflow.python.framework import dtypes
    from tensorflow.lite.python.util import run_graph_optimizations
    from tensorflow.lite.python.convert import OpsSet
    from tensorflow.lite.python.util import get_grappler_config
    input_signature = None
    # If the model's call is not a `tf.function`, then we need to first get its
    # input signature from `model_input_signature` method.
    if not isinstance(model.call, def_function.Function):
        input_signature = model_input_signature(model, keep_original_batch_size=False)

    func = trace_model_call(model, input_signature)
    concrete_func = func.get_concrete_function()
    funcs = [concrete_func]

    frozen_func, graph_def = (
        convert_to_constants.convert_variables_to_constants_v2_as_graph(
            funcs[0], lower_control_flow=False))

    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != dtypes.resource
    ]
    output_tensors = frozen_func.outputs
    # Grappler will also try to lower while loop into switch merge
    # representation which is undesired for Ophints, so we simply remove
    # those attributes to prevent Grappler from doing so.
    graph = convert_to_constants.disable_lower_using_switch_merge(graph_def)
    # Run function inlining optimization to ensure any models generated
    # through the from_frozen_graph path have been inlined.
    # grappler_config = get_grappler_config(['function'])
    # graph_def = run_graph_optimizations(
    #             graph,
    #             input_tensors,
    #             output_tensors,
    #             config=grappler_config)
    input_names = [tensor.name.split(':')[0] for tensor in input_tensors]
    output_names = [tensor.name.split(':')[0] for tensor in output_tensors]
    return graph_def, input_names, output_names

def _check_keras_format(model, saved_model_dir):
    from tensorflow.python import saved_model
    from tensorflow.python.saved_model.load import load
    from tensorflow.python.saved_model import save_options
    from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info
    version = 'saved_model_v2'
    try:
        saved_model.save(
            model,
            saved_model_dir,
            options=save_options.SaveOptions(save_debug_info=True))
    except:
        return 'trackable_object'
    saved_model_proto, _ = parse_saved_model_with_debug_info(saved_model_dir)
    saved_model_version = saved_model_proto.saved_model_schema_version
    if saved_model_version == 0:
        return 'saved_model_v1'
    if saved_model_version not in [1, 2]:
        raise ValueError("SavedModel file format({0}) is not supported".format(
            saved_model_version))
    return version

def _get_graph_from_saved_model_v1(model):
    from tensorflow.python.framework import ops
    from tensorflow.python.saved_model import constants
    from tensorflow.python.client import session
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.lite.python.convert_saved_model import get_meta_graph_def
    from tensorflow.lite.python.convert_saved_model import get_signature_def
    from tensorflow.lite.python.convert_saved_model import get_inputs_outputs
    saved_model_tags = set([tag_constants.SERVING])
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    meta_graph = get_meta_graph_def(model, saved_model_tags)
    signature_def = get_signature_def(meta_graph, signature_key)
    inputs, outputs = get_inputs_outputs(signature_def)
    # Check SavedModel for assets directory.
    collection_def = meta_graph.collection_def
    if constants.ASSETS_KEY in collection_def:
        raise ValueError("SavedModels with assets/ directory are not supported.")

    from tensorflow.python.saved_model import loader
    from tensorflow.python.framework import graph_util as tf_graph_util
    graph = ops.Graph()
    import tensorflow as tf
    with session.Session(graph=graph) as sess:
        loader.load(sess, meta_graph.meta_info_def.tags, model)
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        output_nodes = list(set([output.split(':')[0] for output in outputs]))
        node_ops = [node.op for node in graph.as_graph_def().node]
        if 'MakeIterator' in node_ops:
            output_nodes.append('MakeIterator')
        table_ops = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TABLE_INITIALIZERS)
        # For table initialization
        for table_op in table_ops:
            output_nodes.append(table_op.name)
        if len(table_ops) > 0:
            output_nodes.append('init_all_tables')
        graph_def = tf_graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), output_nodes)
    return graph_def, inputs, outputs

def keras_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with keras model.

    Args:
        model (string or tf.keras.Model): model path or tf.keras.Model object.
        input_tensor_names (list of string): input_tensor_names of model.
        output_tensor_names (list of string): output_tensor_names of model.

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object.
        input_tensor_names (list of string): validated input_tensor_names.
        output_tensor_names (list of string): validated output_tensor_names.
    """
    temp_dir = tempfile.mkdtemp()
    if tf.version.VERSION > '2.1.0':
        if not isinstance(model, tf.keras.Model):
            model = tf.keras.models.load_model(model)
        keras_format = _check_keras_format(model, temp_dir)
        if keras_format == 'saved_model_v2':
            try:
                graph_def, input_names, output_names = _get_graph_from_saved_model_v2(
                    temp_dir, input_tensor_names, output_tensor_names)
                if '_FusedBatchNormEx' in [node.op for node in graph_def.node]:
                    keras_format = 'trackable_object'
            except:
                keras_format = 'trackable_object'
        if keras_format == 'trackable_object':
            try:
                graph_def, input_names, output_names = _get_graph_from_original_keras_v2(
                                                       model, temp_dir)
            except:
                keras_format = 'saved_model_v1'
        if keras_format == 'saved_model_v1':
            try:
                tf.keras.backend.set_learning_phase(0)
                graph_def, input_names, output_names = _get_graph_from_saved_model_v1(model)
            except:
                raise ValueError('Not supported keras model type...')

    # tensorflow 1.x use v1 convert method
    else:
        tf.keras.backend.set_learning_phase(0)
        graph_def, input_names, output_names = _get_graph_from_saved_model_v1(model)
    shutil.rmtree(temp_dir, True)
    return graph_def_session(graph_def, input_names, output_names, **kwargs)


def slim_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with slim model.

    Args:
        model (string): model path.
        input_tensor_names (list of string): input_tensor_names of model.
        output_tensor_names (list of string): output_tensor_names of model.

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object.
        input_tensor_names (list of string): validated input_tensor_names.
        output_tensor_names (list of string): validated output_tensor_names.
    """
    assert version1_lt_version2(tf.version.VERSION, '2.0.0'), 'slim model only used in tensorflow 1.x'
    from .nets_factory import TFSlimNetsFactory
    factory = TFSlimNetsFactory()
    assert 'name' in kwargs, 'model name should be set in slim checkpoint....'
    assert kwargs['name'] in factory.default_slim_models, \
        'only support topology {}'.format(factory.default_slim_models)
    net = copy.deepcopy(factory.networks_map[kwargs['name']])
    model_func = net.pop('model')
    arg_scope = net.pop('arg_scope')()
    inputs_shape = net.pop('input_shape')
    kwargs = net
    import tf_slim as slim
    with tf.Graph().as_default():
        images = tf.compat.v1.placeholder(name='input', dtype=tf.float32, \
            shape=inputs_shape)
        with tf.compat.v1.Session() as sess:
            with slim.arg_scope(arg_scope) as scope:  # pylint: disable=not-context-manager
                model_func(images, is_training=False, **kwargs)
            graph_def = sess.graph.as_graph_def()
            output_tensor_names = output_tensor_names if len(output_tensor_names) > 0 \
                else [graph_def.node[-1].name]

            from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos
            graph_def = freeze_graph_with_def_protos(
                input_graph_def=graph_def,
                input_saver_def=None,
                input_checkpoint=model,
                output_node_names=','.join(output_tensor_names),
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0',
                output_graph='',
                clear_devices=True,
                initializer_nodes='')

    return graph_def_session(graph_def, ['input'], output_tensor_names)

def checkpoint_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with ckpt model.

    Args:
        model (string): model path.
        input_tensor_names (list of string): input_tensor_names of model.
        output_tensor_names (list of string): validated output_tensor_names of model.

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object.
        input_tensor_names (list of string): validated input_tensor_names.
        output_tensor_names (list of string): validated output_tensor_names.
    """
    assert output_tensor_names is not None and len(output_tensor_names) > 0, \
        'outputs should not be None of checkpoint....'

    ckpt_prefix = [os.path.splitext(i)[0] for i in os.listdir(model) \
        if i.endswith(".meta")][0]

    config = tf.compat.v1.ConfigProto()
    config.use_per_session_threads = 1
    config.inter_op_parallelism_threads = 1
    graph = tf.Graph()
    sess = tf.compat.v1.Session(graph=graph, config=config)
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(\
            os.path.join(model, ckpt_prefix + '.meta'), clear_devices=True)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, os.path.join(model, ckpt_prefix))

    from neural_compressor.adaptor.tf_utils.util import get_input_output_node_names
    if validate_graph_node(sess.graph.as_graph_def(), tensor_to_node(input_tensor_names)):
        input_tensor_names = input_tensor_names
    else:
        input_tensor_names, _ = get_input_output_node_names(sess.graph.as_graph_def())
    return sess, input_tensor_names, output_tensor_names

def estimator_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with estimator model.

    Args:
        model (tf.estimator.Estimator): tf.estimator.Estimator object.
        input_tensor_names (list of string): input_tensor_names of model.
        output_tensor_names (list of string): output_tensor_names of model.
        kwargs (dict): other required parameters, like input_fn.

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object.
        input_tensor_names (list of string): validated input_tensor_names.
        output_tensor_names (list of string): validated output_tensor_names.
    """
    assert 'input_fn' in kwargs, 'input func should be supplied for estimator session....'
    with tf.Graph().as_default() as g:
        features, input_hooks = model._get_features_from_input_fn(
            kwargs['input_fn'], tf.estimator.ModeKeys.PREDICT)
        estimator_spec = model._call_model_fn(features, None,
            tf.estimator.ModeKeys.PREDICT, model.config)

        if len(output_tensor_names) == 0:
            outputs = [tensor.name for tensor in estimator_spec.predictions.values()] if\
                isinstance(estimator_spec.predictions, dict) else \
                    [estimator_spec.predictions.name]
        else:
            outputs = output_tensor_names

        logger.info("Estimator output tensor names are {}.".format(outputs))
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

        return graph_def_session(graph_def, input_tensor_names, outputs)

def saved_model_session(model, input_tensor_names, output_tensor_names, **kwargs):
    """Build session with saved model.

    Args:
        model (string): model path.
        input_tensor_names (list of string): input_tensor_names of model.
        output_tensor_names (list of string): output_tensor_names of model.

    Returns:
        sess (tf.compat.v1.Session): tf.compat.v1.Session object.
        input_tensor_names (list of string): validated input_tensor_names.
        output_tensor_names (list of string): validated output_tensor_names.
    """
    try:
        graph_def, input_names, output_names = _get_graph_from_saved_model_v2(
            model, input_tensor_names, output_tensor_names)
    except:
        graph_def, input_names, output_names = _get_graph_from_saved_model_v1(model)
    assert graph_def is not None, 'Can not parse the saved model...'
    return graph_def_session(graph_def, input_names, output_names, **kwargs)

# it's necessary that a session with input output tensors to run the model
SESSIONS = {'frozen_pb': frozen_pb_session,
            'graph_def': graph_def_session,
            'graph': graph_session,
            'saved_model': saved_model_session,
            'keras': keras_session,
            'checkpoint': checkpoint_session,
            'estimator': estimator_session,
            'slim': slim_session,}


class TensorflowBaseModel(BaseModel):
    """Build Tensorflow Base Model."""

    def __init__(self, model, **kwargs):
        """Initialize a Tensorflow model.

        Args:
            model (string or tensorflow model object): model path or model object.
        """
        self._model = model
        self._name = ''
        self._weights = None
        self.kwargs = kwargs
        self._graph_info = {}
        self._input_tensor_names = []
        self._output_tensor_names = []
        self._model_type = ''
        self._sess = None
        self._iter_op = None
        self._workspace_path = ''
        self._q_config = None

    def framework(self):
        """Return framework."""
        return 'tensorflow'

    @property
    def name(self):
        """Renturn name."""
        return self._name

    @name.setter
    def name(self, name):
        """Set name."""
        self.kwargs.update({'name': name})
        self._name = name

    @property
    def weights(self):
        """Return weights."""
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        """Set weights."""
        self._weights = new_weights

    @property
    def q_config(self):
        """Return q_config."""
        return self._q_config

    @q_config.setter
    def q_config(self, q_config):
        """Set q_config."""
        self._q_config = q_config

    @property
    def workspace_path(self):
        """Return workspace path."""
        return self._workspace_path

    @workspace_path.setter
    def workspace_path(self, path):
        """Set workspace path."""
        self._workspace_path = path

    @property
    def model_type(self):
        """Return model type."""
        return self._model_type

    @model_type.setter
    def model_type(self, model_type):
        """Set model type."""
        assert model_type in SESSIONS, 'model type not supported....'
        self._model_type = model_type

    @property
    def model(self):
        """Return model itself."""
        return self.graph

    @property
    def graph_def(self):
        """Return graph defination."""
        return self.graph.as_graph_def()

    @property
    def graph_info(self):
        """Return graph info."""
        self._graph_info = {}
        for node in self.graph_def.node:
            self._graph_info[node.name] = node.op
        return self._graph_info

    @property
    def sess(self):
        """Return Session object."""
        if self._sess is None:
            self._load_sess(self._model, **self.kwargs)
        return self._sess

    @property
    def graph(self):
        """Return model graph."""
        return self.sess.graph

    @graph_def.setter
    def graph_def(self, graph_def):
        """Set graph defination."""
        if self._sess is not None:
            self._sess.close()
        output_sess =  SESSIONS['graph_def'](graph_def,\
                                             self._input_tensor_names, \
                                             self._output_tensor_names)

        self._sess = output_sess[0]
        self._input_tensor_names = output_sess[1]
        self._output_tensor_names = output_sess[2]
        self.model_type = 'graph_def'

    def _load_sess(self, model, **kwargs):
        if self.name:
            kwargs.update({'name': self.name})
        # assert self.model_type, 'model type not set....'
        output_sess = SESSIONS[self.model_type](model,
                                                self._input_tensor_names, \
                                                self._output_tensor_names,
                                                **kwargs)
        self._sess = output_sess[0]
        self._input_tensor_names = output_sess[1]
        self._output_tensor_names = output_sess[2]

        tf.compat.v1.get_variable_scope().reuse_variables()
        return self._sess

    @property
    def iter_op(self):
        """Return model iter op list."""
        self._iter_op = []
        if self._sess is None:
            self._load_sess(self._model, **self.kwargs)
        op_list = [node.op for node in self._sess.graph.as_graph_def().node]
        if 'MakeIterator' in op_list:
            self._iter_op.append(self._sess.graph.get_operation_by_name('MakeIterator'))
        return self._iter_op

    @property
    def input_tensor_names(self):
        """Return input tensor names."""
        if self._sess is None:
            self._load_sess(self._model, **self.kwargs)
        return copy.deepcopy(self._input_tensor_names)

    @input_tensor_names.setter
    def input_tensor_names(self, tensor_names):
        """Set input tensor names."""
        if len(tensor_names) == 0:
            logger.warn("Input tensor names is empty.")
            return
        if self._sess is not None:
            assert validate_graph_node(\
                self.graph_def, tensor_to_node(tensor_names)), \
                'tensor names {} not in graph'.format(tensor_names)
        self._input_tensor_names = tensor_names

    @property
    def output_tensor_names(self):
        """Return output tensor names."""
        if len(self._output_tensor_names) == 0:
            self._load_sess(self._model, **self.kwargs)
        return copy.deepcopy(self._output_tensor_names)

    @output_tensor_names.setter
    def output_tensor_names(self, tensor_names):
        """Set output tensor names."""
        if len(tensor_names) == 0:
            logger.warn("Output tensor names should not be empty.")
            return
        if self._sess is not None:
            assert validate_graph_node(\
                self.graph_def, tensor_to_node(tensor_names)), \
                'tensor names {} not in graph'.format(tensor_names)
        self._output_tensor_names = tensor_names

    # input/output node names and input/output tensor
    # come from input/output tensor names, so do not support assign these values
    @property
    def input_node_names(self):
        """Return input node names."""
        return copy.deepcopy(tensor_to_node(self.input_tensor_names))

    @property
    def output_node_names(self):
        """Return output node names."""
        output_node_names = tensor_to_node(self.output_tensor_names)
        iter_op_list = self.iter_op
        if iter_op_list != []:
            output_node_names += [iter_op.name for iter_op in iter_op_list]
        return copy.deepcopy(output_node_names)

    @property
    def input_tensor(self):
        """Return input tensor."""
        from neural_compressor.adaptor.tf_utils.util import get_tensor_by_name
        return [get_tensor_by_name(\
            self.graph, x) for x in self.input_tensor_names]

    @property
    def output_tensor(self):
        """Return output tensor."""
        from neural_compressor.adaptor.tf_utils.util import get_tensor_by_name
        return [get_tensor_by_name(\
            self.graph, x) for x in self.output_tensor_names]

    def save(self, root=None):
        """Save Tensorflow model."""
        if not root:
            root = cfg.default_workspace + '/save.pb'
        root = os.path.abspath(os.path.expanduser(root))
        # if not have suffix, default append .pb
        os.makedirs(os.path.dirname(root), exist_ok=True)
        pb_file = root if os.path.split(root)[-1].endswith('.pb') else root + '.pb'
        f = tf.io.gfile.GFile(pb_file, 'wb')
        f.write(self.graph_def.SerializeToString())
        logger.info("Save quantized model to {}.".format(pb_file))


class TensorflowSavedModelModel(TensorflowBaseModel):
    """Build Tensorflow saved model."""

    def get_all_weight_names(self):
        """Get weight names of model.

        Returns:
            list: weight names list.
        """
        import tensorflow as tf
        names = []
        for index, layer in enumerate(tf.keras.models.load_model(self._model).layers):
            if len(layer.weights):
                names.append(index)
        return names

    def update_weights(self, tensor_name, new_tensor):
        """Update model weights."""
        pass

    def get_weight(self, tensor_name):
        """Return model wight with a given tensor name.

        Args:
            tensor_name (str): name of a tensor.
        """
        return self.weights[tensor_name]

    @property
    def model(self):
        """Return model itself."""
        import time
        import shutil
        root = os.path.abspath(os.path.expanduser(cfg.default_workspace))
        root += str(time.time())
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)
        if not self._sess:
            self._load_sess(self._model, **self.kwargs)
        _, builder = self.build_saved_model(root)
        builder.save()
        model = tf.saved_model.load(root)
        shutil.rmtree(root)
        return model

    def report_sparsity(self):
        """Get sparsity of the model.

        Returns:
            df (DataFrame): DataFrame of sparsity of each weight.
            total_sparsity (float): total sparsity of model.
        """
        import pandas as pd
        import tensorflow as tf
        import numpy as np
        df = pd.DataFrame(columns=['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)', "Sparsity(%)",
                                   'Std', 'Mean', 'Abs-Mean'])
        pd.set_option('display.precision', 2)
        param_dims = [2, 4]
        params_size = 0
        sparse_params_size = 0
        for index, layer in enumerate(tf.keras.models.load_model(self._model).layers):
            if not len(layer.weights):
                continue
            # Extract just the actual parameter's name, which in this context we treat
            # as its "type"
            weights = layer.get_weights()[0]
            if weights.ndim in param_dims:
                param_size, sparse_param_size, dense_param_size = compute_sparsity(
                    weights)
                density = dense_param_size / param_size
                params_size += param_size
                sparse_params_size += sparse_param_size
                df.loc[len(df.index)] = ([
                    index,
                    list(weights.shape),
                    dense_param_size,
                    sparse_param_size,
                    (1 - density) * 100,
                    np.std(weights),
                    np.mean(weights),
                    np.mean(np.abs(weights))
                ])

        total_sparsity = sparse_params_size / params_size * 100

        df.loc[len(df.index)] = ([
            'Total sparsity:',
            params_size,
            "-",
            int(sparse_params_size),
            total_sparsity,
            0, 0, 0])

        return df, total_sparsity

    def build_saved_model(self, root=None):
        """Build Tensorflow saved model.

        Args:
            root (str, optional): path to saved model. Defaults to None.

        Returns:
            root (str): path to saved model.
            builder (tf.compat.v1.saved_model.builder.SavedModelBuilder): builds 
                the SavedModel protocol buffer and saves variables and assets.
        """
        if not root:
            root = cfg.default_workspace
        root = os.path.abspath(os.path.expanduser(root))
        if os.path.exists(root):
            import shutil
            shutil.rmtree(root)

        os.makedirs(root, exist_ok=True)

        from tensorflow.python.saved_model import signature_constants
        from tensorflow.python.saved_model import tag_constants
        from neural_compressor.adaptor.tf_utils.util import get_tensor_by_name
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(root)
        sigs = {}
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            #(TODO) not directly use self._sess.graph, use self.graph
            tf.import_graph_def(self.graph.as_graph_def(), name="")
            g = tf.compat.v1.get_default_graph()
            inp = [get_tensor_by_name(g, x) for x in self._input_tensor_names]
            out = [get_tensor_by_name(g, x) for x in self._output_tensor_names]
            sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
                {k: v for k, v in zip(self._input_tensor_names, inp)},
                {k: v for k, v in zip(self._output_tensor_names, out)})
            builder.add_meta_graph_and_variables(sess,
                                                 [tag_constants.SERVING],
                                                 signature_def_map=sigs)
        return root, builder

    def save(self, root=None):
        """Save Tensorflow model."""
        root, builder = self.build_saved_model(root)
        builder.save()
        logger.info("Save quantized model to {}.".format(root))

class TensorflowQATModel(TensorflowSavedModelModel):
    """Build Tensorflow QAT model."""

    def __init__(self, model='', **kwargs):
        """Initialize a Tensorflow QAT model.

        Args:
            model (string or tensorflow model object): model path or model object.
        """
        super(TensorflowQATModel, self).__init__(model)
        self.keras_model = None
        self.model_type = 'keras'

    @property
    def model(self):
        """Return model itself."""
        if self.keras_model == None:
            self.keras_model = tf.keras.models.load_model(self._model)
        return self.keras_model

    @model.setter
    def model(self, q_model):
        """Set model itself."""
        self.keras_model = q_model

    def save(self, root=None):
        """Save Tensorflow QAT model."""
        if not root:
            root = cfg.default_workspace + '/saved_model'
        root = os.path.abspath(os.path.expanduser(root))
        # if not have suffix, default append .pb
        os.makedirs(os.path.dirname(root), exist_ok=True)
        q_aware_model = self.keras_model
        q_aware_model.save(root)
        saved_format = 'saved_model'
        if root.endswith('.h5'):
            saved_format = 'h5 file'
        logger.info("Save quantized model to {}.".format(saved_format))
        return root

class TensorflowCheckpointModel(TensorflowBaseModel):
    """Build Tensorflow checkpoint model."""

    @property
    def graph_def(self):
        """Return graph defination."""
        if self.model_type == 'graph_def':
            return self.sess.graph.as_graph_def()
        from neural_compressor.adaptor.tf_utils.util import _parse_ckpt_bn_input
        from tensorflow.python.framework import graph_util
        graph_def = self.sess.graph.as_graph_def()
        graph_def = _parse_ckpt_bn_input(graph_def)
        return graph_util.convert_variables_to_constants(
            sess=self._sess,
            input_graph_def=graph_def,
            output_node_names=self.output_node_names)

    @graph_def.setter
    def graph_def(self, graph_def):
        """Set graph defination."""
        if self._sess is not None:
            self._sess.close()
        output_sess = SESSIONS['graph_def'](graph_def,
                                            self._input_tensor_names, \
                                            self._output_tensor_names)
        self._sess = output_sess[0]
        self._input_tensor_names = output_sess[1]
        self._output_tensor_names = output_sess[2]
        self.model_type = 'graph_def'


TENSORFLOW_MODELS = {'frozen_pb': TensorflowBaseModel,
                     'graph_def': TensorflowBaseModel,
                     'graph': TensorflowBaseModel,
                     'checkpoint': TensorflowCheckpointModel,
                     'estimator': TensorflowBaseModel,
                     'slim': TensorflowBaseModel,
                     'saved_model': TensorflowSavedModelModel,
                     'keras': TensorflowSavedModelModel
                     }

class TensorflowModel(object):
    """A wrapper to construct a Tensorflow Model."""

    def __new__(cls, model_type, root, **kwargs):
        """Create a new instance object of TensorflowModel.

        Args:
            model_type (str): model type.
            root (str): model path.

        Returns:
            tensorflow model object: tensorflow model.
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        model = TENSORFLOW_MODELS[model_type](root, **kwargs)
        model.model_type = model_type
        return model
