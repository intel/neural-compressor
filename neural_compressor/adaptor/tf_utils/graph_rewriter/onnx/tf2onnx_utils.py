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
"""Utils for Tensorflow model converting to ONNX model."""

import copy
import logging
import os
import re

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from onnx import OperatorSetIdProto, TensorProto, defs, helper, numpy_helper, onnx_pb, shape_inference
from tensorflow.core.framework import tensor_pb2, types_pb2
from tensorflow.python.framework import tensor_util

from neural_compressor.utils.utility import LazyImport

t2o = LazyImport("tf2onnx")

logger = logging.getLogger("neural_compressor")

DEFAULT_OPSET_VERSION = 14

PREFERRED_OPSET = 14

ONNX_UNKNOWN_DIMENSION = -1

NCHW_TO_NHWC = [0, 2, 3, 1]
NHWC_TO_NCHW = [0, 3, 1, 2]


# Built-in supported domains
ONNX_DOMAIN = ""
AI_ONNX_ML_DOMAIN = "ai.onnx.ml"

# Built-in supported opset
AI_ONNX_ML_OPSET = helper.make_opsetid(AI_ONNX_ML_DOMAIN, 2)

ONNX_EMPTY_INPUT = ""

# ignore the following attributes
TF2ONNX_IGNORED_NODE_ATTRS = {
    "T",
    "unknown_rank",
    "_class",
    "Tshape",
    "use_cudnn_on_gpu",
    "Index",
    "Tpaddings",
    "TI",
    "Tparams",
    "Tindices",
    "Tlen",
    "Tdim",
    "Tin",
    "dynamic_size",
    "Tmultiples",
    "Tblock_shape",
    "Tcrops",
    "index_type",
    "Taxis",
    "U",
    "maxval",
    "Tout",
    "Tlabels",
    "Tindex",
    "element_shape",
    "Targmax",
    "Tperm",
    "Tcond",
    "T_threshold",
    "shape_type",
    "_lower_using_switch_merge",
    "parallel_iterations",
    "_num_original_outputs",
    "output_types",
    "output_shapes",
    "key_dtype",
    "value_dtype" "capacity",
    "component_types",
    "shapes",
    "SrcT",
    "Treal",
    "Toutput_types",
    "dense_shapes",
    "Tdense",
    "Tsegmentids",
    "Tshift",
    "Tnumsegments",
}

TF2ONNX_SUBGRAPH_ATTRS = {"body", "cond", "then_branch", "else_branch", "f"}

TF2ONNX_DTYPE_MAP = {
    types_pb2.DT_FLOAT: onnx_pb.TensorProto.FLOAT,
    types_pb2.DT_DOUBLE: onnx_pb.TensorProto.DOUBLE,
    types_pb2.DT_HALF: onnx_pb.TensorProto.FLOAT16,
    types_pb2.DT_BFLOAT16: onnx_pb.TensorProto.FLOAT16,
    types_pb2.DT_INT8: onnx_pb.TensorProto.INT8,
    types_pb2.DT_INT16: onnx_pb.TensorProto.INT16,
    types_pb2.DT_INT32: onnx_pb.TensorProto.INT32,
    types_pb2.DT_UINT8: onnx_pb.TensorProto.UINT8,
    types_pb2.DT_QUINT8: onnx_pb.TensorProto.UINT8,
    types_pb2.DT_QINT8: onnx_pb.TensorProto.INT8,
    types_pb2.DT_UINT16: onnx_pb.TensorProto.UINT16,
    types_pb2.DT_UINT32: onnx_pb.TensorProto.UINT32,
    types_pb2.DT_UINT64: onnx_pb.TensorProto.UINT64,
    types_pb2.DT_INT64: onnx_pb.TensorProto.INT64,
    types_pb2.DT_STRING: onnx_pb.TensorProto.STRING,
    types_pb2.DT_COMPLEX64: onnx_pb.TensorProto.COMPLEX64,
    types_pb2.DT_COMPLEX128: onnx_pb.TensorProto.COMPLEX128,
    types_pb2.DT_BOOL: onnx_pb.TensorProto.BOOL,
    types_pb2.DT_RESOURCE: onnx_pb.TensorProto.INT64,
    types_pb2.DT_VARIANT: onnx_pb.TensorProto.UNDEFINED,
}


#
# mapping dtypes from onnx to numpy
#
ONNX_TO_NUMPY_DTYPE = {
    onnx_pb.TensorProto.FLOAT: np.float32,
    onnx_pb.TensorProto.FLOAT16: np.float16,
    onnx_pb.TensorProto.DOUBLE: np.float64,
    onnx_pb.TensorProto.INT32: np.int32,
    onnx_pb.TensorProto.INT16: np.int16,
    onnx_pb.TensorProto.INT8: np.int8,
    onnx_pb.TensorProto.UINT8: np.uint8,
    onnx_pb.TensorProto.UINT16: np.uint16,
    onnx_pb.TensorProto.UINT32: np.uint32,
    onnx_pb.TensorProto.UINT64: np.uint64,
    onnx_pb.TensorProto.INT64: np.int64,
    onnx_pb.TensorProto.BOOL: bool,
    onnx_pb.TensorProto.COMPLEX64: np.complex64,
    onnx_pb.TensorProto.COMPLEX128: np.complex128,
    onnx_pb.TensorProto.STRING: object,
}

# Mapping opset to IR version.
# Note: opset 7 and opset 8 came out with IR3 but we need IR4 because of PlaceholderWithDefault
# Refer from https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
OPSET_TO_IR_VERSION = {
    1: 3,
    2: 3,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    7: 4,
    8: 4,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 7,
    14: 7,
    15: 8,
    16: 8,
    17: 8,
}


DEFAULT_TARGET = []

INSERTED_OP_NAME = 1


def set_name(name):
    """Set op name for inserted ops."""
    global INSERTED_OP_NAME
    INSERTED_OP_NAME += 1
    return "{}__{}".format(name, INSERTED_OP_NAME)


def find_opset(opset):
    """Find opset."""
    if opset is None or opset == 0:
        opset = defs.onnx_opset_version()
        if opset > PREFERRED_OPSET:
            # if we use a newer onnx opset than most runtimes support, default to the one most supported
            opset = PREFERRED_OPSET
    return opset


def assert_error(bool_val, error_msg, *args):
    """Raise error message."""
    if not bool_val:
        raise ValueError("Assert failure: " + error_msg % args)


def map_numpy_to_onnx_dtype(np_dtype):
    """Map numpy dtype to ONNX dtype."""
    for onnx_dtype, numpy_dtype in ONNX_TO_NUMPY_DTYPE.items():
        if numpy_dtype == np_dtype:
            return onnx_dtype
    raise ValueError("unsupported numpy dtype '%s' for mapping to onnx" % np_dtype)


def map_onnx_to_numpy_type(onnx_type):
    """Map ONNX dtype to numpy dtype."""
    return ONNX_TO_NUMPY_DTYPE[onnx_type]


def add_port_to_name(name, nr=0):
    """Map node output number to name."""
    return name + ":" + str(nr)


def get_tensorflow_node_attr(node, name):
    """Parse tensorflow node attribute."""
    return node.get_attr(name)


def get_tensorflow_tensor_shape(tensor):
    """Get shape from tensorflow tensor."""
    shape = []
    try:
        shape = tensor.get_shape().as_list()
    except Exception:  # pylint: disable=broad-except
        shape = None
    return shape


def get_tensorflow_node_shape_attr(node):
    """Get shape from tensorflow attr "shape"."""
    dims = None
    try:
        shape = get_tensorflow_node_attr(node, "shape")
        if not shape.unknown_rank:
            dims = [int(d.size) for d in shape.dim]
    except:  # pylint: disable=bare-except
        pass
    return dims


def map_tensorflow_dtype(dtype):
    """Convert tensorflow dtype to ONNX."""
    if dtype:
        dtype = TF2ONNX_DTYPE_MAP[dtype]
    return dtype


def get_tensorflow_tensor_data(tensor):
    """Get data from tensorflow tensor."""
    if not isinstance(tensor, tensor_pb2.TensorProto):
        raise ValueError("Require the tensor is instance of TensorProto")
    np_data = tensor_util.MakeNdarray(tensor)
    if not isinstance(np_data, np.ndarray):
        raise ValueError("np_data=", np_data, " isn't ndarray")
    return np_data


def convert_tensorflow_tensor_to_onnx(tensor, name=""):
    """Convert tensorflow tensor to onnx tensor."""
    np_data = get_tensorflow_tensor_data(tensor)
    if np_data.dtype == object:
        # assume np_data is string, numpy_helper.from_array accepts ndarray,
        # in which each item is of str while the whole dtype is of object.
        try:
            # Faster but fails on Unicode
            np_data = np_data.astype(np.str).astype(object)
        except UnicodeDecodeError:
            decode = np.vectorize(lambda x: x.decode("UTF-8"))
            np_data = decode(np_data).astype(object)
        except:  # pylint: disable=bare-except
            raise RuntimeError("Not support type: {}".format(type(np_data.flat[0])))
    return numpy_helper.from_array(np_data, name=name)


def read_tensorflow_node_attrs(node):
    """Read tensorflow node attribute names."""
    attr = {}

    for attr_name in node.node_def.attr:
        value = get_tensorflow_node_attr(node, attr_name)
        if attr_name == "T" and node.type in ("QuantizeV2", "Dequantize"):
            attr[attr_name] = (
                TensorProto.INT8 if get_tensorflow_node_attr(node, attr_name) == "qint8" else TensorProto.UINT8
            )
        elif (
            attr_name in TF2ONNX_IGNORED_NODE_ATTRS
            or attr_name in TF2ONNX_SUBGRAPH_ATTRS
            or isinstance(value, tensor_pb2.TensorProto)
        ):
            pass
        elif attr_name == "shape":
            shape = get_tensorflow_node_shape_attr(node)
            if shape is not None:
                attr[attr_name] = shape
        elif attr_name == "DstT":
            attr["to"] = map_tensorflow_dtype(value)
        elif isinstance(value, tf.DType):
            attr[attr_name] = map_tensorflow_dtype(value)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], tf.DType):
            attr[attr_name] = [map_tensorflow_dtype(v) for v in value]
        else:
            attr[attr_name] = get_tensorflow_node_attr(node, attr_name)

    return attr


def infer_onnx_shape_dtype(node, opset_version, input_shapes, input_dtypes, initializers=None):
    """Infer shapes and dtypes for outputs of the node.

    Sometimes, shape inference needs the values of node's inputs, so initializers are used.
    """

    def build_onnx_op(node):
        """Build onnx op."""
        onnx_node = helper.make_node(node.type, node.input, node.output, name=node.name)
        # deal with attributes
        attr = []
        attr_graphs = node.get_body_graphs()
        if attr_graphs:
            for attr_name, sub_graph in attr_graphs.items():
                copied_sub_graph = copy.deepcopy(sub_graph)
                graph_proto = copied_sub_graph.make_graph("graph for " + node.name + " " + attr_name)
                attr.append(helper.make_attribute(attr_name, graph_proto))
        attr.extend(node.get_onnx_attrs().values())
        if attr:
            onnx_node.attribute.extend(attr)
        return onnx_node

    inputs = []
    outputs = []
    for inp, shape, dtype in zip(node.input, input_shapes, input_dtypes):
        inputs.append(make_onnx_inputs_outputs(inp, dtype, shape))
    for output in node.output:
        outputs.append(make_onnx_inputs_outputs(output, TensorProto.UNDEFINED, None))
    graph_proto = helper.make_graph([build_onnx_op(node)], "infer-graph", inputs, outputs, initializer=initializers)
    imp = OperatorSetIdProto()
    imp.version = opset_version
    model_proto = helper.make_model(graph_proto, opset_imports=[imp])

    inferred_model = None
    try:
        try:
            inferred_model = shape_inference.infer_shapes(model_proto, strict_mode=True)
        except TypeError:
            # strict_mode arg doesn't exist in old onnx packages
            inferred_model = shape_inference.infer_shapes(model_proto)
    except Exception:  # pylint: disable=broad-except
        logger.warning("ONNX Failed to infer shapes and dtypes for [%s, type: %s]", node.name, node.type, exc_info=1)
        return None, None

    shapes = {}
    dtypes = {}
    for output in inferred_model.graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.HasField("elem_type"):
            dtypes[output.name] = tensor_type.elem_type
        else:
            dtypes[output.name] = TensorProto.UNDEFINED
        # Missing dim_value in shapes of onnx means unknown which is -1 in our converter
        if tensor_type.HasField("shape"):
            shapes[output.name] = [dim.dim_value if dim.HasField("dim_value") else -1 for dim in tensor_type.shape.dim]
        else:
            shapes[output.name] = None
    output_shapes = []
    output_dtypes = []
    for output in node.output:
        if output in shapes:
            output_shapes.append(shapes[output])
        else:
            output_shapes.append(None)
        if output in dtypes:
            output_dtypes.append(dtypes[output])
        else:
            output_dtypes.append(TensorProto.UNDEFINED)
    return output_shapes, output_dtypes


def make_onnx_shape(shape):
    """Shape with -1 is not valid in onnx ...

    make it a name.
    """
    if shape:
        # don't do this if input is a scalar
        return [set_name("unk") if i == -1 else i for i in shape]
    return shape


class SeqType:
    """Wrap around TensorProto.* to signify a tensor sequence of a given type."""

    def __init__(self, tensor_dtype):
        """Initlization."""
        self.dtype = tensor_dtype

    def __eq__(self, other):
        """Check if the SeqType is same."""
        if isinstance(other, SeqType):
            return self.dtype == other.dtype
        return NotImplemented

    def __repr__(self):
        """Return string of SeqType's dtype."""
        return "SeqType(%r)" % self.dtype


def make_onnx_inputs_outputs(name, elem_type, shape, **kwargs):
    """Wrapper for creating onnx graph inputs or outputs.

    Args:
        name: Text
        elem_type: TensorProto.DataType
        shape: Optional[Sequence[int]]
    """
    if elem_type is None:
        elem_type = onnx_pb.TensorProto.UNDEFINED
    elif isinstance(elem_type, SeqType):
        return helper.make_tensor_sequence_value_info(name, elem_type.dtype, make_onnx_shape(shape), **kwargs)
    return helper.make_tensor_value_info(name, elem_type, make_onnx_shape(shape), **kwargs)


def save_protobuf(path, message, as_text=False):
    """Save ONNX protobuf file."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if as_text:
        with open(path, "w") as f:
            f.write(text_format.MessageToString(message))
    else:
        with open(path, "wb") as f:
            f.write(message.SerializeToString())


def is_onnx_domain(domain):
    """Check if it's onnx domain."""
    if domain is None or domain == "":
        return True
    return False


def is_list_or_tuple(obj):
    """Check the object is list or tuple."""
    return isinstance(obj, (list, tuple))


def are_shapes_equal(src, dest):
    """Check whether 2 shapes are equal."""
    if src is None:
        return dest is None
    if dest is None:
        return src is None

    assert_error(is_list_or_tuple(src), "invalid type for src")
    assert_error(is_list_or_tuple(dest), "invalid type for dest")

    if len(src) != len(dest):
        return False
    return all(i == j for i, j in zip(src, dest))


def get_subgraphs_from_onnx(model_proto):
    """Returns an iterator over the graphs/subgraphs of a model (using dfs)."""
    stack = [model_proto.graph]
    while stack:
        g = stack.pop()
        yield g
        for node in g.node:
            for attr in node.attribute:
                if hasattr(attr, "g"):
                    stack.append(attr.g)
                if hasattr(attr, "graphs"):
                    stack.extend(attr.graphs)


def initialize_name_counter(model_proto):
    """Avoid name conflicts by initializing the counter used by make_name based on the provided model."""
    suffix_regex = re.compile(r"__(\d+)(:\d+)?$")

    def avoid_name(name):
        global INSERTED_OP_NAME
        suffix = suffix_regex.search(name)
        if suffix:
            INSERTED_OP_NAME = max(INSERTED_OP_NAME, int(suffix.group(1)) + 1)

    for g in get_subgraphs_from_onnx(model_proto):
        for n in g.node:
            avoid_name(n.name)
            for out in n.output:
                avoid_name(out)


def get_index_from_strided_slice_of_shape(node, outputs_to_values):
    """Returns the index of the dimension that the strided slice is reading from the shape node or None."""
    attr_vals = {"shrink_axis_mask": 1, "ellipsis_mask": 0, "begin_mask": 0, "new_axis_mask": 0, "end_mask": 0}
    for a in node.node_def.attr:
        if a in attr_vals:
            i = get_tensorflow_node_attr(node, a)
            if i != attr_vals[a]:
                return None
    i1 = outputs_to_values.get(node.inputs[1].name)
    i2 = outputs_to_values.get(node.inputs[2].name)
    i3 = outputs_to_values.get(node.inputs[3].name)
    if i1 is None or i2 is None or i3 is None:
        return None
    if i1.shape != (1,) or i2.shape != (1,) or i3.shape != (1,):
        return None
    i1, i2, i3 = i1[0], i2[0], i3[0]
    if i1 + 1 != i2 or i3 != 1:
        return None
    return i1


def compute_const_folding_using_tf(g, const_node_values, graph_outputs):
    """Find nodes with constant inputs and compute their values using TF."""
    if const_node_values is None:
        const_node_values = {}
    graph_outputs = set(graph_outputs)

    ops = g.get_operations()
    outputs_to_values = {}
    outputs_to_dtypes = {}
    outputs_to_shapes = {}
    shape_node_outputs = {}

    def is_small_shape(x):
        return np.product(x) <= 1000

    def is_huge_shape(x):
        return np.product(x) >= 1000000

    for node in ops:
        # Load values of constants. Use const_node_values if possible
        if node.type in ["Const", "ConstV2"]:
            tensor = node.node_def.attr["value"].tensor
            if node.name in const_node_values:
                tensor.tensor_content = const_node_values[node.name]
            outputs_to_values[node.outputs[0].name] = get_tensorflow_tensor_data(tensor)
            outputs_to_dtypes[node.outputs[0].name] = node.outputs[0].dtype
        for out in node.outputs:
            outputs_to_shapes[out.name] = get_tensorflow_tensor_shape(out)

    for node in ops:
        if node.type == "Shape":
            shape = outputs_to_shapes.get(node.inputs[0].name)
            if shape is not None:
                shape_node_outputs[node.outputs[0].name] = shape

    unneeded_outputs = set()
    progress = True
    while progress:
        progress = False
        for node in ops:
            # Find ops with constant inputs and compute their values
            input_names = [i.name for i in node.inputs]
            output_names = [i.name for i in node.outputs]
            if (
                node.type == "StridedSlice"
                and input_names[0] in shape_node_outputs
                and output_names[0] not in outputs_to_values
            ):
                shape = shape_node_outputs[input_names[0]]
                i = get_index_from_strided_slice_of_shape(node, outputs_to_values)
                if i is not None and 0 <= i < len(shape) and shape[i] is not None:
                    np_dtype = map_onnx_to_numpy_type(map_tensorflow_dtype(node.outputs[0].dtype))
                    outputs_to_values[output_names[0]] = np.array(shape[i], dtype=np_dtype)
                    outputs_to_dtypes[node.outputs[0].name] = node.outputs[0].dtype
                    progress = True
            can_fold = node.type not in [
                "Enter",
                "Placeholder",
                "PlaceholderWithDefault",
                "Switch",
                "Merge",
                "NextIteration",
                "Exit",
                "QuantizeAndDequantizeV2",
                "QuantizeAndDequantizeV3",
                "QuantizeAndDequantizeV4",
            ]
            can_fold = can_fold and not node.type.startswith("Random")
            can_fold = can_fold and len(input_names) > 0 and all(inp in outputs_to_values for inp in input_names)
            # We can only fold nodes with a single output
            can_fold = can_fold and len(output_names) == 1 and output_names[0] not in outputs_to_values
            # Skip if value already computed, used, and discarded
            can_fold = can_fold and output_names[0] not in unneeded_outputs and output_names[0] not in graph_outputs
            if can_fold:
                # Make a mini graph containing just the node to fold
                g2 = tf.Graph()
                with g2.as_default():
                    for inp in input_names:
                        t2o.tf_loader.tf_placeholder(outputs_to_dtypes[inp], name=inp.split(":")[0])
                    mini_graph_def = g2.as_graph_def()
                    mini_graph_def.node.append(node.node_def)
                g3 = tf.Graph()
                with g3.as_default():
                    feed_dict = {}
                    inp_shapes = []
                    for inp in input_names:
                        inp_np = outputs_to_values[inp]
                        feed_dict[inp] = inp_np
                        inp_shapes.append(inp_np.shape)
                    try:
                        with t2o.tf_loader.tf_session() as sess:
                            tf.import_graph_def(mini_graph_def, name="")
                            results = sess.run(output_names, feed_dict=feed_dict)
                        if is_huge_shape(results[0].shape) and all(is_small_shape(inp) for inp in inp_shapes):
                            logger.debug(
                                "Skipping folding of node %s since result shape %s is much larger "
                                "than input shapes %s",
                                node.name,
                                results[0].shape,
                                inp_shapes,
                            )
                        else:
                            outputs_to_values[output_names[0]] = results[0]
                            outputs_to_dtypes[output_names[0]] = node.outputs[0].dtype
                            progress = True
                    except Exception:  # pylint: disable=broad-except
                        logger.debug("Could not fold node %s", node.name)
        unneeded_outputs.update(outputs_to_values.keys())
        for node in ops:
            # Mark values we need to keep
            input_names = [i.name for i in node.inputs]
            output_names = [i.name for i in node.outputs]
            if len(output_names) == 1 and output_names[0] in outputs_to_values:
                continue
            for i in input_names:
                if i in unneeded_outputs:
                    unneeded_outputs.remove(i)
        for node in unneeded_outputs:
            # Remove unneeded values to prevent memory usage explosion
            if node in outputs_to_values:
                del outputs_to_values[node]
                del outputs_to_dtypes[node]

    for node in ops:
        # We don't need the constants any more
        if node.type in ["Const", "ConstV2"] and node.outputs[0].name in outputs_to_values:
            del outputs_to_values[node.outputs[0].name]
            del outputs_to_dtypes[node.outputs[0].name]

    logger.info("Computed %d values for constant folding", len(outputs_to_values))
    return outputs_to_values, outputs_to_dtypes
