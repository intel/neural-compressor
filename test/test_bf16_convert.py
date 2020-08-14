import unittest
import tensorflow as tf
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from ilit.adaptor.tf_utils.transform_graph.bf16_convert import BF16Convert

class TestBF16Convert(unittest.TestCase):

    def create_test_graph(self):
        input_node = node_def_pb2.NodeDef()
        input_node.name = "input"
        input_node.op = "Placeholder"
        input_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
            type=dtypes.float32.as_datatype_enum))
        
        conv1_weight_node = node_def_pb2.NodeDef()
        conv1_weight_node.name = "conv1_weights"
        conv1_weight_node.op = "Const"
        conv1_weight_value = np.float32(np.abs(np.random.randn(3,3,3,32)))
        conv1_weight_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
        conv1_weight_value, conv1_weight_value.dtype.type, conv1_weight_value.shape)))
        
        conv1_node = node_def_pb2.NodeDef()
        conv1_node.name = "conv1"
        conv1_node.op = "Conv2D"
        conv1_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
            type=dtypes.float32.as_datatype_enum))
        conv1_node.input.extend([input_node.name, conv1_weight_node.name])
        conv1_node.attr['strides'].CopyFrom(attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(i=[1,2,2,1])))
        conv1_node.attr['dilations'].CopyFrom(attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
        conv1_node.attr['padding'].CopyFrom(attr_value_pb2.AttrValue(s=b'SAME'))
        conv1_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))
        
        bias_node = node_def_pb2.NodeDef()
        bias_node.name = "conv1_bias"
        bias_node.op = "Const"
        bias_value = np.float32(np.abs(np.random.randn(32)))
        bias_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            bias_value, bias_value.dtype.type, bias_value.shape)))
        
        bias_add_node = node_def_pb2.NodeDef()
        bias_add_node.name = "conv1_bias_add"
        bias_add_node.op = "BiasAdd"
        bias_add_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        bias_add_node.input.extend([conv1_node.name, bias_node.name])
        bias_add_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))
        
        relu_node = node_def_pb2.NodeDef()
        relu_node.op = "Relu"
        relu_node.name = "relu"
        relu_node.input.extend([bias_add_node.name])
        
        conv2_weight_node = node_def_pb2.NodeDef()
        conv2_weight_node.name = "conv2_weights"
        conv2_weight_node.op = "Const"
        conv2_weight_value = np.float32(np.abs(np.random.randn(3,3,3,32)))
        conv2_weight_node.attr['value'].CopyFrom(attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
        conv2_weight_value, conv2_weight_value.dtype.type, conv2_weight_value.shape)))
        
        conv2_node = node_def_pb2.NodeDef()
        conv2_node.name = "conv2"
        conv2_node.op = "Conv2D"
        conv2_node.attr['T'].CopyFrom(attr_value_pb2.AttrValue(
            type=dtypes.float32.as_datatype_enum))
        conv2_node.input.extend([input_node.name, conv2_weight_node.name])
        conv2_node.attr['strides'].CopyFrom(attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(i=[1,2,2,1])))
        conv2_node.attr['dilations'].CopyFrom(attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(i=[1,1,1,1])))
        conv2_node.attr['padding'].CopyFrom(attr_value_pb2.AttrValue(s=b'SAME'))
        conv2_node.attr['data_format'].CopyFrom(attr_value_pb2.AttrValue(s=b'NHWC'))
        self.test_graph = graph_pb2.GraphDef()

        self.test_graph.node.extend([input_node, 
                                     conv1_weight_node, 
                                     conv1_node, 
                                     bias_node, 
                                     bias_add_node, 
                                     relu_node,
                                     conv2_weight_node, 
                                     conv2_node, 
                                    ])

    def test_do_transform(self):
        self.create_test_graph()
        bf16_converter = BF16Convert(self.test_graph, "cpu", ["conv2"], [], ["conv1"])
        new_graph = bf16_converter.do_transformation()
        bf16_converter._parse_graph()
        new_conv1 = bf16_converter.node_name_mapping["conv1"].node
        self.assertEqual(new_conv1.attr["T"].type, dtypes.bfloat16)
        self.assertTrue("input_FP32toBF16" in new_conv1.input)

if __name__ == "__main__":
    unittest.main()
             
