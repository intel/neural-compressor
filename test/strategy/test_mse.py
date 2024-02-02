"""Tests for quantization."""

import shutil
import unittest

import numpy as np


def build_fake_model():
    import tensorflow as tf

    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float64, shape=(1, 3, 3, 1), name="x")
            y = tf.constant(np.random.random((2, 2, 1, 1)), name="y")
            op = tf.nn.conv2d(input=x, filter=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["op_to_store"])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1, 3, 3, 1), name="x")
            y = tf.compat.v1.constant(np.random.random((2, 2, 1, 1)), name="y")
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    return graph


def create_test_graph():
    from tensorflow.core.framework import attr_value_pb2, graph_pb2, node_def_pb2
    from tensorflow.python.framework import dtypes, tensor_util

    input_node = node_def_pb2.NodeDef()
    input_node.name = "input"
    input_node.op = "Placeholder"
    input_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))

    conv1_weight_node = node_def_pb2.NodeDef()
    conv1_weight_node.name = "conv1_weights"
    conv1_weight_node.op = "Const"
    conv1_weight_value = np.float32(np.abs(np.random.randn(3, 3, 3, 32)))
    conv1_weight_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv1_weight_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                conv1_weight_value, conv1_weight_value.dtype.type, conv1_weight_value.shape
            )
        )
    )

    conv1_node = node_def_pb2.NodeDef()
    conv1_node.name = "conv1"
    conv1_node.op = "Conv2D"
    conv1_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv1_node.input.extend([input_node.name, conv1_weight_node.name])
    conv1_node.attr["strides"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv1_node.attr["dilations"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv1_node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=b"SAME"))
    conv1_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    bias_node = node_def_pb2.NodeDef()
    bias_node.name = "conv1_bias"
    bias_node.op = "Const"
    bias_value = np.float32(np.abs(np.random.randn(32)))
    bias_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(bias_value, bias_value.dtype.type, bias_value.shape)
        )
    )

    bias_add_node = node_def_pb2.NodeDef()
    bias_add_node.name = "conv1_bias_add"
    bias_add_node.op = "BiasAdd"
    bias_add_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node.input.extend([conv1_node.name, bias_node.name])
    bias_add_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    relu_node = node_def_pb2.NodeDef()
    relu_node.op = "Relu"
    relu_node.name = "relu"
    relu_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node.input.extend([bias_add_node.name])

    conv2_weight_node = node_def_pb2.NodeDef()
    conv2_weight_node.name = "conv2_weights"
    conv2_weight_node.op = "Const"
    conv2_weight_value = np.float32(np.abs(np.random.randn(3, 3, 32, 32)))
    conv2_weight_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv2_weight_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                conv2_weight_value, conv2_weight_value.dtype.type, conv2_weight_value.shape
            )
        )
    )

    conv2_node = node_def_pb2.NodeDef()
    conv2_node.name = "conv2"
    conv2_node.op = "Conv2D"
    conv2_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv2_node.input.extend([relu_node.name, conv2_weight_node.name])
    conv2_node.attr["strides"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv2_node.attr["dilations"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv2_node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=b"SAME"))
    conv2_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    bias_node2 = node_def_pb2.NodeDef()
    bias_node2.name = "conv2_bias"
    bias_node2.op = "Const"
    bias_value2 = np.float32(np.abs(np.random.randn(32)))
    bias_node2.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_node2.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(bias_value2, bias_value2.dtype.type, bias_value2.shape)
        )
    )

    bias_add_node2 = node_def_pb2.NodeDef()
    bias_add_node2.name = "conv2_bias_add"
    bias_add_node2.op = "BiasAdd"
    bias_add_node2.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    bias_add_node2.input.extend([conv2_node.name, bias_node2.name])
    bias_add_node2.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    relu_node2 = node_def_pb2.NodeDef()
    relu_node2.op = "Relu"
    relu_node2.name = "relu2"
    relu_node2.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    relu_node2.input.extend([bias_add_node2.name])

    conv3_weight_node = node_def_pb2.NodeDef()
    conv3_weight_node.name = "conv3_weights"
    conv3_weight_node.op = "Const"
    conv3_weight_value = np.float32(np.abs(np.random.randn(3, 3, 32, 32)))
    conv3_weight_node.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv3_weight_node.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                conv3_weight_value, conv3_weight_value.dtype.type, conv3_weight_value.shape
            )
        )
    )

    conv3_node = node_def_pb2.NodeDef()
    conv3_node.name = "conv3"
    conv3_node.op = "Conv2D"
    conv3_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    conv3_node.input.extend([relu_node2.name, conv3_weight_node.name])
    conv3_node.attr["strides"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv3_node.attr["dilations"].CopyFrom(
        attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=[1, 1, 1, 1]))
    )
    conv3_node.attr["padding"].CopyFrom(attr_value_pb2.AttrValue(s=b"SAME"))
    conv3_node.attr["data_format"].CopyFrom(attr_value_pb2.AttrValue(s=b"NHWC"))

    identity_node = node_def_pb2.NodeDef()
    identity_node.name = "final"
    identity_node.op = "Identity"
    identity_node.attr["T"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
    identity_node.input.extend([conv3_node.name])

    test_graph = graph_pb2.GraphDef()

    test_graph.node.extend(
        [
            input_node,
            conv1_weight_node,
            conv1_node,
            bias_node,
            bias_add_node,
            relu_node,
            conv2_weight_node,
            conv2_node,
            bias_node2,
            bias_add_node2,
            relu_node2,
            conv3_weight_node,
            conv3_node,
            identity_node,
        ]
    )
    return test_graph


def objective_func(params):
    return params["x1"] ** 2 + params["x2"]


class TestQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        self.test_graph = create_test_graph()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("saved", ignore_errors=True)

    def test_run_mse_one_trial(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"]((100, 3, 3, 1), label=True)
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        tune_cri = TuningCriterion(strategy="mse", max_trials=1)
        acc_cri = AccuracyCriterion(tolerable_loss=0.01)

        conf = PostTrainingQuantConfig(quant_level=1, tuning_criterion=tune_cri, accuracy_criterion=acc_cri)

        def fake_eval(model):
            return 1

        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader=dataloader, eval_func=fake_eval)
        self.assertNotEqual(q_model, None)

    def test_run_mse_max_trials(self):
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"]((100, 3, 3, 1), label=True)
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        tune_cri = TuningCriterion(strategy="mse", max_trials=3)
        acc_cri = AccuracyCriterion(tolerable_loss=0.01)

        op_name_dict = {
            "conv1": {
                "activation": {"dtype": ["fp32"]},
            },
        }

        acc = [0, 1, 0.9, 1]

        def fake_eval(model):
            acc.pop(0)
            return acc[0]

        conf = PostTrainingQuantConfig(
            quant_level=1, op_name_dict=op_name_dict, tuning_criterion=tune_cri, accuracy_criterion=acc_cri
        )
        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader=dataloader, eval_func=fake_eval)
        self.assertNotEqual(q_model, None)


if __name__ == "__main__":
    unittest.main()
