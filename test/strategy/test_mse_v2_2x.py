import copy
import os
import shutil
import unittest

import numpy as np
import onnx
import tensorflow as tf
import torch
import torchvision
from onnx import TensorProto, helper, numpy_helper
from onnx import onnx_pb as onnx_proto


def build_ox_model():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5, 5])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 5, 2])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 5, 2])
    H = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 5, 2])

    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())

    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    add = onnx.helper.make_node("Add", ["C", "E"], ["D"], name="add")

    f_value = np.random.randint(2, size=(10)).astype(np.float32)
    F_init = helper.make_tensor("F", TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())
    add2 = onnx.helper.make_node("Add", ["D", "F"], ["H"], name="add2")

    graph = helper.make_graph([matmul_node, add, add2], "test_graph_1", [A], [H], [B_init, E_init, F_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_ox_model2():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 5, 5])
    D = helper.make_tensor_value_info("D", TensorProto.FLOAT, [1, 5, 2])
    H = helper.make_tensor_value_info("H", TensorProto.FLOAT, [1, 5, 2])
    F = helper.make_tensor_value_info("F", TensorProto.FLOAT, [1, 5, 2])

    e_value = np.random.randint(2, size=(10)).astype(np.float32)
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    E_init = helper.make_tensor("E", TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())

    matmul_node = onnx.helper.make_node("MatMul", ["A", "B"], ["C"], name="Matmul")
    add = onnx.helper.make_node("Add", ["C", "E"], ["D"], name="add")

    add2 = onnx.helper.make_node("Add", ["D", "F"], ["H"], name="add2")

    graph = helper.make_graph([matmul_node, add, add2], "test_graph_1", [A, F], [H], [B_init, E_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{"opset_imports": [helper.make_opsetid("", 13)]})
    return model


def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 3, 1), name="x")
            y = tf.constant(np.random.random((2, 2, 1, 1)).astype(np.float32), name="y")
            z = tf.constant(np.random.random((1, 1, 1, 1)).astype(np.float32), name="z")
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")
            op2 = tf.nn.conv2d(
                input=op,
                filters=z,
                strides=[1, 1, 1, 1],
                padding="VALID",
            )
            last_identity = tf.identity(op2, name="op2_to_store")
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op2_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1, 3, 3, 1), name="x")
            y = tf.constant(np.random.random((2, 2, 1, 1)).astype(np.float32), name="y")
            z = tf.constant(np.random.random((1, 1, 1, 1)).astype(np.float32), name="z")
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")
            op2 = tf.nn.conv2d(input=op, filters=z, strides=[1, 1, 1, 1], padding="VALID")
            last_identity = tf.identity(op2, name="op2_to_store")

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op2_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    return graph


class Test_MSEV2Strategy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tf_model = build_fake_model()
        self.torch_model = torchvision.models.resnet18()
        self.onnx_model = build_ox_model()
        self.onnx_model2 = build_ox_model2()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("nc_workspace", ignore_errors=True)

    def test_mse_v2_tf(self):
        i = [0]  # use a mutable type (list) to wrap the int object

        def fake_eval_func(_):
            #               1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            eval_list = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
            i[0] += 1
            return eval_list[i[0]]

        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)

        conf = PostTrainingQuantConfig(
            approach="static", quant_level=1, tuning_criterion=TuningCriterion(strategy="mse_v2")
        )

        q_model = fit(
            model=self.tf_model,
            conf=conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_func=fake_eval_func,
        )
        self.assertIsNotNone(q_model)

    def test_mse_v2_tf_with_confidence_batches(self):
        i = [0]  # use a mutable type (list) to wrap the int object

        def fake_eval_func(_):
            #               1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            eval_list = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
            i[0] += 1
            return eval_list[i[0]]

        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)

        conf = PostTrainingQuantConfig(
            approach="static",
            quant_level=1,
            tuning_criterion=TuningCriterion(
                strategy="mse_v2",
                strategy_kwargs={
                    "confidence_batches": 5,
                },
            ),
        )

        q_model = fit(
            model=self.tf_model,
            conf=conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_func=fake_eval_func,
        )
        self.assertIsNotNone(q_model)

    def test_mse_v2_saved_torch(self):
        i = [0]

        def fake_eval_func(model):
            acc_lst = [1, 1, 0, 0, 0, 0, 1, 1.1, 1.5, 1.1]
            i[0] += 1
            return acc_lst[i[0]]

        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        dataset = Datasets("pytorch")["dummy"](((1, 3, 224, 224)))
        dataloader = DATALOADERS["pytorch"](dataset)

        conf = PostTrainingQuantConfig(
            approach="static", quant_level=1, tuning_criterion=TuningCriterion(strategy="mse_v2")
        )

        q_model = fit(
            model=self.torch_model,
            conf=conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_func=fake_eval_func,
        )
        self.assertIsNotNone(q_model)

    def test_mse_v2_saved_onnx(self):
        i = [0]

        def fake_eval_func(model):
            acc_lst = [1, 1, 0, 0, 0, 0, 1, 1.1, 1.5, 1.1]
            i[0] += 1
            return acc_lst[i[0]]

        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        dataset = Datasets("onnxrt_qdq")["dummy_v2"]((5, 5), (5, 1))
        dataloader = DATALOADERS["onnxrt_qdq"](dataset)

        conf = PostTrainingQuantConfig(
            approach="static", quant_level=1, tuning_criterion=TuningCriterion(strategy="mse_v2", max_trials=9)
        )

        q_model = fit(
            model=self.onnx_model,
            conf=conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_func=fake_eval_func,
        )
        self.assertIsNotNone(q_model)

        i = [0]
        dataset = Datasets("onnxrt_qdq")["dummy_v2"]([(5, 5), (5, 2)], [(5, 1), (5, 1)])
        dataloader = DATALOADERS["onnxrt_qdq"](dataset)
        q_model = fit(
            model=self.onnx_model2,
            conf=conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_func=fake_eval_func,
        )
        self.assertIsNotNone(q_model)


if __name__ == "__main__":
    unittest.main()
