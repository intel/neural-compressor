"""Tests for model."""

import os
import platform
import unittest

import numpy as np
import tensorflow as tf
from pkg_resources import parse_version

from neural_compressor.tensorflow.utils.model import Model


def build_graph():
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session(graph=graph) as sess:
            x = tf.placeholder(tf.float64, shape=(1, 256, 256, 1), name="x")
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
        with tf.compat.v1.Session(graph=graph) as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1, 256, 256, 1), name="x")
            y = tf.compat.v1.constant(np.random.random((3, 3, 1, 1)), name="y")
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    return graph


def build_estimator():
    def model_fn(features, labels, mode):
        logits = tf.keras.layers.Dense(12)(features)
        logits = tf.keras.layers.Dense(56)(logits)
        logits = tf.keras.layers.Dense(4)(logits)

        output_spec = tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=logits)
        return output_spec

    return model_fn


def build_input_fn():
    def input_fun():
        tf.compat.v1.disable_eager_execution()
        raw_dataset = np.ones([100, 224, 224, 3], dtype=np.float32)
        tf_dataset = tf.compat.v1.data.Dataset.from_tensor_slices(raw_dataset)
        tf_dataset = tf_dataset.batch(1)
        ds_iterator = tf_dataset.make_initializable_iterator()
        iter_tensors = ds_iterator.get_next()
        return iter_tensors

    return input_fun


def build_keras():
    from tensorflow import keras

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

    train_images = train_images.astype(np.float32) / 255.0

    # Create Keras model
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(28, 28), name="input"),
            keras.layers.Reshape(target_shape=(28, 28, 1)),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation="softmax", name="output"),
        ]
    )

    # Compile model with optimizer
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # # Train model
    model.fit(x={"input": train_images[0:100]}, y={"output": train_labels[0:100]}, epochs=1)
    return model


class TestModelWrappers(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        os.remove("model_test.pb")

    def test_graph(self):
        graph = build_graph()
        model = Model(graph)
        model.input_tensor_names = ["x"]
        model.output_tensor_names = ["op_to_store"]

        self.assertEqual(True, isinstance(model.graph_def, tf.compat.v1.GraphDef))
        self.assertEqual(model.input_node_names[0], "x")
        self.assertEqual(model.output_node_names[0], "op_to_store")
        model.save("model_test.pb")

        model = Model("model_test.pb")
        self.assertEqual(model.input_tensor_names[0], "x")
        self.assertEqual(model.output_tensor_names[0], "op_to_store")
        self.assertEqual(model.input_tensor[0].name, "x:0")
        self.assertEqual(model.output_tensor[0].name, "op_to_store:0")

        # test wrong input tensor names can't set
        with self.assertRaises(AssertionError):
            model.input_tensor_names = ["wrong_input"]
        with self.assertRaises(AssertionError):
            model.output_tensor_names = ["wrong_output"]

        # test right tensor
        model.input_tensor_names = ["x_1"]
        model.output_tensor_names = ["op_to_store_1"]
        self.assertEqual(True, isinstance(model.graph_def, tf.compat.v1.GraphDef))

    def test_validate_graph_node(self):
        from neural_compressor.tensorflow.utils.model_wrappers import validate_graph_node

        graph = build_graph()
        self.assertEqual(False, validate_graph_node(graph.as_graph_def(), []))
        self.assertEqual(False, validate_graph_node(graph.as_graph_def(), ["test"]))
        self.assertEqual(True, validate_graph_node(graph.as_graph_def(), ["x"]))

    @unittest.skipIf(
        parse_version(tf.version.VERSION) >= parse_version("2.16.1"),
        "Only supports tf previous to the version 2.16.1",
    )
    def test_estimator(self):
        model_fn = build_estimator()
        input_fn = build_input_fn()
        estimator = tf.estimator.Estimator(model_fn, model_dir=None, config=None, params=None, warm_start_from=None)
        with self.assertRaises(AssertionError):
            graph_def = Model(estimator).graph_def
        model = Model(estimator, input_fn=input_fn)

        self.assertEqual(model.output_tensor_names[0], "dense_2/BiasAdd:0")

    def test_ckpt(self):
        mobilenet_ckpt_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz"
        dst_path = "/tmp/.neural_compressor/mobilenet_v1_1.0_224.tgz"
        if platform.system().lower() == "windows":
            model_path = "C:\\tmp\.neural_compressor\\mobilenet_v1_1.0_224"
        else:
            model_path = "./ckpt"
            if not os.path.exists(dst_path):
                os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {}".format(mobilenet_ckpt_url, dst_path))
            if not os.path.getsize(dst_path):
                os.system("rm -fr {0} && wget {1} -O {0}".format(dst_path, mobilenet_ckpt_url))
            os.system("mkdir -p ckpt && tar xvf {0} -C {1}".format(dst_path, model_path))
        model = Model(model_path)
        model.output_tensor_names = ["MobilenetV1/Predictions/Reshape_1"]

        self.assertEqual(model_path, model.model_path)
        self.assertGreaterEqual(len(model.input_tensor_names), 1)
        self.assertEqual(len(model.output_tensor_names), 1)
        graph_def = model.graph_def
        self.assertEqual(True, isinstance(graph_def, tf.compat.v1.GraphDef))

        model.graph_def = graph_def
        os.system("rm -rf ckpt")

    def test_slim(self):
        tf.compat.v1.reset_default_graph()
        inception_ckpt_url = "http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz"
        if platform.system().lower() == "windows":
            dst_path = "C:\\tmp\\.neural_compressor\\inception_v1_2016_08_28.tar.g"
        elif platform.system().lower() == "linux":
            dst_path = "/tmp/.neural_compressor/slim/inception_v1_2016_08_28.tar.gz"
        if platform.system().lower() == "linux":
            if not os.path.exists(dst_path):
                os.system("mkdir -p /tmp/.neural_compressor/slim")
                os.system("wget {} -O {}".format(inception_ckpt_url, dst_path))
            if not os.path.getsize(dst_path):
                os.system("rm -fr {0} && wget {1} -O {0}".format(dst_path, inception_ckpt_url))
            os.system("mkdir -p slim_ckpt && tar xvf {} -C slim_ckpt".format(dst_path))
        if parse_version(tf.version.VERSION) > parse_version("2.0.0"):
            return
        model = Model("./slim_ckpt/inception_v1.ckpt")
        model.name = "inception_v1"
        graph_def = model.graph_def
        self.assertGreaterEqual(len(model.output_node_names), 1)
        self.assertGreaterEqual(len(model.input_node_names), 1)
        self.assertEqual(model.model_path, "./slim_ckpt/inception_v1.ckpt")
        # test net factory
        from neural_compressor.tensorflow.utils.utility import TFSlimNetsFactory

        factory = TFSlimNetsFactory()
        from tf_slim.nets import inception

        input_shape = [None, 224, 224, 3]
        model_func = inception.inception_v1
        arg_scope = inception.inception_v1_arg_scope
        num_classes = 1001
        factory.register("inceptionv1", model_func, input_shape, arg_scope, num_classes=num_classes)
        os.system("rm -rf slim_ckpt")

    @unittest.skipIf(
        parse_version(tf.version.VERSION) < parse_version("2.4.0") or platform.system().lower() == "windows",
        "Only supports tf 2.4.0 or above",
    )
    def test_saved_model(self):
        ssd_resnet50_ckpt_url = "http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz"
        center_resnet50_saved_model_url = (
            "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1?tf-hub-format=compressed"
        )
        dst_path = "/tmp/.neural_compressor/saved_model.tar.gz"
        center_dst_path = "/tmp/.neural_compressor/center_saved_model.tar.gz"

        if not os.path.exists(dst_path):
            os.system("mkdir -p /tmp/.neural_compressor && wget {} -O {}".format(ssd_resnet50_ckpt_url, dst_path))
        if not os.path.getsize(dst_path):
            os.system("rm -fr {0} && wget {1} -O {0}".format(dst_path, ssd_resnet50_ckpt_url))
        if not os.path.exists(center_dst_path):
            os.system(
                "mkdir -p /tmp/.neural_compressor && wget {} -O {}".format(
                    center_resnet50_saved_model_url, center_dst_path
                )
            )
        if not os.path.getsize(center_dst_path):
            os.system("rm -fr {0} && wget {1} -O {0}".format(center_dst_path, center_resnet50_saved_model_url))

        os.system("tar -xvf {}".format(dst_path))
        unzip_center_model = "unzip_center_model"
        os.system("mkdir -p {} ".format(unzip_center_model))
        os.system("tar -xvf {} -C {}".format(center_dst_path, unzip_center_model))

        model = Model("ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model")
        center_model = Model("unzip_center_model")

        from tensorflow.python.trackable.autotrackable import AutoTrackable

        assert isinstance(
            model.model, AutoTrackable
        ), "The model getter of TensorflowSavedModelModel is not correctly run."

        from tensorflow.compat.v1 import graph_util

        graph_def = graph_util.convert_variables_to_constants(
            sess=model.sess, input_graph_def=model.graph_def, output_node_names=model.output_node_names
        )

        model.graph_def = graph_def
        tmp_saved_model_path = "./tmp_saved_model"

        if os.path.exists(tmp_saved_model_path):
            os.system("rm -rf {}".format(tmp_saved_model_path))
        os.system("mkdir -p {}".format(tmp_saved_model_path))

        self.assertTrue(isinstance(model.graph_def, tf.compat.v1.GraphDef))
        self.assertTrue(isinstance(model.graph, tf.compat.v1.Graph))
        model.save(tmp_saved_model_path)
        # load again to make sure model can be loaded
        model = Model(tmp_saved_model_path)
        os.system("rm -rf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03")
        os.system("rm -rf temp_saved_model")
        os.system("rm -rf {}".format(tmp_saved_model_path))

        center_graph_def = graph_util.convert_variables_to_constants(
            sess=center_model.sess,
            input_graph_def=center_model.graph_def,
            output_node_names=center_model.output_node_names,
        )

        center_model.graph_def = center_graph_def

        self.assertTrue(isinstance(center_model.graph_def, tf.compat.v1.GraphDef))
        self.assertTrue(isinstance(center_model.graph, tf.compat.v1.Graph))

        from neural_compressor.tensorflow.utils.model_wrappers import _get_graph_from_saved_model_v1

        graph_def, input_names, output_names = _get_graph_from_saved_model_v1(unzip_center_model)
        assert graph_def is not None, "Can not parse the saved model..."

        from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info

        from neural_compressor.tensorflow.utils.model_wrappers import _contains_function_with_implements_attr

        saved_model_proto, _ = parse_saved_model_with_debug_info(unzip_center_model)
        self.assertEqual(False, _contains_function_with_implements_attr(saved_model_proto))

        os.system("rm -rf unzip_center_model")


if __name__ == "__main__":
    unittest.main()
