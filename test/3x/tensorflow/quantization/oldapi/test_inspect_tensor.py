"""
test_tensorflow_inspect_tensor_in_mse_tuning.py:
test inspect_tensor API called by mse tuning strategy
1. Create a quantizer for fake tensorflow model
2. The quantizer fitting process will call inspect_tensor API for both fp32 model and quantized model
3. Check the inspecting result in local disk

Note:
    use '-s' to disable pytest capturing the sys.stderr which will be used in quantization process
"""

import logging
import os
import pickle
import platform
import shutil
import unittest

import numpy as np

np.random.seed(0)

def build_fake_model():
    import tensorflow as tf

    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.Session() as sess:
        tf.compat.v1.set_random_seed(0)
        x = tf.compat.v1.placeholder(tf.float32, [1, 28, 28, 1], name="input")
        conv_weights1 = tf.compat.v1.get_variable(
            "weight1", [2, 2, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        x = tf.nn.conv2d(x, conv_weights1, strides=[1, 2, 2, 1], padding="SAME", name="conv2d_1")
        x = tf.nn.relu(x)
        conv_weights2 = tf.compat.v1.get_variable(
            "weight2", [3, 3, 1, 1], initializer=tf.compat.v1.random_normal_initializer()
        )
        x = tf.nn.conv2d(x, conv_weights2, strides=[1, 3, 3, 1], padding="SAME", name="conv2d_2")
        x = tf.compat.v1.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=1, strides=[1, 2, 2, 1], padding="SAME", name="pool_1")
        # TODO to support inspect max_pool
        x = tf.nn.relu(x, name="output")
        sess.run(tf.compat.v1.global_variables_initializer())
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=sess.graph_def, output_node_names=[x.name.split(":")[0]]
        )

    graph_def.ParseFromString(constant_graph.SerializeToString())

    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph


def load_data_from_pkl(path, filename):
    try:
        file_path = os.path.join(path, filename)
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
            return data
    except FileExistsError:
        logging.getLogger().info("Can not open %s." % path)


class TestTensorflowInspectTensortinMSETuning(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if platform.system().lower() == "linux":
            self.cfg_path = os.path.join(os.getcwd(), "./nc_workspace/")
            self.dumped_tensor_path = os.path.join(os.getcwd(), "./nc_workspace/")
        else:
            self.cfg_path = os.path.join(os.getcwd(), "nc_workspace\\")
            self.dumped_tensor_path = os.path.join(os.getcwd(), "nc_workspace\\")
        self.cfg_file_path = os.path.join(self.cfg_path, "cfg.pkl")
        self.dumped_tensor_file_path = os.path.join(self.dumped_tensor_path, "inspect_result.pkl")

    @classmethod
    def tearDownClass(self):
        os.remove(self.dumped_tensor_file_path)
        shutil.rmtree(self.dumped_tensor_path)

    def test_tensorflow_inspect_tensor(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        model = build_fake_model()

        from neural_compressor.tensorflow import quantize_model
        from neural_compressor.tensorflow.utils import BaseDataLoader, DummyDataset
        
        dataset = DummyDataset(shape=(128, 28, 28, 1), label=True)
        calib_dataloader = BaseDataLoader(dataset)
        quant_config = {
            "static_quant": {
                "global": {
                    "weight_dtype": "int8",
                    "weight_sym": True,
                    "weight_granularity": "per_tensor",
                    "weight_algorithm": "minmax",
                },
            }
        }
        qmodel = quantize_model(model, quant_config, calib_dataloader)

        self.assertEqual(os.path.exists(self.dumped_tensor_path), True)
        data = load_data_from_pkl(self.dumped_tensor_path, "inspect_result.pkl")
        self.assertEqual("activation" in data, True)
        self.assertEqual(set(data["activation"][0].keys()), set(["pool_1", "conv2d_2", "conv2d_1"]))
        self.assertEqual(len(data["activation"][0].keys()), 3)
        self.assertEqual(data["activation"][0]["pool_1"]["pool_1"].shape, (1, 3, 3, 1))
        self.assertEqual(data["activation"][0]["conv2d_1"]["conv2d_1"].shape, (1, 14, 14, 1))
        self.assertEqual(data["activation"][0]["conv2d_2"]["conv2d_2"].shape, (1, 5, 5, 1))


if __name__ == "__main__":
    unittest.main()
