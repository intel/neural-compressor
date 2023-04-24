import unittest
import tensorflow as tf
from neural_compressor.adaptor.tf_utils.util import disable_random
from neural_compressor.data.dataloaders.dataloader import DataLoader
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.utils.utility import set_random_seed
from tensorflow.python.framework import graph_util


class TestSmoothQuantTF(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @disable_random()
    def test_conv_sq(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        add = tf.raw_ops.Add(x=normed, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])

        set_random_seed(9527)
        config = PostTrainingQuantConfig(
            quant_level=1,
            recipes={"smooth_quant": True, "smooth_quant_args": {'alpha': 0.6}},
            calibration_sampling_size=[200])

        from neural_compressor.data import Datasets
        dataset = Datasets('tensorflow')['dummy'](shape=(100, 56, 56, 16), label=True)
        dataloader = DataLoader(framework='tensorflow', dataset=dataset, batch_size=1)
        output_graph = fit(
            model=output_graph_def,
            conf=config,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader)

        mul_count = 0
        for i in output_graph.graph_def.node:
            if i.op == 'Mul':
                mul_count += 1

        self.assertEqual(mul_count, 2)


if __name__ == '__main__':
    unittest.main()
