#
#  -*- coding: utf-8 -*-
#
import unittest

from neural_compressor.adaptor.tf_utils.util import disable_random
from neural_compressor.experimental import common
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig, \
             TuningCriterion, AccuracyCriterion, AccuracyLoss, set_random_seed
from neural_compressor.adaptor.tf_utils.util import version1_lt_version2

import tensorflow as tf
from tensorflow.python.framework import graph_util

class TestItexNewAPI(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @disable_random()
    @unittest.skipIf(version1_lt_version2(tf.version.VERSION, '2.8.0'), "Only supports tf greater 2.7.0")
    def test_itex_new_api(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        normed = tf.compat.v1.layers.batch_normalization(conv)
        # relu = tf.nn.relu(normed)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.compat.v1.layers.batch_normalization(conv2)
        # relu2 = tf.nn.relu(normed2)
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

        tuning_criterion = TuningCriterion(
            strategy="basic",
            timeout=0,
            max_trials=100,
            objective="accuracy")

        tolerable_loss = AccuracyLoss(loss=0.01)
        accuracy_criterion = AccuracyCriterion(
            higher_is_better=True,
            criterion='relative',
            tolerable_loss=tolerable_loss)

        config = PostTrainingQuantConfig(
            device="cpu",
            backend="itex",
            quant_format="QDQ",
            inputs=[],
            outputs=[],
            approach="static",
            calibration_sampling_size=[200],
            op_type_list=None,
            op_name_list=None,
            reduce_range=None,
            extra_precisions=[],
            tuning_criterion=tuning_criterion,
            accuracy_criterion=accuracy_criterion)

        from neural_compressor.data import DATASETS
        dataset = DATASETS('tensorflow')['dummy'](shape=(100, 56, 56, 16), label=True)
        output_graph = fit(
            model=common.Model(output_graph_def),
            conf=config,
            calib_dataloader=common.DataLoader(dataset=dataset, batch_size=1),
            calib_func=None,
            eval_dataloader=None,
            eval_func=None,
            eval_metric=None)

        dequant_count = 0
        quantize_count = 0
        for i in output_graph.graph_def.node:
            if i.op == 'Dequantize':
                dequant_count += 1
            if i.op == 'QuantizeV2':
                quantize_count += 1

        self.assertEqual(dequant_count, 5)
        self.assertEqual(quantize_count, 4)

if __name__ == "__main__":
    unittest.main()
