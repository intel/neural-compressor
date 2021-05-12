#
#  -*- coding: utf-8 -*-
#
import unittest
import tensorflow as tf

from tensorflow.python.framework import graph_util
from lpot.adaptor.tf_utils.graph_rewriter.generic.fuse_gelu import FuseGeluOptimizer
from lpot.adaptor.tf_utils.util import disable_random


class TestGeluFusion(unittest.TestCase):
    def gelu(self, input_tensor, mul_value=0.5, addv2_value=1.0, sqrt_value=2.0):
        cdf = mul_value * (addv2_value + tf.math.erf(input_tensor / tf.sqrt(sqrt_value)))
        return input_tensor * cdf

    def gelu_enable_approximation(self, input_tensor,
                                  another_mul_value=0.5,
                                  mul1_value=0.044715,
                                  addv2_value=1.0,
                                  mul2_value=0.7978845608028654,
                                  pow_value=3):
        coeff = tf.cast(mul1_value, input_tensor.dtype)
        return another_mul_value * input_tensor * (
            addv2_value + tf.tanh(mul2_value *
                                  (input_tensor + coeff * tf.pow(input_tensor, pow_value))))


    def gelu_enable_approximation_varaint(self, input_tensor,
                                  another_mul_value=0.5,
                                  mul1_value=0.044715,
                                  addv2_value=1.0,
                                  mul2_value=0.7978845608028654,
                                  pow_value=3):
        coeff = tf.cast(mul1_value, input_tensor.dtype)
        cdf = another_mul_value * (
            addv2_value + tf.tanh(mul2_value *
                                  (input_tensor + coeff * tf.pow(input_tensor, pow_value))))

        return input_tensor * cdf

    def gelu_disable_approximation(self, input_tensor,
                                   another_add_value=0.5,
                                   mul1_value=0.044715,
                                   addv2_value=1.0,
                                   mul2_value=0.7978845608028654,
                                   pow_value=3):
        coeff = tf.cast(mul1_value, input_tensor.dtype)
        return (another_add_value + input_tensor) * (
            addv2_value + tf.tanh(mul2_value *
                                  (input_tensor + coeff * tf.pow(input_tensor, pow_value))))

    @disable_random()
    def test_gelu_disable_approximation_fusion(self):
        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu_disable_approximation(conv_bias)
        relu = tf.nn.relu(gelu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_approximation_fusion(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu_enable_approximation(conv_bias)
        relu = tf.nn.relu(gelu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, True)

    @disable_random()
    def test_gelu_approximation_fusion_varaint(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu_enable_approximation_varaint(conv_bias)
        relu = tf.nn.relu(gelu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, True)
    @disable_random()
    def test_gelu_approximation_fusion_with_invalid_pow_value(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu_enable_approximation(conv_bias, pow_value=1.0)
        relu = tf.nn.relu(gelu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_approximation_fusion_with_invalid_mul2_value(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu_enable_approximation(conv_bias, mul2_value=1.0)
        relu = tf.nn.relu(gelu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_approximation_fusion_with_invalid_addv2_value(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu_enable_approximation(conv_bias, addv2_value=12.0)
        relu = tf.nn.relu(gelu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_approximation_fusion_with_invalid_mul1_value(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu_enable_approximation(conv_bias, mul1_value=1.0)
        relu = tf.nn.relu(gelu)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_approximation_fusion_with_invalid_another_mul(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu_enable_approximation(conv_bias, another_mul_value=1.0)
        relu = tf.nn.relu(gelu)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_fusion_with_invalid_sqrt(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu(conv_bias, sqrt_value=1.0)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[gelu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_fusion_with_invalid_addv2(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu(conv_bias, addv2_value=10.0)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[gelu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_fusion_with_invalid_mul(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu(conv_bias, mul_value=1.0)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[gelu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, False)

    @disable_random()
    def test_gelu_fusion(self):

        x = tf.compat.v1.placeholder(tf.float32, [1, 224, 224, 3], name="input")

        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 32],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv_bias = tf.compat.v1.get_variable("bias", [32],
                                              initializer=tf.compat.v1.random_normal_initializer())
        conv1 = tf.nn.conv2d(x, conv_weights, strides=[1, 1, 1, 1], padding="SAME")
        conv_bias = tf.math.add(conv1, conv_bias)

        gelu = self.gelu(conv_bias)
        relu = tf.nn.relu(gelu)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[relu.name.split(':')[0]])

            output_graph_def = FuseGeluOptimizer(output_graph_def).do_transformation()

            found_gelu = False
            for i in output_graph_def.node:
                if i.op == 'Gelu':
                    found_gelu = True
                    break

            self.assertEqual(found_gelu, True)


if __name__ == '__main__':
    unittest.main()
