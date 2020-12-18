#
#  -*- coding: utf-8 -*-
#
import unittest
import os
import yaml
import tensorflow as tf

from lpot.adaptor.tf_utils.quantize_graph.quantize_graph_for_intel_cpu import QuantizeGraphForIntel
from lpot.adaptor.tf_utils.graph_rewriter.generic.strip_unused_nodes import StripUnusedNodesOptimizer
from lpot.adaptor.tf_utils.graph_rewriter.generic.fold_batch_norm import FoldBatchNormNodesOptimizer
from tensorflow.python.framework import graph_util
from lpot.adaptor.tensorflow import TensorflowQuery

def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: op_to_store
        device: cpu
        quantization:
          model_wise:
            weight:
                granularity: per_tensor
                scheme: sym
                dtype: int8
                algorithm: minmax
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: basic
            accuracy_criterion:
              relative: 0.1
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestConvBiasAddAddReluFusion(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

        import tensorflow as tf
        self.disable_s8 = bool(tf.version.VERSION < '2.1.0' and
                               tf.version.VERSION != '1.15.0-up1')

    def test_conv_relu_fusion(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        relu = tf.nn.relu(conv)

        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot import Quantization

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            dataloader = quantizer.dataloader(dataset)
            output_graph = quantizer(
                output_graph_def,
                q_dataloader=dataloader,
                eval_dataloader=dataloader
            )
            found_conv_fusion = True

            for i in output_graph.as_graph_def().node:
                if i.op == 'Relu':
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, False)

    def test_conv_biasadd_addv2_relu_fusion(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
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
        add = tf.raw_ops.AddV2(x=normed, y=normed2, name='addv2')
        relu = tf.nn.relu(add)
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            from lpot import Quantization

            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            dataloader = quantizer.dataloader(dataset)
            output_graph = quantizer(
                output_graph_def,
                q_dataloader=dataloader,
                eval_dataloader=dataloader
            )
            found_conv_fusion = False

            for i in output_graph.as_graph_def().node:
                if i.op == 'QuantizedConv2DWithBiasSignedSumAndReluAndRequantize':
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)


class TestGraphConvFusion(unittest.TestCase):
    rn50_fp32_pb_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/resnet50_fp32_pretrained_model.pb'
    pb_path = '/tmp/resnet50_fp32_pretrained_model.pb'
    inputs = ['input']
    outputs = ['predict']

    op_wise_config = {
        "v0/resnet_v13/conv14/conv2d/Conv2D": (False, 'minmax', False),
        "v0/resnet_v13/conv11/conv2d/Conv2D": (False, 'minmax', False),
        "v0/resnet_v17/conv27/conv2d/Conv2D": (False, 'minmax', False)
    }

    @classmethod
    def setUpClass(self):
        os.system('wget {} -O {} '.format(self.rn50_fp32_pb_url, self.pb_path))
        self.input_graph = tf.compat.v1.GraphDef()
        with open(self.pb_path, "rb") as f:
            self.input_graph.ParseFromString(f.read())

    @classmethod
    def tearDownClass(self):
        os.system(
            'rm -rf {}'.format(self.pb_path))

    def test_conv_biasadd_relu_fusion(self):
        tf.compat.v1.disable_eager_execution()

        self._tmp_graph_def = graph_util.remove_training_nodes(self.input_graph, self.outputs)

        self._tmp_graph_def = StripUnusedNodesOptimizer(self._tmp_graph_def,
                                                        self.inputs, self.outputs).do_transformation()

        self._tmp_graph_def = FoldBatchNormNodesOptimizer(self._tmp_graph_def).do_transformation()
        op_wise_sequences = TensorflowQuery(local_config_file=os.path.join(
            os.path.dirname(__file__), "../lpot/adaptor/tensorflow.yaml")).get_eightbit_patterns()

        output_graph = QuantizeGraphForIntel(self._tmp_graph_def, self.outputs,
                                             self.op_wise_config, op_wise_sequences,
                                             'cpu').do_transform()

        node_name_type_mapping = {}
        for i in output_graph.node:
            node_name_type_mapping[i.name] = i.op

        should_disable_sum_node_name = 'v0/resnet_v17/conv27/conv2d/Conv2D_eightbit_quantized_conv'
        should_enable_sum_node_name = 'v0/resnet_v13/conv11/conv2d/Conv2D_eightbit_quantized_conv'
        should_disable_sum_flag = should_disable_sum_node_name in node_name_type_mapping and node_name_type_mapping[
            should_disable_sum_node_name] == 'QuantizedConv2DWithBias'
        should_enable_sum_flag = should_enable_sum_node_name in node_name_type_mapping and node_name_type_mapping[
            should_enable_sum_node_name] == 'QuantizedConv2DWithBiasSumAndRelu'
        self.assertEqual(should_enable_sum_flag, True)
        self.assertEqual(should_disable_sum_flag, True)


if __name__ == '__main__':
    unittest.main()
