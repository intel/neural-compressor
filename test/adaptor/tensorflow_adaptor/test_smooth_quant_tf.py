import unittest
import tensorflow as tf
from neural_compressor.data import Datasets
from neural_compressor.adaptor.tf_utils.smooth_quant import TFSmoothQuant
from neural_compressor.adaptor.tf_utils.util import disable_random
from tensorflow.python.framework import graph_util
import yaml
import os

def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
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
            exit_policy:
              performance_only: True
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


class TestSmoothQuantTF(unittest.TestCase):
    # @classmethod
    # def setUpClass(self):
    #     class RandDataloader:
    #         def __init__(self):
    #             pass

    #         def __iter__(self):
    #             yield tf.constant((1, 3, 1, 1))

    #     self.dl = RandDataloader()
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
    
    # @disable_random()
    # @unittest.skipIf(tf.__version__ < "2.0", "does not support on 1.15up3")
    # def test_sq_depthwiseconv(self): # test_depthwiseconv_biasadd_fusion_with_negative_input
    #     x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
    #     top_relu = tf.nn.relu(x)
    #     paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    #     x_pad = tf.pad(top_relu, paddings, "CONSTANT")
    #     conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
    #                                              initializer=tf.compat.v1.random_normal_initializer())
    #     conv = tf.nn.depthwise_conv2d(x_pad, conv_weights, strides=[1, 1, 1, 1], padding="VALID")

    #     normed = tf.compat.v1.layers.batch_normalization(conv, name='op_to_store')
    #     out_name = normed.name.split(':')[0]

    #     with tf.compat.v1.Session() as sess:
    #         sess.run(tf.compat.v1.global_variables_initializer())
    #         output_graph_def = graph_util.convert_variables_to_constants(
    #             sess=sess,
    #             input_graph_def=sess.graph_def,
    #             output_node_names=[out_name])
            
    #         f = tf.io.gfile.GFile('debug_conv.pb', 'wb')
    #         f.write(output_graph_def.SerializeToString())

    #         from neural_compressor.experimental import Quantization, common
    #         quantizer = Quantization('fake_yaml.yaml')
    #         dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
    #         quantizer.eval_dataloader = common.DataLoader(dataset)
    #         quantizer.calib_dataloader = common.DataLoader(dataset)
    #         quantizer.model = output_graph_def
    #         output_graph = quantizer.fit()
    #         found_conv_fusion = False
            
    #         f = tf.io.gfile.GFile('debug_conv_after_quant.pb', 'wb')
    #         f.write(output_graph_def.SerializeToString())

    #         for i in output_graph.graph_def.node:
    #             if i.op == 'QuantizedDepthwiseConv2DWithBias':
    #                 found_conv_fusion = True
    #                 break

    #         self.assertEqual(found_conv_fusion, True)

    @disable_random()
    def test_sq_conv(self):  # test_conv_biasadd_add_relu_fusion
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)

        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 16, 16],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        conv2 = tf.nn.conv2d(top_relu, conv_weights2, strides=[1, 2, 2, 1], padding="SAME")
        normed2 = tf.nn.bias_add(conv2, tf.constant([3.0, 1.2,1,2,3,4,5,6,7,8,9,0,12,2,3,4]))
        relu = tf.nn.relu(normed2 + tf.constant([3.0]))
        relu6 = tf.nn.relu6(relu, name='op_to_store')

        out_name = relu6.name.split(':')[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=sess.graph_def,
                output_node_names=[out_name])
            f = tf.io.gfile.GFile('debug_conv.pb', 'wb')
            f.write(output_graph_def.SerializeToString())
            
            
            ######## Here use smoothquant
            from neural_compressor.experimental import common
            # sq = TFSmoothQuant(output_graph_def, common.DataLoader(quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)))

            from neural_compressor.data import Datasets
            dataset = Datasets('tensorflow')['dummy']((), {'shape': (100, 56, 56, 16), 'label': True})
            dataloader = common.DataLoader(dataset)
            sq = TFSmoothQuant(output_graph_def, dataloader)
            sq.smooth_transform(alpha=0.5, calib_iter=5)
            ########

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 56, 56, 16), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()

            found_conv_fusion = False
            f = tf.io.gfile.GFile('debug_conv_after_quant.pb', 'wb')
            f.write(output_graph.graph_def.SerializeToString())
            for i in output_graph.graph_def.node:
                if i.op.find('QuantizedConv2D') != -1:
                    found_conv_fusion = True
                    break

            self.assertEqual(found_conv_fusion, True)


if __name__ == '__main__':
    unittest.main()
