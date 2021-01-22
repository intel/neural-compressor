#
#  -*- coding: utf-8 -*-
#
import unittest
import tensorflow as tf

from tensorflow.python.framework import graph_util

def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input 
          outputs: op_to_store 
        device: cpu
        quantization: 
          op_wise: {
                     \"conv1_[1-2]\": {
                       \"activation\":  {\"dtype\": [\"uint8\"]},
                     },
                   }
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
            exit_policy:
              timeout: 200
            accuracy_criterion:
              relative: 0.01
            workspace:
              path: saved
        '''
    with open('fake_yaml.yaml',"w",encoding="utf-8") as f:
        f.write(fake_yaml)
    f.close()


class TestConfigRegex(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        build_fake_yaml()

    def test_config_regex(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        x = tf.compat.v1.placeholder(tf.float32, [1, 56, 56, 16], name="input")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 16, 16],
                                                 initializer=tf.compat.v1.random_normal_initializer())
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID", name='conv1_1')
        normed2 = tf.compat.v1.layers.batch_normalization(conv)

        relu = tf.nn.relu(normed2)

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
                if i.op == 'Conv':
                    found_conv_fusion = False
                    break

            self.assertEqual(found_conv_fusion, True)

if __name__ == '__main__':
    unittest.main()