"""
test_tensorflow_inspect_tensor_in_mse_tuning.py:
test inspect_tensor API called by mse tuning strategy
1. Create a quantizer for fake tensorflow model
2. The quantizer fitting process will call inspect_tensor API for both fp32 model and quantized model
3. Check the inspecting result in local disk

Note:
    use '-s' to disable pytest capturing the sys.stderr which will be used in quantization process
"""
import os
import unittest
import yaml
import tensorflow as tf
import numpy as np
import pickle
import logging
import shutil
np.random.seed(0)


def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: input
          outputs: output
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
              name: mse
            accuracy_criterion:
              relative: -0.01
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_model():
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.Session() as sess:
        tf.compat.v1.set_random_seed(0)
        x = tf.compat.v1.placeholder(tf.float32, [1, 28, 28, 1], name="input")
        conv_weights1 = tf.compat.v1.get_variable("weight1", [2, 2, 1, 1],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.conv2d(x, conv_weights1, strides=[1, 2, 2, 1], padding="SAME", name='conv2d_1')
        x = tf.nn.relu(x)
        conv_weights2 = tf.compat.v1.get_variable("weight2", [3, 3, 1, 1],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.conv2d(x, conv_weights2, strides=[1, 3, 3, 1], padding="SAME", name='conv2d_2')
        x = tf.compat.v1.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=1, strides=[1, 2, 2, 1], padding="SAME", name='pool_1')
        # TODO to support inspect max_pool
        x = tf.nn.relu(x, name='output')
        sess.run(tf.compat.v1.global_variables_initializer())
        constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=[x.name.split(':')[0]])

    graph_def.ParseFromString(constant_graph.SerializeToString())

    with graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return graph


def load_data_from_pkl(path, filename):
    try:
        file_path = os.path.join(path, filename)
        with open(file_path, 'rb') as fp:
            data = pickle.load(fp)
            return data
    except FileExistsError:
        logging.getLogger().info('Can not open %s.' % path)


class TestTensorflowInspectTensortinMSETuning(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        self.cfg_path = os.path.join(os.getcwd(), './nc_workspace/')
        self.cfg_file_path = os.path.join(self.cfg_path, 'cfg.pkl')
        self.dumped_tensor_path = os.path.join(os.getcwd(), './nc_workspace/')
        self.dumped_tensor_file_path = os.path.join(self.dumped_tensor_path, 'inspect_result.pkl')

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove(self.dumped_tensor_file_path)
        shutil.rmtree(self.dumped_tensor_path)

    def test_tensorflow_inspect_tensort_in_mse_tuning(self):
        import tensorflow.compat.v1 as tf
        from neural_compressor.experimental import Quantization, common
        tf.disable_v2_behavior()
        model = build_fake_model()
        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(128, 28, 28, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = model
        quantizer.fit()
        self.assertEqual(os.path.exists(self.dumped_tensor_path), True)
        data = load_data_from_pkl(self.dumped_tensor_path, 'inspect_result.pkl')
        self.assertEqual('activation' in data, True)
        self.assertEqual(set(data['activation'][0].keys()), set(['pool_1', 'conv2d_2', 'conv2d_1']))
        self.assertEqual(len(data['activation'][0].keys()), 3)
        self.assertEqual(data['activation'][0]['pool_1']['pool_1'].shape, (1, 3, 3, 1))
        self.assertEqual(data['activation'][0]['conv2d_1']['conv2d_1'].shape, (1, 14, 14, 1))
        self.assertEqual(data['activation'][0]['conv2d_2']['conv2d_2'].shape, (1, 5, 5, 1))


if __name__ == '__main__':
    unittest.main()
