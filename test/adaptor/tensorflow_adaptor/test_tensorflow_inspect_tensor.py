"""
test_tensorflow_inspect_tensor.py: test inspect_tensor API
1. Create a quantizer for fake tensorflow model
2. Call inspect_tensor to dump the activation in local disk for both fp32 model and quantized model
3. Compare the inspecting result between fp32 model and quantized model

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
from packaging import version

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
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(y, f)
    f.close()


def build_fake_model():
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.Session() as sess:
        tf.compat.v1.set_random_seed(0)
        x = tf.compat.v1.placeholder(tf.float32, [1, 64, 64, 3], name='input')
        conv_weights1 = tf.compat.v1.get_variable('weight1', [2, 2, 3, 3],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.conv2d(x, conv_weights1, strides=[1, 2, 2, 1], padding='SAME', name='conv2d_1')
        x = tf.nn.relu(x)
        conv_weights2 = tf.compat.v1.get_variable('weight2', [3, 3, 3, 3],
                                                  initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.conv2d(x, conv_weights2, strides=[1, 3, 3, 1], padding='SAME', name='conv2d_2')
        x = tf.compat.v1.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        depthwise_weights = tf.compat.v1.get_variable('depthwise_weights', [3, 3, 3, 6],
                                                      initializer=tf.compat.v1.random_normal_initializer())
        x = tf.nn.depthwise_conv2d(x, depthwise_weights, strides=[1, 1, 1, 1], padding='VALID',
                                   name='depthwise_conv2d_1')
        x = tf.nn.max_pool(x, ksize=2, strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
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

def build_fake_diagnosis_yaml():
    fake_diagnosis_yaml = '''
        model:
          name: fake_diagnosis_yaml
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
            diagnosis:
                diagnosis_after_tuning: True
                op_list:  conv2d_1 conv2d_2 depthwise_conv2d_1
                iteration_list: 1
                inspect_type: activation
                save_to_disk: True
        '''
    y = yaml.load(fake_diagnosis_yaml, Loader=yaml.SafeLoader)
    with open('fake_diagnosis_yaml.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(y, f)
    f.close()


def build_fake_diagnosis_yaml2():
    fake_diagnosis_yaml2 = '''
        model:
          name: fake_diagnosis_yaml2
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
            diagnosis:
                diagnosis_after_tuning: True
                iteration_list: 1
                inspect_type: activation
                save_to_disk: True
                save_path: save_path_test
        '''
    y = yaml.load(fake_diagnosis_yaml2, Loader=yaml.SafeLoader)
    with open('fake_diagnosis_yaml2.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(y, f)
    f.close()

class TestTensorflowInspectTensor(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        build_fake_yaml()
        build_fake_diagnosis_yaml()
        build_fake_diagnosis_yaml2()
        self.model = build_fake_model()
        self.fp32_dumped_tensor_path = os.path.join(os.getcwd(), './fake_graph_inspect_res_fp32/')
        self.quan_dumped_tensor_path = os.path.join(os.getcwd(), './fake_graph_inspect_res_quan/')
        self.fp32_dumped_tensor_file_path = os.path.join(self.fp32_dumped_tensor_path, 'inspect_result.pkl')
        self.quan_dumped_tensor_file_path = os.path.join(self.quan_dumped_tensor_path, 'inspect_result.pkl')
        self.workspace = os.path.join(os.getcwd(), 'nc_workspace')

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove(self.fp32_dumped_tensor_file_path)
        os.rmdir(self.fp32_dumped_tensor_path)
        os.remove(self.quan_dumped_tensor_file_path)
        os.rmdir(self.quan_dumped_tensor_path)
        shutil.rmtree(self.workspace)
        shutil.rmtree(os.path.join(os.getcwd(), 'save_path_test'))

    def test_tensorflow_inspect_tensor(self):
        import tensorflow.compat.v1 as tf
        from neural_compressor.experimental import Quantization, common
        from neural_compressor.utils.utility import load_data_from_pkl
        tf.disable_v2_behavior()
        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(128, 64, 64, 3), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.model
        q_model = quantizer.fit()
        self.quan_graph_def = q_model.graph_def
        self.fp32_graph_def = quantizer.model.graph_def
        self.dataloader = quantizer.calib_dataloader
        self.node_list = ['conv2d_1', 'conv2d_2', 'depthwise_conv2d_1']
        # Tensorflow 2.5.0 enabled the s8 input for pooling op
        # TODO check the specific version 
        if version.parse(tf.version.VERSION) >= version.parse('2.6.0'):
            self.node_list.append('pool_1')
        self.quantizer = quantizer
        self.iteration_list = [1, 5]

        logging.getLogger().debug(f'Start to inspect tensor :{self.node_list} in  fp32 model.')
        quantizer = self.quantizer
        quantizer.strategy.adaptor.inspect_tensor(self.fp32_graph_def, dataloader=self.dataloader,
                                                  op_list=self.node_list, iteration_list=self.iteration_list,
                                                  inspect_type='all', save_to_disk=True,
                                                  save_path=self.fp32_dumped_tensor_path,
                                                  quantization_cfg=quantizer.strategy.tune_cfg)
        self.assertEqual(os.path.exists(self.fp32_dumped_tensor_file_path), True)

        logging.getLogger().debug(f'Start to inspect tensor :{self.node_list} in  quan model.')
        quantizer = self.quantizer
        quantizer.strategy.adaptor.inspect_tensor(self.quan_graph_def, dataloader=self.dataloader,
                                                  op_list=self.node_list, iteration_list=self.iteration_list,
                                                  inspect_type='all', save_to_disk=True,
                                                  save_path=self.quan_dumped_tensor_path,
                                                  quantization_cfg=quantizer.strategy.tune_cfg)
        self.assertEqual(os.path.exists(self.quan_dumped_tensor_file_path), True)


        fp32_data = load_data_from_pkl(self.fp32_dumped_tensor_path, 'inspect_result.pkl')
        quan_data = load_data_from_pkl(self.quan_dumped_tensor_path, 'inspect_result.pkl')
        self.assertEqual(fp32_data.keys(), quan_data.keys())
        self.assertIn('activation', fp32_data)
        self.assertEqual(len(fp32_data['activation']), len(quan_data['activation']))  # have same itertaion index
        self.assertEqual(len(self.iteration_list),len(fp32_data['activation']))
        for iter_indx, iter in enumerate(self.iteration_list):
            fp32_iter_data = fp32_data['activation'][iter_indx]
            quan_iter_data = quan_data['activation'][iter_indx]
            for node_name in fp32_iter_data.keys():
                self.assertEqual(fp32_iter_data[node_name][node_name].shape, quan_iter_data[node_name][node_name].shape)

    def test_tensorflow_diagnosis(self):
        import tensorflow.compat.v1 as tf
        from neural_compressor.experimental import Quantization, common
        tf.disable_v2_behavior()
        quantizer = Quantization('fake_diagnosis_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(128, 64, 64, 3), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.model
        quantizer.fit()
        self.assertEqual(os.path.exists(os.path.join(os.getcwd(), './nc_workspace/inspect_saved/fp32/inspect_result.pkl')), True)
        self.assertEqual(os.path.exists(os.path.join(os.getcwd(), './nc_workspace/inspect_saved/quan/inspect_result.pkl')), True)

    def test_tensorflow_diagnosis2(self):
        import tensorflow.compat.v1 as tf
        from neural_compressor.experimental import Quantization, common
        tf.disable_v2_behavior()
        quantizer = Quantization('fake_diagnosis_yaml2.yaml')
        dataset = quantizer.dataset('dummy', shape=(128, 64, 64, 3), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.model
        quantizer.fit()
        self.assertEqual(os.path.exists(os.path.join(os.getcwd(), './save_path_test/fp32/inspect_result.pkl')), True)
        self.assertEqual(os.path.exists(os.path.join(os.getcwd(), './save_path_test/quan/inspect_result.pkl')), True)


if __name__ == '__main__':
    unittest.main()
