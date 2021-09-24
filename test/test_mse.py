"""Tests for quantization"""
import numpy as np
import unittest
import os
import shutil
import yaml
import tensorflow as tf
import torch
import torchvision

def build_fake_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: x
          outputs: op_to_store
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
            strategy:
              name: mse
            accuracy_criterion:
              relative: 0.01
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml2():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: x
          outputs: op_to_store
        device: cpu
        evaluation:
          accuracy:
            metric:
              topk: 1
        tuning:
          strategy:
            name: mse
          exit_policy:
            max_trials: 5
          accuracy_criterion:
            relative: -0.01
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml2.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_ox_yaml():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: onnxrt_qlinearops
          inputs: input
          outputs: output
        evaluation:
          accuracy:
            metric:
              Accuracy: {}
        tuning:
            strategy:
              name: mse
            accuracy_criterion:
              relative: -0.01
              higher_is_better: False
            exit_policy:
              max_trials: 3
            workspace:
              path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('ox_yaml.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float64, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)), name='y')
            op = tf.nn.conv2d(input=x, filter=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1,3,3,1), name='x')
            y = tf.compat.v1.constant(np.random.random((2,2,1,1)), name='y')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

def build_ox_model():
    path = "mb_v2.onnx"
    model = torchvision.models.mobilenet_v2()

    x = torch.randn(100, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    torch.onnx.export(model,
                      x,
                      path,
                      export_params=True,
                      opset_version=12,
                      do_constant_folding=True,
                      input_names = ["input"],
                      output_names = ["output"],
                      dynamic_axes={"input" : {0 : "batch_size"},
                                    "output" : {0 : "batch_size"}})

class dataset:
    def __init__(self):
        self.data = []
        self.label = []
        for i in range(10):
            self.data.append(np.zeros((3, 224, 224)).astype(np.float32))
            self.label.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

class TestQuantization(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        build_fake_yaml()
        build_fake_yaml2()
        build_ox_model()
        build_ox_yaml()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml2.yaml')
        os.remove('ox_yaml.yaml')
        os.remove('mb_v2.onnx')

        shutil.rmtree("saved", ignore_errors=True)

    def test_ru_mse_one_trial(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer()

    def test_ru_mse_max_trials(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('fake_yaml2.yaml')
        dataset = quantizer.dataset('dummy', (100, 3, 3, 1), label=True)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer()

    def test_ox_mse(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('ox_yaml.yaml')
        ds = dataset()
        quantizer.calib_dataloader = common.DataLoader(ds)
        quantizer.eval_dataloader = common.DataLoader(ds)
        quantizer.model = common.Model('mb_v2.onnx')
        quantizer()

if __name__ == "__main__":
    unittest.main()
