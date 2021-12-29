"""Tests for neural_compressor quantization"""
import unittest
import os
import importlib
import shutil
import yaml
import numpy as np
import tensorflow as tf
import onnx
from onnx import helper, TensorProto
import random

def build_fake_yaml_footprint():
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
          performance: {}
        tuning:
          multi_objective:
            objective: footprint
          strategy:
            name: fake
          accuracy_criterion:
            relative: 0.01
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_footprint.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

def build_fake_yaml_model_size():
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
          performance: {}
        tuning:
          multi_objective:
            objective: modelsize
          strategy:
            name: fake
          accuracy_criterion:
            relative: 0.01
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml_model_size.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

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
          performance: {}
        tuning:
          strategy:
            name: fake
          accuracy_criterion:
            relative: 0.01
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()

def build_fake_model():
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session(graph=graph) as sess:
            x = tf.placeholder(tf.float64, shape=(1, 256, 256, 1), name='x')
            y = tf.constant(np.random.random((2, 2, 1, 1)), name='y')
            op = tf.nn.conv2d(input=x, filter=y, strides=[1, 1, 1, 1], \
                              padding='VALID', name='op_to_store')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session(graph=graph) as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1, 256, 256, 1), name='x')
            y = tf.compat.v1.constant(np.random.random((3, 3, 1, 1)), name='y')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], \
                              padding='VALID', name='op_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

def build_fake_model1():
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session(graph=graph) as sess:
            x = tf.placeholder(tf.float64, shape=(1, 256, 256, 1), name='x')
            y_1 = tf.constant(np.random.random((3, 3, 1, 1)), name='y_1')
            y_2 = tf.constant(np.random.random((3, 3, 1, 1)), name='y_2')
            conv1 = tf.nn.conv2d(input=x, filter=y_1, strides=[1, 1, 1, 1], \
                                 padding='VALID', name='conv1')
            op = tf.nn.conv2d(input=conv1, filter=y_2, strides=[1, 1, 1, 1], \
                              padding='VALID', name='op_to_store')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session(graph=graph) as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1, 256, 256, 1), name='x')
            y_1 = tf.constant(np.random.random((3, 3, 1, 1)), name='y_1')
            y_2 = tf.constant(np.random.random((3, 3, 1, 1)), name='y_2')
            conv1 = tf.nn.conv2d(input=x, filters=y_1, strides=[1, 1, 1, 1], \
                                 padding='VALID', name='conv1')
            op = tf.nn.conv2d(input=conv1, filters=y_2, strides=[1, 1, 1, 1], \
                              padding='VALID', name='op_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph

def build_fake_strategy():
    with open(os.path.join(os.path.dirname(importlib.util.find_spec('neural_compressor').origin), \
              'strategy/fake.py'), 'w', encoding='utf-8') as f:
        seq = [
            "import time\n",
            "from .strategy import strategy_registry, TuneStrategy\n",
            "from collections import OrderedDict\n",
            "import copy\n",
            "@strategy_registry\n",
            "class FakeTuneStrategy(TuneStrategy):\n",
            "  def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None, q_hooks=None):\n",
            "    self.id = 0\n",
            "    self.resume = True if dicts else False\n",
            "    super(FakeTuneStrategy, self).__init__(model, cfg, q_dataloader, q_func, eval_dataloader, eval_func, dicts)\n",
            "  def __getstate__(self):\n",
            "    for history in self.tuning_history:\n",
            "      if self._same_yaml(history['cfg'], self.cfg):\n",
            "        history['id'] = self.id\n",
            "    save_dict = super(FakeTuneStrategy, self).__getstate__()\n",
            "    return save_dict\n",
            "  def next_tune_cfg(self):\n",
            "    if self.resume:\n",
            "      assert self.id == 1\n",
            "      assert len(self.tuning_history) == 1\n",
            "      history = self.tuning_history[0]\n",
            "      assert self._same_yaml(history['cfg'], self.cfg)\n",
            "      assert len(history['history'])\n",
            "      for h in history['history']:\n",
            "        assert h\n",
            "    op_cfgs = {}\n",
            "    for iterations in self.calib_iter:\n",
            "      op_cfgs['calib_iteration'] = int(iterations)\n",
            "      op_cfgs['op'] = OrderedDict()\n",
            "      for op in self.opwise_quant_cfgs:\n",
            "        op_cfgs['op'][op] = copy.deepcopy(\n",
            "                                self.opwise_tune_cfgs[op][0])\n",
            "      self.id += 1\n",
            "      yield op_cfgs\n",
            "      return\n"
        ]
        f.writelines(seq)
    f.close()

class TestObjective(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        self.constant_graph_1 = build_fake_model1()
        build_fake_yaml()
        build_fake_yaml_footprint()
        build_fake_yaml_model_size()
        build_fake_strategy()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml_model_size.yaml')
        os.remove('fake_yaml_footprint.yaml')
        os.remove(os.path.join(os.path.dirname(importlib.util.find_spec('neural_compressor').origin), 'strategy/fake.py'))
        shutil.rmtree('./saved', ignore_errors=True)

    def test_performance(self):
        from neural_compressor.data import DATASETS
        dataset = DATASETS('tensorflow')['dummy']((100, 256, 256, 1), label=True)

        from neural_compressor.experimental import Quantization, common
        from neural_compressor.utils.utility import get_size

        quantizer = Quantization('fake_yaml.yaml')
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        q_model = quantizer.fit()

        from neural_compressor.experimental import Benchmark, common
        benchmarker = Benchmark('fake_yaml.yaml')
        benchmarker.b_dataloader = common.DataLoader(dataset)
        benchmarker.model = self.constant_graph_1
        benchmarker.fit()

    def test_model_size(self):
        from neural_compressor.experimental import Benchmark, common
        from neural_compressor.data import DATASETS
        dataset = DATASETS('tensorflow')['dummy']((100, 256, 256, 1), label=True)

        benchmarker = Benchmark('fake_yaml_model_size.yaml')
        benchmarker.b_dataloader = common.DataLoader(dataset)
        benchmarker.model = self.constant_graph_1
        benchmarker(mode='accuracy')

    def test_footprint(self):
        from neural_compressor.experimental import Benchmark, common
        from neural_compressor.data import DATASETS
        dataset = DATASETS('tensorflow')['dummy']((100, 256, 256, 1), label=True)

        benchmarker = Benchmark('fake_yaml_footprint.yaml')
        benchmarker.b_dataloader = common.DataLoader(dataset)
        benchmarker.model = self.constant_graph_1
        benchmarker.fit()

def build_matmul_model():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 5, 1])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 1])
    matmul_node = onnx.helper.make_node('MatMul', ['A', 'B'], ['C'], name='Matmul')
    graph = helper.make_graph([matmul_node], 'test_graph_1', [A, B], [C])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    return model

class TestObjs(unittest.TestCase):

    def test_model(self):
        def eval(model):
            return random.random()
        
        model = build_matmul_model()
        
        from neural_compressor import conf
        from neural_compressor.experimental import Quantization
        
        conf.model.framework = 'onnxrt_integerops'
        conf.quantization.approach = 'post_training_dynamic_quant'
        conf.tuning.accuracy_criterion.absolute = 0.3
        conf.tuning.multi_objective.objective = ['accuracy', 'performance']
        conf.tuning.multi_objective.weight = [0.8, 0.2]
        conf.tuning.exit_policy.timeout = 10000
        conf.tuning.exit_policy.max_trials = 2
        quantize = Quantization(conf)
        quantize.model = model
        quantize.eval_func = eval
        q_model = quantize()

    def test_tune_data(self):
        from neural_compressor.objective import MultiObjective
        obj = MultiObjective(['accuracy', 'modelsize', 'performance'],
                             {'relative': 0.1},
                             [0.7, 0.2, 0.1])
        baseline = [0.8, [0.8, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 420, 0.24]],
            [0.750, [0.750, 430, 0.22]],
            [0.720, [0.720, 410, 0.18]],
            [0.790, [0.790, 360, 0.15]], 
            [0.750, [0.750, 430, 0.24]],
            [0.785, [0.785, 360, 0.13]]]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 4)

        obj = MultiObjective(['accuracy', 'modelsize', 'performance'],
                             {'relative': 0.1})
        baseline = [0.8, [0.8, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 420, 0.24]],
            [0.750, [0.750, 430, 0.22]],
            [0.720, [0.720, 410, 0.18]],
            [0.790, [0.790, 360, 0.15]], 
            [0.750, [0.750, 430, 0.24]],
            [0.785, [0.785, 360, 0.13]]]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(['accuracy', 'modelsize', 'performance'],
                             {'absolute': 0.3})
        baseline = [0.8, [0.8, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 420, 0.24]],
            [0.750, [0.750, 430, 0.22]],
            [0.720, [0.720, 410, 0.18]],
            [0.790, [0.790, 360, 0.15]], 
            [0.750, [0.750, 430, 0.24]],
            [0.785, [0.785, 360, 0.13]]]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(['accuracy', 'modelsize', 'performance'],
                             {'absolute': 0.3},
                             [0.6, 0.1, 0.3])
        baseline = [0.8, [0.8, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 400, 0.24]],
            [0.750, [0.750, 400, 0.22]],
            [0.720, [0.720, 400, 0.18]],
            [0.790, [0.790, 400, 0.15]], 
            [0.750, [0.750, 400, 0.24]],
            [0.785, [0.785, 400, 0.13]]]
        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

        obj = MultiObjective(['accuracy', 'modelsize', 'performance'],
                             {'absolute': 0.04, 'higher_is_better': False},
                             [0.6, 0.1, 0.3])
        baseline = [0.75, [0.75, 780, 0.6]]
        tune_data = [
            [0.760, [0.760, 400, 0.23]],
            [0.778, [0.778, 400, 0.10]],
            [0.750, [0.750, 400, 0.22]],
            [0.720, [0.720, 400, 0.18]],
            [0.790, [0.790, 400, 0.15]], 
            [0.750, [0.750, 400, 0.24]],
            [0.785, [0.785, 400, 0.13]]]
        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 3)

        obj = MultiObjective(['accuracy', 'modelsize', 'performance'],
                             {'absolute': 0.4, 'higher_is_better': False},
                             [0.6, 0.1, 0.3])
        baseline = [0.0, [0.0, 780, 0.6]]
        tune_data = [
            [0.00, [0.00, 400, 0.23]],
            [0.80, [0.80, 400, 0.10]],
            [0.02, [0.02, 400, 0.22]],
            [0.10, [0.10, 400, 0.18]],
            [0.20, [0.20, 400, 0.15]], 
            [0.00, [0.00, 400, 0.24]],
            [0.50, [0.50, 400, 0.13]]]
        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 0)
 
        obj = MultiObjective(['modelsize', 'performance'],
                             {'relative': 0.08},
                             [0.2, 0.8])
        baseline = [0.8, [780, 0.6]]
        tune_data = [
            [0.760, [400, 0.23]],
            [0.778, [420, 0.24]],
            [0.750, [430, 0.22]],
            [0.720, [410, 0.18]],
            [0.790, [360, 0.15]], 
            [0.750, [430, 0.24]],
            [0.785, [360, 0.13]]]

        num, _ = obj.best_result(tune_data, baseline)
        self.assertEqual(num, 6)

if __name__ == "__main__":
    unittest.main()
