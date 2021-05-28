"""Tests for lpot quantization"""
import unittest
import os
import importlib
import shutil
import yaml
import numpy as np
import tensorflow as tf

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
          objective: performance
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
    with open(os.path.join(os.path.dirname(importlib.util.find_spec('lpot').origin), \
              'strategy/fake.py'), 'w', encoding='utf-8') as f:
        seq = [
            "import time\n",
            "from .strategy import strategy_registry, TuneStrategy\n",
            "from collections import OrderedDict\n",
            "import copy\n",
            "@strategy_registry\n",
            "class FakeTuneStrategy(TuneStrategy):\n",
            "  def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, eval_func=None, dicts=None):\n",
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
        os.remove(os.path.join(os.path.dirname(importlib.util.find_spec('lpot').origin), 'strategy/fake.py'))
        shutil.rmtree('./saved', ignore_errors=True)

    def test_performance(self):
        from lpot.data import DATASETS
        dataset = DATASETS('tensorflow')['dummy']((100, 256, 256, 1), label=True)

        from lpot.experimental import Quantization, common
        from lpot.utils.utility import get_size

        quantizer = Quantization('fake_yaml.yaml')
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        q_model = quantizer()

        from lpot.experimental import Benchmark, common
        benchmarker = Benchmark('fake_yaml.yaml')
        benchmarker.b_dataloader = common.DataLoader(dataset)
        benchmarker.model = self.constant_graph_1
        benchmarker()

    def test_model_size(self):
        from lpot.experimental import Benchmark, common
        from lpot.data import DATASETS
        dataset = DATASETS('tensorflow')['dummy']((100, 256, 256, 1), label=True)

        benchmarker = Benchmark('fake_yaml_model_size.yaml')
        benchmarker.b_dataloader = common.DataLoader(dataset)
        benchmarker.model = self.constant_graph_1
        benchmarker(mode='accuracy')

    def test_footprint(self):
        from lpot.experimental import Benchmark, common
        from lpot.data import DATASETS
        dataset = DATASETS('tensorflow')['dummy']((100, 256, 256, 1), label=True)

        benchmarker = Benchmark('fake_yaml_footprint.yaml')
        benchmarker.b_dataloader = common.DataLoader(dataset)
        benchmarker.model = self.constant_graph_1
        benchmarker()


if __name__ == "__main__":
    unittest.main()
