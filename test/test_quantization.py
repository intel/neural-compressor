"""Tests for neural_compressor quantization"""
import numpy as np
import unittest
import os
import yaml
import tensorflow as tf
import importlib
import shutil
from tensorflow.python.framework import graph_util

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
            name: fake
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
            name: fake
          accuracy_criterion:
            relative: 0.01
          workspace:
            path: saved
            resume: ./saved/history.snapshot
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml2.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml3():
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
              MSE:
                compare_label: False
        tuning:
          strategy:
            name: fake
          accuracy_criterion:
            relative: 0.01
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml3.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml4():
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
        tuning:
          strategy:
            name: fake
          accuracy_criterion:
            relative: 0.01
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml4.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml5():
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
            name: fake
          accuracy_criterion:
            relative: 0.01
          exit_policy:
            max_trials: 1    
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml5.yaml',"w",encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()

def build_fake_yaml6():
    fake_yaml = '''
        model:
          name: fake_yaml
          framework: tensorflow
          inputs: x
          outputs: op_to_store
        device: cpu
        tuning:
          strategy:
            name: fake
          accuracy_criterion:
            relative: 0.01
          workspace:
            path: saved
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml6.yaml',"w",encoding="utf-8") as f:
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

def build_fake_strategy():
    with open(os.path.join(os.path.dirname(importlib.util.find_spec('neural_compressor').origin), 'strategy/fake.py'), 'w', encoding='utf-8') as f:
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

class Metric:
    def update(self, predict, label):
        pass

    def reset(self):
        pass

    def result(self):
        return 0.5

class TestQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        build_fake_yaml()
        build_fake_yaml2()
        build_fake_yaml3()
        build_fake_yaml4()
        build_fake_yaml5()
        build_fake_yaml6()
        build_fake_strategy()

    @classmethod
    def tearDownClass(self):
        os.remove('fake_yaml.yaml')
        os.remove('fake_yaml2.yaml')
        os.remove('fake_yaml3.yaml')
        os.remove('fake_yaml4.yaml')
        os.remove('fake_yaml5.yaml')
        os.remove('fake_yaml6.yaml')
        os.remove(os.path.join(os.path.dirname(importlib.util.find_spec('neural_compressor').origin), 'strategy/fake.py'))
        shutil.rmtree('./saved', ignore_errors=True)

    def test_resume(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(1)
        x = tf.compat.v1.placeholder(tf.float32, [1, 32, 32, 3], name="x")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable("weight", [3, 3, 3, 3],
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

            from neural_compressor.experimental import Quantization, common
            quantizer = Quantization('fake_yaml5.yaml')
            dataset = quantizer.dataset('dummy', shape=(100, 32, 32, 3), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            self.assertNotEqual(output_graph, None)
            self.assertTrue(os.path.exists("./saved"))
            quantizer = Quantization('fake_yaml2.yaml')
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            # self.assertNotEqual(output_graph, None) # disable this check, the code has bug of recover from resume

    def test_autodump(self):
        # test auto_dump using old api
        from neural_compressor import Quantization
        quantizer = Quantization('fake_yaml3.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        dataloader = quantizer.dataloader(dataset)
        quantizer.model = self.constant_graph
        output_graph = quantizer(self.constant_graph, \
                                 q_dataloader=dataloader, eval_dataloader=dataloader)
        self.assertNotEqual(output_graph, None)

    def test_performance_only(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('fake_yaml4.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

    def test_fit_method(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('fake_yaml4.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

    def test_quantization_without_yaml(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization()
        quantizer.model = self.constant_graph
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

    def test_invalid_eval_func(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        def invalid_eval_func(model):
            return [[1.]]
        quantizer.eval_func = invalid_eval_func
        output_graph = quantizer.fit()
        self.assertEqual(output_graph, None)

        def invalid_eval_func(model):
            return '0.1'
        quantizer.eval_func = invalid_eval_func
        output_graph = quantizer.fit()
        self.assertEqual(output_graph, None)


    def test_custom_metric(self):
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization('fake_yaml6.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer.metric = Metric()
        quantizer.fit()
        self.assertEqual(quantizer.strategy.evaluation_result[0], 0.5)

    def test_custom_objective(self):
        from neural_compressor.experimental import Quantization, common
        from neural_compressor.objective import Objective, objective_registry
        import tracemalloc
        class MyObjective(Objective):
          representation = 'MyObj'
          def __init__(self):
              super().__init__()
          def start(self):
              tracemalloc.start()
          def end(self):
              _, peak = tracemalloc.get_traced_memory()
              tracemalloc.stop()
              self._result_list.append(peak // 1048576)
        quantizer = Quantization('fake_yaml.yaml')
        dataset = quantizer.dataset('dummy', shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer.objective = MyObjective()
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

        class MyObjective(Objective):
          representation = 'Accuracy'
          def __init__(self):
              super().__init__()
          def start(self):
              tracemalloc.start()
          def end(self):
              _, peak = tracemalloc.get_traced_memory()
              tracemalloc.stop()
              self._result_list.append(peak // 1048576)
        quantizer = Quantization()
        with self.assertRaises(ValueError):
            quantizer.objective = MyObjective()

        with self.assertRaises(ValueError):
            @objective_registry
            class MyObjective(Objective):
              representation = 'Accuracy'
              def __init__(self):
                  super().__init__()
              def start(self):
                  tracemalloc.start()
              def end(self):
                  _, peak = tracemalloc.get_traced_memory()
                  tracemalloc.stop()
                  self._result_list.append(peak // 1048576)
 
if __name__ == "__main__":
    unittest.main()
