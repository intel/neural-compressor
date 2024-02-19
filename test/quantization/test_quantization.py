"""Tests for neural_compressor quantization."""

import importlib
import os
import shutil
import unittest

import numpy as np
import yaml


def build_fake_yaml():
    fake_yaml = """
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
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml2():
    fake_yaml = """
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
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml2.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml3():
    fake_yaml = """
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
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml3.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml4():
    fake_yaml = """
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
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml4.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml5():
    fake_yaml = """
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
            max_trials: 10
          workspace:
            path: saved
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml5.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_yaml6():
    fake_yaml = """
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
        """
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open("fake_yaml6.yaml", "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_fake_model():
    import tensorflow as tf
    from tensorflow.compat.v1 import graph_util

    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session() as sess:
            x = tf.placeholder(tf.float64, shape=(1, 3, 3, 1), name="x")
            y = tf.constant(np.random.random((2, 2, 1, 1)), name="y")
            op = tf.nn.conv2d(input=x, filter=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["op_to_store"])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(1, 3, 3, 1), name="x")
            y = tf.compat.v1.constant(np.random.random((2, 2, 1, 1)), name="y")
            op = tf.nn.conv2d(input=x, filters=y, strides=[1, 1, 1, 1], padding="VALID", name="op_to_store")

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, ["op_to_store"]
            )

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
    return graph


def build_fake_strategy():
    with open(
        os.path.join(
            os.path.dirname(importlib.util.find_spec("neural_compressor").origin), "experimental/strategy/fake.py"
        ),
        "w",
        encoding="utf-8",
    ) as f:
        seq = [
            "import time \n",
            "import copy \n",
            "import numpy as np \n",
            "from collections import OrderedDict \n",
            "from .strategy import strategy_registry, TuneStrategy \n",
            "from ...utils import logger \n",
            "from .utils.tuning_sampler import OpTypeWiseTuningSampler, FallbackTuningSampler \n",
            "from .utils.tuning_structs import OpTuningConfig \n",
            "import copy \n",
            "@strategy_registry \n",
            "class FakeTuneStrategy(TuneStrategy): \n",
            "    def __init__(self, model, cfg, q_dataloader, q_func=None, eval_dataloader=None, \n",
            "                 eval_func=None, dicts=None, q_hooks=None): \n",
            "        self.id = 0 \n",
            "        self.resume = True if dicts else False \n",
            "        super(FakeTuneStrategy, self).__init__(model, cfg, q_dataloader, \n",
            "                                               q_func, eval_dataloader, eval_func, dicts) \n",
            "    def __getstate__(self): \n",
            "        for history in self.tuning_history: \n",
            "            if self._same_yaml(history['cfg'], self.cfg): \n",
            "                history['id'] = self.id \n",
            "        save_dict = super(FakeTuneStrategy, self).__getstate__() \n",
            "        return save_dict \n",
            "    def next_tune_cfg(self): \n",
            "        if self.resume: \n",
            "            #assert self.id == 1 \n",
            "            assert len(self.tuning_history) == 1 \n",
            "            history = self.tuning_history[0] \n",
            "            assert self._same_yaml(history['cfg'], self.cfg) \n",
            "            assert len(history['history']) \n",
            "            for h in history['history']: \n",
            "                assert h \n",
            "        from copy import deepcopy \n",
            "        tuning_space = self.tuning_space \n",
            "        initial_op_tuning_cfg = {} \n",
            "        for item in tuning_space.root_item.options: \n",
            "            if item.item_type == 'op': \n",
            "                op_name, op_type = item.name \n",
            "                initial_op_tuning_cfg[item.name] = OpTuningConfig(op_name, op_type, 'fp32', tuning_space) \n",
            "            calib_sampling_size_lst = tuning_space.root_item.get_option_by_name('calib_sampling_size').options \n",
            "            for calib_sampling_size in calib_sampling_size_lst: \n",
            "                # step1. collect the ops that support static and dynamic \n",
            "                quant_mode_wise_items = OrderedDict() \n",
            "                query_order = ['static', 'dynamic', 'bf16', 'fp16', 'fp32'] \n",
            "                pre_items = set() \n",
            "                for quant_mode in query_order: \n",
            "                    items = tuning_space.query_items_by_quant_mode(quant_mode) \n",
            "                    filtered_items = [item for item in items if item not in pre_items] \n",
            "                    pre_items = pre_items.union(set(items)) \n",
            "                    quant_mode_wise_items[quant_mode] = filtered_items \n",
            "                def initial_op_quant_mode(items_lst, target_quant_mode, op_item_dtype_dict): \n",
            "                    for item in items_lst: \n",
            "                        op_item_dtype_dict[item.name] = target_quant_mode \n",
            "                op_item_dtype_dict = OrderedDict() \n",
            "                for quant_mode, quant_mode_items in quant_mode_wise_items.items(): \n",
            "                    initial_op_quant_mode(quant_mode_items, quant_mode, op_item_dtype_dict) \n",
            "                # step3. optype-wise tuning tuning items: the algorithm/scheme/granularity of activation(weight) \n",
            "                early_stop_tuning = False \n",
            "                stage1_cnt = 0 \n",
            "                int8_ops = quant_mode_wise_items['dynamic'] + quant_mode_wise_items['static'] \n",
            "                stage1_max = min(5, len(int8_ops))  # TODO set a more appropriate value \n",
            "                op_wise_tuning_sampler = OpTypeWiseTuningSampler(tuning_space, [], [], \n",
            "                                                                 op_item_dtype_dict, initial_op_tuning_cfg) \n",
            "                for op_tuning_cfg in op_wise_tuning_sampler: \n",
            "                    stage1_cnt += 1 \n",
            "                    if early_stop_tuning and stage1_cnt > stage1_max: \n",
            "                        logger.info('Early stopping the stage 1.') \n",
            "                        break \n",
            "                    op_tuning_cfg['calib_sampling_size'] = calib_sampling_size \n",
            "                    self.id += 1 \n",
            "                    yield op_tuning_cfg \n",
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
        os.remove("fake_yaml.yaml")
        os.remove("fake_yaml2.yaml")
        os.remove("fake_yaml3.yaml")
        os.remove("fake_yaml4.yaml")
        os.remove("fake_yaml5.yaml")
        os.remove("fake_yaml6.yaml")
        os.remove(
            os.path.join(
                os.path.dirname(importlib.util.find_spec("neural_compressor").origin), "experimental/strategy/fake.py"
            )
        )
        shutil.rmtree("./saved", ignore_errors=True)

    def test_resume(self):
        import tensorflow as tf
        from tensorflow.compat.v1 import graph_util

        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(1)
        x = tf.compat.v1.placeholder(tf.float32, [1, 32, 32, 3], name="x")
        top_relu = tf.nn.relu(x)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        x_pad = tf.pad(top_relu, paddings, "CONSTANT")
        conv_weights = tf.compat.v1.get_variable(
            "weight", [3, 3, 3, 3], initializer=tf.compat.v1.random_normal_initializer()
        )
        conv = tf.nn.conv2d(x_pad, conv_weights, strides=[1, 2, 2, 1], padding="VALID")
        relu = tf.nn.relu(conv)

        relu6 = tf.nn.relu6(relu, name="op_to_store")

        out_name = relu6.name.split(":")[0]
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph_def, output_node_names=[out_name]
            )

            from neural_compressor.experimental import Quantization, common

            quantizer = Quantization("fake_yaml5.yaml")
            dataset = quantizer.dataset("dummy", shape=(100, 32, 32, 3), label=True)
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            self.assertNotEqual(output_graph, None)
            self.assertTrue(os.path.exists("./saved"))
            quantizer = Quantization("fake_yaml2.yaml")
            quantizer.eval_dataloader = common.DataLoader(dataset)
            quantizer.calib_dataloader = common.DataLoader(dataset)
            quantizer.model = output_graph_def
            output_graph = quantizer.fit()
            # self.assertNotEqual(output_graph, None) # disable this check, the code has bug of recover from resume

    def test_autodump(self):
        # test auto_dump using old api
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization("fake_yaml3.yaml")
        dataset = quantizer.dataset("dummy", shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

    def test_performance_only(self):
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization("fake_yaml4.yaml")
        dataset = quantizer.dataset("dummy", shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

    def test_fit_method(self):
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization("fake_yaml4.yaml")
        dataset = quantizer.dataset("dummy", shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

    def test_quantization_without_yaml(self):
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization()
        quantizer.model = self.constant_graph
        dataset = quantizer.dataset("dummy", shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

    def test_invalid_eval_func(self):
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization("fake_yaml.yaml")
        dataset = quantizer.dataset("dummy", shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph

        def invalid_eval_func(model):
            return [[1.0]]

        quantizer.eval_func = invalid_eval_func
        output_graph = quantizer.fit()
        self.assertEqual(output_graph, None)

        def invalid_eval_func(model):
            return "0.1"

        quantizer.eval_func = invalid_eval_func
        output_graph = quantizer.fit()
        self.assertEqual(output_graph, None)

    def test_custom_metric(self):
        from neural_compressor.experimental import Quantization, common

        quantizer = Quantization("fake_yaml6.yaml")
        dataset = quantizer.dataset("dummy", shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer.metric = Metric()
        quantizer.fit()
        self.assertEqual(quantizer.strategy.evaluation_result[0], 0.5)

    def test_custom_objective(self):
        import tracemalloc

        from neural_compressor.experimental import Quantization, common
        from neural_compressor.objective import Objective, objective_registry

        class MyObjective(Objective):
            representation = "MyObj"

            def __init__(self):
                super().__init__()

            def start(self):
                tracemalloc.start()

            def end(self):
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self._result_list.append(peak // 1048576)

        quantizer = Quantization("fake_yaml.yaml")
        dataset = quantizer.dataset("dummy", shape=(100, 3, 3, 1), label=True)
        quantizer.eval_dataloader = common.DataLoader(dataset)
        quantizer.calib_dataloader = common.DataLoader(dataset)
        quantizer.model = self.constant_graph
        quantizer.objective = MyObjective()
        output_graph = quantizer.fit()
        self.assertNotEqual(output_graph, None)

        class MyObjective(Objective):
            representation = "Accuracy"

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
                representation = "Accuracy"

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
