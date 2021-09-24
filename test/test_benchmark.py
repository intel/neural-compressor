"""Tests for neural_compressor benchmark"""
import unittest
import os
import yaml
import numpy as np
import tensorflow as tf
import tempfile
import re
from neural_compressor.adaptor.tf_utils.util import write_graph

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
          performance:
            warmup: 5
            iteration: 20
            configs:
               cores_per_instance: 4
               num_of_instance: 2 
        tuning:
          accuracy_criterion:
            relative: 0.01
        '''
    y = yaml.load(fake_yaml, Loader=yaml.SafeLoader)
    with open('fake_yaml.yaml', "w", encoding="utf-8") as f:
        yaml.dump(y, f)
    f.close()


def build_benchmark():
    seq = '''
from argparse import ArgumentParser
arg_parser = ArgumentParser(description='Parse args')
arg_parser.add_argument('--input_model', dest='input_model', default='input_model', help='input odel')
args = arg_parser.parse_args()
from neural_compressor.data import DATASETS
dataset = DATASETS('tensorflow')['dummy']((100, 256, 256, 1), label=True)
from neural_compressor.experimental import Benchmark, common
from neural_compressor.conf.config import Benchmark_Conf
benchmarker = Benchmark('fake_yaml.yaml')
benchmarker.b_dataloader = common.DataLoader(dataset, batch_size=10)
benchmarker.model = args.input_model
benchmarker()
conf = Benchmark_Conf('fake_yaml.yaml')
benchmarker = Benchmark(conf)
benchmarker.b_dataloader = common.DataLoader(dataset, batch_size=10)
benchmarker.model = args.input_model
benchmarker()
    '''
    # test normal case
    with open('fake.py', "w", encoding="utf-8") as f:
        f.writelines(seq)
    f.close()
    # test batchsize > len(dataset), use first batch
    fake_data_5 = seq.replace('100, 256, 256, 1', '5, 256, 256, 1')
    with open('fake_data_5.py', "w", encoding="utf-8") as f:
        f.writelines(fake_data_5)
    f.close()
    # test batchsize < len(dataset) < 2*batchsize, discard first batch
    fake_data_15 = seq.replace('100, 256, 256, 1', '15, 256, 256, 1')
    with open('fake_data_15.py', "w", encoding="utf-8") as f:
        f.writelines(fake_data_15)
    f.close()
    # test 2*batchsize < len(dataset) < warmup*batchsize, discard last batch
    fake_data_25 = seq.replace('100, 256, 256, 1', '25, 256, 256, 1')
    with open('fake_data_25.py', "w", encoding="utf-8") as f:
        f.writelines(fake_data_25)
    f.close()

def build_benchmark_without_yaml():
    seq = [
        "from argparse import ArgumentParser\n",
        "arg_parser = ArgumentParser(description='Parse args')\n",
        "arg_parser.add_argument('--input_model', dest='input_model', default='input_model', help='input model')\n",
        "args = arg_parser.parse_args()\n",

        "from neural_compressor.data import DATASETS\n",
        "dataset = DATASETS('tensorflow')['dummy']((100, 256, 256, 1), label=True)\n",

        "from neural_compressor.experimental import Benchmark, common\n",
        "benchmarker = Benchmark()\n",
        "benchmarker.model = args.input_model\n",
        "benchmarker.b_dataloader = common.DataLoader(dataset)\n",
        "benchmarker()\n"
    ]

    with open('fake2.py', "w", encoding="utf-8") as f:
        f.writelines(seq)

def build_fake_model():
    graph_path = tempfile.mkstemp(suffix='.pb')[1]
    try:
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.Session(graph=graph) as sess:
            x = tf.placeholder(tf.float64, shape=(None, 256, 256, 1), name='x')
            y_1 = tf.constant(np.random.random((3, 3, 1, 1)), name='y_1')
            y_2 = tf.constant(np.random.random((3, 3, 1, 1)), name='y_2')
            conv1 = tf.nn.conv2d(input=x, filter=y_1, strides=[1, 1, 1, 1], \
                                 padding='VALID', name='conv1')
            op = tf.nn.conv2d(input=conv1, filter=y_2, strides=[1, 1, 1, 1], \
                              padding='VALID', name='op_to_store')

            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        write_graph(graph_def, graph_path)
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session(graph=graph) as sess:
            x = tf.compat.v1.placeholder(tf.float64, shape=(None, 256, 256, 1), name='x')
            y_1 = tf.constant(np.random.random((3, 3, 1, 1)), name='y_1')
            y_2 = tf.constant(np.random.random((3, 3, 1, 1)), name='y_2')
            conv1 = tf.nn.conv2d(input=x, filters=y_1, strides=[1, 1, 1, 1], \
                                 padding='VALID', name='conv1')
            op = tf.nn.conv2d(input=conv1, filters=y_2, strides=[1, 1, 1, 1], \
                              padding='VALID', name='op_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        write_graph(graph_def, graph_path)
    return graph_path


class TestObjective(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.graph_path = build_fake_model()
        build_fake_yaml()
        build_benchmark()
        build_benchmark_without_yaml()

    @classmethod
    def tearDownClass(self):
        if os.path.exists('fake_yaml.yaml'):
            os.remove('fake_yaml.yaml')
        if os.path.exists('fake.py'):
            os.remove('fake.py')
        if os.path.exists('fake_data_5.py'):
            os.remove('fake_data_5.py')
        if os.path.exists('fake_data_15.py'):
            os.remove('fake_data_15.py')
        if os.path.exists('fake_data_25.py'):
            os.remove('fake_data_25.py')
        if os.path.exists('2_4_0.log'):
            os.remove('2_4_0.log')
        if os.path.exists('2_4_1.log'):
            os.remove('2_4_1.log')

    def test_benchmark(self):
        os.system("python fake.py --input_model={}".format(self.graph_path))
        for i in range(2):
            with open(f'2_4_{i}.log', "r") as f:
                for line in f:
                    throughput = re.search(r"Throughput:\s+(\d+(\.\d+)?) images/sec", line)
            self.assertIsNotNone(throughput)

    def test_benchmark_data_5(self):
        os.system("python fake_data_5.py --input_model={}".format(self.graph_path))
        for i in range(2):
            with open(f'2_4_{i}.log', "r") as f:
                for line in f:
                    throughput = re.search(r"Throughput:\s+(\d+(\.\d+)?) images/sec", line)
            self.assertIsNotNone(throughput)

    def test_benchmark_data_15(self):
        os.system("python fake_data_15.py --input_model={}".format(self.graph_path))
        for i in range(2):
            with open(f'2_4_{i}.log', "r") as f:
                for line in f:
                    throughput = re.search(r"Throughput:\s+(\d+(\.\d+)?) images/sec", line)
            self.assertIsNotNone(throughput)


    def test_benchmark_data_25(self):
        os.system("python fake_data_25.py --input_model={}".format(self.graph_path))
        for i in range(2):
            with open(f'2_4_{i}.log', "r") as f:
                for line in f:
                    throughput = re.search(r"Throughput:\s+(\d+(\.\d+)?) images/sec", line)
            self.assertIsNotNone(throughput)

    def test_benchmark_without_yaml(self):
        from neural_compressor.utils import logger
        os.system("python fake2.py --input_model={}".format(self.graph_path))
        for i in range(2):
            with open(f'2_4_{i}.log', "r") as f:
                for line in f:
                    throughput = re.search(r"Throughput:\s+(\d+(\.\d+)?) images/sec", line)
            self.assertIsNotNone(throughput)

if __name__ == "__main__":
    unittest.main()
