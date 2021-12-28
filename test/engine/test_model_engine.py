import sys
import os
import unittest
import numpy as np
import shutil

from neural_compressor.model.engine_model import EngineModel
from engine.compile.ops.tensor import Tensor
from engine.compile.graph.graph import Graph
from engine.compile.ops.op import Operator
from neural_compressor.experimental import Quantization, Benchmark, common

def build_yaml():
    fake_yaml = """
        model:
          name: bert
          framework: engine
        quantization:
          calibration:
            sampling_size: 10
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
          op_wise: {
            'bert/encoder/layer_0/attention/self/query/BiasAdd':{
                'activation': {'dtype': ['fp32']},
                'weight': {'dtype': ['fp32'], 'granularity': ['per_tensor']}
            }
          }
        evaluation:
          accuracy:
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
            postprocess:
              transform:
                LabelShift: -1
            metric:
              MSE:
                compare_label: False
          performance:
            iteration: 10
            dataloader:
              dataset:
                dummy_v2:
                  input_shape: [[324], [324], [324]]
                  label_shape: [324,2]
                  low: [1, 0, 0, 0]
                  high: [128, 1, 1, 128]
                  dtype: [int32, int32, int32, float32]
        tuning:
          exit_policy:
            max_trials: 1
    """
    with open("test.yaml",  "w", encoding="utf-8") as f:
        f.write(fake_yaml)

class TestDeepengineModel(unittest.TestCase):

    def setUp(self):
        build_yaml()

    def test_model(self):
        model_dir = "/home/tensorflow/test-engine/bert_mlperf_2none.pb"
        if not os.path.exists(model_dir):
           print("The model dir is not not found, therefore test may not all round")
           return
        model = EngineModel(model_dir)
        self.assertEqual(model.framework(), 'engine')
        self.assertTrue(isinstance(model.graph, Graph))
        self.assertTrue(isinstance(model.model, str))
        self.assertTrue(isinstance(model.nodes[0], Operator))

        input_0 = np.random.randint(0,384,(1,32)).reshape(1,32)
        input_1 = np.random.randint(0,2,(1, 32)).reshape(1,32)
        input_2 = np.random.randint(0,2,(1, 32)).reshape(1,32)
        model.engine_init()
        out = model.inference([input_0, input_1, input_2])
        self.assertEqual(1, len(out))
        model.save('./ir')

        evaluator = Benchmark('test.yaml')
        evaluator.model = './ir'
        evaluator("performance")
        self.assertNotEqual(evaluator, None)

        model.model = model.model
        model.nodes = model.nodes
        tensor = model.dump_tensor([model.nodes[-1].input_tensors[0].name])
        self.assertTrue(model.nodes[-1].input_tensors[0].name \
             in tensor['model']['operator']['output_data']['input'])
        
        model.change_node_input_tensors('bert/encoder/Reshape_1', 0, tensor=Tensor(
            name='input_ids:0', source_op=['input_data']), mode='insert')
        self.assertEqual(4, \
            len(model.nodes[model.get_node_id('bert/encoder/Reshape_1')].input_tensors))

        num_nodes = len(model.nodes)
        node = model.nodes[0]
        model.remove_nodes([model.nodes[0].name])
        self.assertEqual(len(model.nodes), num_nodes-1)
        model.insert_nodes(0, [node])
        self.assertEqual(len(model.nodes), num_nodes)
        self.assertEqual(0, model.get_node_id(node.name))
        self.assertEqual(node, model.get_node_by_name(node.name))
        model.rename_node(node.name, 'test')
        self.assertEqual(model.nodes[0].name, 'test')
        self.assertEqual(model.get_pre_node_names(model.nodes[-1].name)[0],
            model.nodes[-2].name)
        self.assertEqual(model.get_next_node_names(model.nodes[-2].name)[0],
            model.nodes[-1].name)
        model.modify_node_connections(node, mode='insert')

    def tearDown(self):
        os.remove('test.yaml')
        shutil.rmtree('./ir')

if __name__ == "__main__":
    unittest.main()
