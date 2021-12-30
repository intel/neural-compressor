import unittest
from engine.compile import compile
from engine.compile.ops.tensor import Tensor
import numpy as np
import os
import shutil


class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_main(self):
        model_dir = '/home/tensorflow/inc_ut/engine/bert_mlperf_2none.pb'
        if not os.path.exists(model_dir):
            print(
                "The model dir is not not found, therefore test may not all round"
            )
            return

        model = compile(model_dir)
        input_0 = np.random.randint(0, 384, (1, 32)).reshape(1, 32)
        input_1 = np.random.randint(0, 2, (1, 32)).reshape(1, 32)
        input_2 = np.random.randint(0, 2, (1, 32)).reshape(1, 32)
        # test of inference
        out = model.inference([input_0, input_1, input_2])
        self.assertEqual(1, len(out))

        # test of engine_init
        net = model.dump_tensor()
        model.engine_init(net)
        out = model.inference([input_0, input_1, input_2])
        self.assertEqual(380, len(out))

        # test of dump_tensor
        net = model.dump_tensor(["iasA"])
        model.engine_init(net)
        out = model.inference([input_0, input_1, input_2])
        self.assertEqual(74, len(out))

        output_dir = os.getcwd() + '/ir'
        model.save(output_dir)
        self.assertTrue(os.path.exists(output_dir))
        shutil.rmtree(output_dir)

        old_name_idx = model.get_node_id(
            'bert/encoder/layer_0/attention/self/query/BiasAdd')
        model.rename_node('bert/encoder/layer_0/attention/self/query/BiasAdd',
                          'test_name')
        new_name_idx = model.get_node_id('test_name')
        self.assertEqual(old_name_idx, new_name_idx)

        model.change_node_input_tensors('test_name',
                                        0,
                                        tensor=Tensor(name='input_ids:0',
                                                      source_op=['input_data'
                                                                 ]),
                                        mode='insert')
        self.assertEqual(4, len(model.nodes[new_name_idx].input_tensors))

        model.change_node_output_tensors('test_name',
                                         1,
                                         tensor=Tensor(
                                             name='input_ids:0',
                                             source_op=['input_data']),
                                         mode='insert')
        model.change_node_output_tensors('test_name',
                                         0,
                                         tensor=Tensor(
                                             name='input_ids:0',
                                             source_op=['input_data']),
                                         mode='modify')
        model.change_node_output_tensors('test_name', 1, mode='remove')
        self.assertEqual(1, len(model.nodes[new_name_idx].output_tensors))
        self.assertEqual('input_ids:0',
                         model.nodes[new_name_idx].output_tensors[0].name)

        pre_node_names = model.get_pre_node_names(
            'bert/encoder/layer_0/attention/self/Reshape')
        next_node_names = model.get_next_node_names('test_name')
        self.assertSequenceEqual(['test_name', 'input_data'], pre_node_names)
        self.assertSequenceEqual([], next_node_names)


if __name__ == "__main__":
    unittest.main()
