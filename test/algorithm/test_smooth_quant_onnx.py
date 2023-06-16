import onnx
from onnx import helper, TensorProto, numpy_helper
import copy
import unittest
import numpy as np
import shutil
from neural_compressor.data import Datasets, DATALOADERS
from neural_compressor.adaptor.ox_utils.smooth_quant import ORTSmoothQuant


def build_onnx_model():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 5, 2])
    H = helper.make_tensor_value_info('H', TensorProto.FLOAT, [1, 5, 2])

    g_value = np.random.uniform(low=0.001, high=0.5, size=(25)).astype(np.float32)
    G_init = helper.make_tensor('G', TensorProto.FLOAT, [5, 5], g_value.reshape(25).tolist())
    matmul_node = onnx.helper.make_node('MatMul', ['A', 'G'], ['C'], name='Matmul')

    b_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    B_init = helper.make_tensor('B', TensorProto.FLOAT, [5, 2], b_value.reshape(10).tolist())
    matmul_node2 = onnx.helper.make_node('MatMul', ['C', 'B'], ['I'], name='Matmul2')

    e_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    E_init = helper.make_tensor('E', TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    matmul_node3 = onnx.helper.make_node('MatMul', ['C', 'E'], ['K'], name='Matmul3')

    add = onnx.helper.make_node('Add', ['I', 'E'], ['D'], name='add')

    f_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    F_init = helper.make_tensor('F', TensorProto.FLOAT, [5, 2], f_value.reshape(10).tolist())
    add2 = onnx.helper.make_node('Add', ['D', 'F'], ['H'], name='add2')

    graph = helper.make_graph([matmul_node, matmul_node2, matmul_node3, add, add2], 'test_graph_1', [A], [H], [B_init, E_init, F_init, G_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    return model


class TestORTSq(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = build_onnx_model()
        dataset = Datasets("onnxrt_qdq")["dummy_v2"]((5,5), (5,1))
        self.dataloader = DATALOADERS['onnxrt_qlinearops'](dataset)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_sq(self):
        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, scales_per_op=False)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 1)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, folding=False, scales_per_op=False)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 2)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, folding=False, scales_per_op=True)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, scales_per_op=True)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, scales_per_op=True, alpha='auto')
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)


        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, alpha='auto', scales_per_op=False)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 1)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)


if __name__ == '__main__':
    unittest.main()
