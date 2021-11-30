import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.matmul_with_bias_sigmoid import MatMulWithBiasSigmoid
import numpy as np


class TestMatMulWithBiasSigmoid(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_matmul_with_bias_relu_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        mat_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(data=np.array(1)), Tensor(data=np.array(1)), 
                            Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'], 
                                    dest_op=['sigmoid'])]
        mat_node.construct('matmul', 'MatMulWithBias', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'src1_perm': '1,0'}))
        
        tanh_node = OPERATORS['Sigmoid']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'], 
                                    dest_op=['sigmoid'])]
        output_tensors = [Tensor(name='sigmoid:0', source_op=['sigmoid'],
                                dest_op=[])]
        tanh_node.construct('sigmoid', 'Sigmoid', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, mat_node, tanh_node])
        graph = MatMulWithBiasSigmoid()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('1,0', graph.nodes[1].attr['src1_perm'])
        self.assertEqual('sigmoid', graph.nodes[1].name)
        self.assertEqual('sigmoid', graph.nodes[1].attr['append_op'])


if __name__ == "__main__":
    unittest.main()
