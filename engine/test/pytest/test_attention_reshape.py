import unittest
from collections import OrderedDict
from engine.converter.ops.op import OPERATORS, Operator
from engine.converter.ops.tensor import Tensor
from engine.converter.graph import Graph
from engine.converter.sub_graph.attention_reshape import AttentionReshape
import numpy as np


class TestAttentionReshape(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_attention_reshape_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['matmul'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'dst_shape': '0,0,768'}))
        
        matmul_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=['matmul']), 
                            Tensor(data=np.array(1)), Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'],
                                dest_op=[])]
        matmul_node.construct('matmul', 'MatMulWithBias', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'src1_perm': '0,1'}))
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, reshape_node, matmul_node])
        graph = AttentionReshape()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('-1,768', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('0,1', graph.nodes[2].attr['src1_perm'])


if __name__ == "__main__":
    unittest.main()