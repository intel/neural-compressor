import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.transpose_batch_matmul import TransposeBatchMatMul
import numpy as np


class TestTransposeBatchMatMul(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_transpose_batch_matmul_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        transpose_1_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='transpose_1:0', source_op=['transpose_1'],
                                dest_op=['fused_matmul'])]
        transpose_1_node.construct('transpose_1', 'Transpose', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'dst_perm': '0,2,1,3'}))
        
        transpose_2_node = OPERATORS['Transpose']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='transpose_2:0', source_op=['transpose_2'],
                                dest_op=['fused_matmul'])]
        transpose_2_node.construct('transpose_2', 'Transpose', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'dst_perm': '0,2,3,1'}))

        fused_matmul_node = OPERATORS['FusedMatMul']()
        input_tensors = [Tensor(name='transpose_1:0', source_op=['transpose_1'],
                                dest_op=['fused_matmul']), Tensor(name='transpose_2:0', 
                                source_op=['transpose_2'], dest_op=['fused_matmul'])]
        output_tensors = [Tensor(name='fused_matmul:0', source_op=['fused_matmul'],
                                dest_op=['add'])]
        fused_matmul_node.construct('fused_matmul', 'FusedMatMul', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'transpose_a': False, 'transpose_b': False, 'alpha': 0.125}))

        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='fused_matmul:0', source_op=['fused_matmul'],
                                dest_op=['add']), Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=[])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, transpose_1_node, transpose_2_node,
                                                    fused_matmul_node, add_node])
        graph = TransposeBatchMatMul()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('0,2,1,3', graph.nodes[1].attr['src0_perm'])
        self.assertEqual('0,2,3,1', graph.nodes[1].attr['src1_perm'])
        self.assertEqual(0.125, graph.nodes[1].attr['output_scale'])
        self.assertEqual('binary_add', graph.nodes[1].attr['append_op'])


if __name__ == "__main__":
    unittest.main()
