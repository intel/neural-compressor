import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.qkv_reshape import QKVReshape


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

        matmul_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(), Tensor(), Tensor()]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'],
                                dest_op=['reshape'])]
        matmul_node.construct('matmul', 'MatMulWithBias', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'src1_perm': '0,1'}))
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'],
                                dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], dest_op=[])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'dst_shape': '0,0,12,64'}))
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, matmul_node, reshape_node])
        graph = QKVReshape()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('0,1', graph.nodes[1].attr['src1_perm'])
        self.assertEqual('-1,-1,12,64', graph.nodes[2].attr['dst_shape'])


if __name__ == "__main__":
    unittest.main()
