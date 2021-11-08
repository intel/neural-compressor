import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.position_embeddings import PositionEmbeddings
import numpy as np


class TestPositionEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_position_embeddings_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        slice_node = OPERATORS['Slice']()
        input_tensors = [Tensor(shape=[512, 768], data=np.array(1)), Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='slice:0', source_op=['slice'], 
                                    dest_op=['reshape'])]
        slice_node.construct('slice', 'Slice', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='slice:0', source_op=['slice'], 
                                    dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=[])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        less_equal_node = OPERATORS['LessEqual']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='less_equal:0', source_op=['less_equal'],
                                dest_op=['all'])]
        less_equal_node.construct('less_equal', 'LessEqual', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        all_node = OPERATORS['All']()
        input_tensors = [Tensor(name='less_equal:0', source_op=['less_equal'],
                                dest_op=['all'])]
        output_tensors = [Tensor(name='all:0', source_op=['all'],
                                dest_op=['assert'])]
        all_node.construct('all', 'All', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        assert_node = OPERATORS['Assert']()
        input_tensors = [Tensor(name='all:0', source_op=['all'],
                                dest_op=['assert'])]
        output_tensors = [Tensor(name='assert:0', source_op=['assert'],
                                dest_op=[])]
        assert_node.construct('assert', 'Assert', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, slice_node, reshape_node,
                                                less_equal_node, all_node, assert_node])
        graph = PositionEmbeddings()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('1,-1,768', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('1,-1', graph.nodes[2].attr['dst_shape'])
        self.assertEqual('reshape', graph.nodes[2].name)


if __name__ == "__main__":
    unittest.main()
