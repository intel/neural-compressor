import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.token_type_embeddings import TokenTypeEmbeddings
import numpy as np


class TestTokenTypeEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_token_type_embeddings_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        
        reshape_1_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='reshape_1:0', source_op=['reshape_1'], dest_op=['one_hot'])]
        reshape_1_node.construct('reshape_1', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        one_hot_node = OPERATORS['OneHot']()
        input_tensors = [Tensor(name='reshape_1:0', source_op=['reshape_1'], 
                                dest_op=['one_hot'])]
        output_tensors = [Tensor(name='one_hot:0', source_op=['one_hot'],
                                dest_op=['matmul'])]
        one_hot_node.construct('one_hot', 'OneHot', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({'test': 1}))
        
        matmul_node = OPERATORS['MatMul']()
        input_tensors = [Tensor(name='one_hot:0', source_op=['one_hot'], dest_op=['matmul']), 
                            Tensor(name='embeddings', shape=[2, 768], dest_op=['matmul'],
                            data=np.array(1))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'],
                                dest_op=['reshape_2'])]
        matmul_node.construct('matmul', 'MatMul', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'transpose_a': False, 'transpose_b': False}))
        

        shape_node = OPERATORS['Shape']()
        input_tensors = [Tensor(np.array(1))]
        output_tensors = [Tensor(name='shape:0', source_op=['shape'], 
                                    dest_op=['strided_slice'])]
        shape_node.construct('shape', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        strided_slice_node = OPERATORS['StridedSlice']()
        input_tensors = [Tensor(name='shape:0', source_op=['shape'], 
                                    dest_op=['strided_slice'])]
        output_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'], 
                                dest_op=['pack'])]
        strided_slice_node.construct('strided_slice', 'StridedSlice', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        pack_node = OPERATORS['Pack']()
        input_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'], 
                                dest_op=['pack'])]
        output_tensors = [Tensor(name='pack:0', source_op=['pack'], dest_op=['reshape_2'])]
        pack_node.construct('pack', 'Pack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reshape_2_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'], dest_op=['reshape_2']),
                            Tensor(name='pack:0', source_op=['pack'], dest_op=['reshape_2'])]
        output_tensors = [Tensor(name='reshape_2:0', source_op=['reshape_2'])]
        reshape_2_node.construct('reshape_2', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        
        graph.insert_nodes(len(graph.nodes), [input_data_node, reshape_1_node, one_hot_node,
                                                matmul_node, shape_node, strided_slice_node,
                                                pack_node, reshape_2_node])
        
        graph = TokenTypeEmbeddings()(graph)
        self.assertEqual(6, len(graph.nodes))
        self.assertEqual(-1, graph.nodes[1].attr['dst_shape'])
        self.assertEqual(1, graph.nodes[2].attr['test'])
        self.assertEqual('1,0', graph.nodes[3].attr['src1_perm'])
        self.assertEqual('-1,-1,768', graph.nodes[4].attr['dst_shape'])
        self.assertEqual('1,2', graph.nodes[5].attr['mul'])
    
    
    def test_token_type_embeddings_2(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(name='segment_ids', dest_op=['gather']), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        gather_node = OPERATORS['Gather']()
        input_tensors = [Tensor(shape=[2, 768], data=np.array(1)), Tensor(name='segment_ids', 
                                source_op=['input_data'], dest_op=['gather'])]
        output_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['add'])]
        gather_node.construct('gather', 'Gather', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'axis': 0, 'batch_dims': 0}))
        
        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='gather:0', source_op=['gather'], dest_op=['add']), Tensor(
                                data=np.array(1))]
        output_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=['layer_norm'])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        ln_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=['layer_norm']), 
                            Tensor(shape=[768], data=np.array(1)), 
                            Tensor(shape=[768], data=np.array(1))]
        output_tensors = [Tensor(name='layer_norm:0', source_op=['layer_norm'], dest_op=[])]
        ln_node.construct('layer_norm', 'LayerNorm', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, gather_node, add_node, ln_node])
        graph = TokenTypeEmbeddings()(graph)
        self.assertEqual(7, len(graph.nodes))
        self.assertEqual(-1, graph.nodes[1].attr['dst_shape'])
        self.assertEqual(0, graph.nodes[2].attr['axis'])
        self.assertEqual('-1,-1,768', graph.nodes[3].attr['dst_shape'])
        self.assertEqual('1,2', graph.nodes[4].attr['mul'])
        self.assertEqual('Add', graph.nodes[5].op_type)
        self.assertEqual('LayerNorm', graph.nodes[6].op_type)


if __name__ == "__main__":
    unittest.main()
