import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.padding_sequence import PaddingSequence
import numpy as np

class TestPaddingSequence(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_padding_sequence_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        shape_node = OPERATORS['Shape']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='shape:0', source_op=['shape'],
                                dest_op=['strided_slice'])]
        shape_node.construct('shape', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        strided_slice_node = OPERATORS['StridedSlice']()
        input_tensors = [Tensor(name='shape:0', source_op=['shape'],
                                dest_op=['strided_slice'])]
        output_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'],
                                dest_op=['pack_0', 'pack_1'])]
        strided_slice_node.construct('strided_slice', 'StridedSlice', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        pack_0_node = OPERATORS['Pack']()
        input_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'],
                                dest_op=['pack_0'])]
        output_tensors = [Tensor(name='pack_0:0', source_op=['pack_0'],
                                dest_op=['fill'])]
        pack_0_node.construct('pack_0', 'Pack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        fill_node = OPERATORS['Fill']()
        input_tensors = [Tensor(name='pack_0:0', source_op=['pack_0'],
                                dest_op=['fill'])]
        output_tensors = [Tensor(name='fill:0', source_op=['fill'],
                                dest_op=['mul_1'])]
        fill_node.construct('fill', 'Fill', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        pack_1_node = OPERATORS['Pack']()
        input_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'],
                                dest_op=['pack_1'])]
        output_tensors = [Tensor(name='pack_1:0', source_op=['pack_1'],
                                dest_op=['reshape'])]
        pack_1_node.construct('pack_1', 'Pack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='pack_1:0', source_op=['pack_1'],
                                dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=['cast'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=['cast'])]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['mul_1'])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mul_1_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='fill:0', source_op=['fill'],
                                dest_op=['mul_1']), Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['mul_1'])]
        output_tensors = [Tensor(name='mul_1:0', source_op=['mul_1'],
                                dest_op=['expand_dims'])]
        mul_1_node.construct('mul_1', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        expand_dims_node = OPERATORS['ExpandDims']()
        input_tensors = [Tensor(name='mul_1:0', source_op=['mul_1'],
                                dest_op=['expand_dims'])]
        output_tensors = [Tensor(name='expand_dims:0', source_op=['expand_dims'],
                                dest_op=['sub'])]
        expand_dims_node.construct('expand_dims', 'ExpandDims', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        sub_node = OPERATORS['Sub']()
        input_tensors = [Tensor(name='expand_dims:0', source_op=['expand_dims'],
                                dest_op=['sub'])]
        output_tensors = [Tensor(name='sub:0', source_op=['sub'],
                                dest_op=['mul_2'])]
        sub_node.construct('sub', 'Sub', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        mul_2_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='sub:0', source_op=['sub'],
                                dest_op=['mul_2'])]
        output_tensors = [Tensor(name='mul_2:0', source_op=['mul_2'],
                                dest_op=['add'])]
        mul_2_node.construct('mul_2', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)                        
        
        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='mul_2:0', source_op=['mul_2'],
                                dest_op=['add'])]
        output_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=[])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)   
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, shape_node, strided_slice_node,
                                                pack_0_node, fill_node, pack_1_node, reshape_node,
                                                cast_node, mul_1_node, expand_dims_node, sub_node,
                                                mul_2_node, add_node])
        graph = PaddingSequence()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('-1,12,0,-1', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('AddV2', graph.nodes[2].op_type)


    def test_padding_sequence_2(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        unsqueeze_1_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'],
                                dest_op=['unsqueeze_2'])]
        unsqueeze_1_node.construct('unsqueeze_1', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unsqueeze_2_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'],
                                dest_op=['unsqueeze_2'])]
        output_tensors = [Tensor(name='unsqueeze_2:0', source_op=['unsqueeze_2'],
                                dest_op=['cast'])]
        unsqueeze_2_node.construct('unsqueeze_2', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='unsqueeze_2:0', source_op=['unsqueeze_2'],
                                dest_op=['cast'])]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['sub'])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        sub_node = OPERATORS['Sub']()
        input_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['sub'])]
        output_tensors = [Tensor(name='sub:0', source_op=['sub'],
                                dest_op=['mul'])]
        sub_node.construct('sub', 'Sub', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mul_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='sub:0', source_op=['sub'],
                                dest_op=['mul'])]
        output_tensors = [Tensor(name='mul:0', source_op=['mul'],
                                dest_op=['add'])]
        mul_node.construct('mul', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='mul:0', source_op=['mul'],
                                dest_op=['add']), Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='add:0', source_op=['add'],
                                dest_op=[])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mat_node = OPERATORS['MatMul']()
        input_tensors = [Tensor(), 
                        Tensor(name='src:0', dest_op=['matmul'], shape=[768])]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'], dest_op=['add_1'])]
        mat_node.construct('matmul', 'MatMul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        add_1_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'], dest_op=['add_1']),
                        Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='add_1:0', source_op=['add_1'], dest_op=['add_2'])]
        add_1_node.construct('add_1', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        add_2_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='add_1:0', source_op=['add_1'], dest_op=['add_2']),
                        Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='add_2:0', source_op=['add_2'], dest_op=['layernorm'])]
        add_2_node.construct('add_2', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        layernorm_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(name='add_2:0', source_op=['add_2'], dest_op=['layernorm']),
                        Tensor(data=np.array(1), shape=[768, 768]), 
                        Tensor(data=np.array(1), shape=[768])]
        output_tensors = [Tensor(name='layernorm:0', source_op=['layernorm'])]
        layernorm_node.construct('layernorm', 'LayerNorm', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        graph.insert_nodes(len(graph.nodes), [input_data_node, unsqueeze_1_node, unsqueeze_2_node,
                                                cast_node, sub_node, mul_node, add_node, mat_node,
                                                add_1_node, add_2_node, layernorm_node])
        graph = PaddingSequence()(graph)
        self.assertEqual(7, len(graph.nodes))
        self.assertEqual('-1,12,0,-1', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('AddV2', graph.nodes[2].op_type)
    

    def test_padding_sequence_3(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        equal_node = OPERATORS['Equal']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='equal:0', source_op=['equal'],
                                dest_op=['reshape'])]
        equal_node.construct('equal', 'Equal', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        shape_0_node = OPERATORS['Shape']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='shape_0:0', source_op=['shape_0'],
                                dest_op=['gather'])]
        shape_0_node.construct('shape_0', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        gather_node = OPERATORS['Gather']()
        input_tensors = [Tensor(name='shape_0:0', source_op=['shape_0'],
                                dest_op=['gather'])]
        output_tensors = [Tensor(name='gather:0', source_op=['gather'],
                                dest_op=['unsqueeze_2'])]
        gather_node.construct('gather', 'Gather', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unsqueeze_1_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'],
                                dest_op=['concat'])]
        unsqueeze_1_node.construct('unsqueeze_1', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unsqueeze_2_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor(name='gather:0', source_op=['gather'],
                                dest_op=['unsqueeze_2'])]
        output_tensors = [Tensor(name='unsqueeze_2:0', source_op=['unsqueeze_2'],
                                dest_op=['concat'])]
        unsqueeze_2_node.construct('unsqueeze_2', 'Unsqueeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'],
                                dest_op=['concat']), Tensor(name='unsqueeze_2:0', 
                                source_op=['unsqueeze_2'], dest_op=['concat'])]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'],
                                dest_op=['reshape'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='equal:0', source_op=['equal'],
                                dest_op=['reshape']), Tensor(name='concat:0', 
                                source_op=['concat'], dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=['expand'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        shape_1_node = OPERATORS['Shape']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='shape_1:0', source_op=['shape_1'],
                                dest_op=['expand'])]
        shape_1_node.construct('shape_1', 'Shape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        expand_node = OPERATORS['Expand']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'],
                                dest_op=['expand']), Tensor(name='shape_1:0', 
                                source_op=['shape_1'], dest_op=['expand'])]
        output_tensors = [Tensor(name='expand:0', source_op=['expand'],
                                dest_op=['cast'])]
        expand_node.construct('expand', 'Expand', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        cast_node = OPERATORS['Cast']()
        input_tensors = [Tensor(name='expand:0', source_op=['expand'],
                                dest_op=['cast'])]
        output_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['where'])]
        cast_node.construct('cast', 'Cast', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        where_node = OPERATORS['Where']()
        input_tensors = [Tensor(name='cast:0', source_op=['cast'],
                                dest_op=['where'])]
        output_tensors = [Tensor(name='where:0', source_op=['where'],
                                dest_op=[])]
        where_node.construct('where', 'Where', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, shape_0_node, gather_node, 
                                                equal_node, unsqueeze_1_node, unsqueeze_2_node, 
                                                concat_node, reshape_node, shape_1_node, 
                                                expand_node, cast_node, where_node])
        graph = PaddingSequence()(graph)
        self.assertEqual(3, len(graph.nodes))
        self.assertEqual('-1,12,0,-1', graph.nodes[1].attr['dst_shape'])
        self.assertEqual('AddV2', graph.nodes[2].op_type)


if __name__ == "__main__":
    unittest.main()
