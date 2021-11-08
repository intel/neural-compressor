import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.last_layer_shape import LastLayerShape


class TestLastLayerShape(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_last_layer_shape_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        pack_node = OPERATORS['Pack']()
        input_tensors = [Tensor(), Tensor(), Tensor(data=768)]
        output_tensors = [Tensor(name='pack:0', source_op=['pack'], dest_op=['reshape'])]
        pack_node.construct('pack', 'Pack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reshape_node = OPERATORS['Reshape']()
        input_tensors = [Tensor(name='pack:0', source_op=['pack'], dest_op=['reshape'])]
        output_tensors = [Tensor(name='reshape:0', source_op=['reshape'], 
                                    dest_op=['strided_slice'])]
        reshape_node.construct('reshape', 'Reshape', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        strided_slice_node = OPERATORS['StridedSlice']()
        input_tensors = [Tensor(name='reshape:0', source_op=['reshape'], 
                                    dest_op=['strided_slice'])]
        output_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'], 
                                dest_op=['squeeze'])]
        strided_slice_node.construct('strided_slice', 'StridedSlice', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({'test': 1}))
        
        squeeze_node = OPERATORS['Squeeze']()
        input_tensors = [Tensor(name='strided_slice:0', source_op=['strided_slice'], 
                                dest_op=['squeeze'])]
        output_tensors = [Tensor(name='squeeze:0', dest_op=[])]
        squeeze_node.construct('squeeze', 'Squeeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, pack_node, reshape_node, 
                                                strided_slice_node, squeeze_node])
        
        graph = LastLayerShape()(graph)
        self.assertEqual(4, len(graph.nodes))
        self.assertEqual('-1,-1,768', graph.nodes[1].attr['dst_shape'])
        self.assertEqual(1, graph.nodes[2].attr['test'])
        self.assertEqual('-1,768', graph.nodes[3].attr['dst_shape'])


if __name__ == "__main__":
    unittest.main()
