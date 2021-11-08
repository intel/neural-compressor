import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.start_end_logits import StartEndLogits


class TestStartEndLogits(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_start_end_logits(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        transpose_node = OPERATORS['Transpose']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                dest_op=['unpack'])]
        transpose_node.construct('transpose', 'Transpose', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unpack_node = OPERATORS['Unpack']()
        input_tensors = [Tensor(name='transpose:0', source_op=['transpose'],
                                dest_op=['unpack'])]
        output_tensors = [Tensor(name='unpack:0', source_op=['unpack'],
                                dest_op=['identity'])]
        unpack_node.construct('unpack', 'Unpack', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        identity_node = OPERATORS['Identity']()
        input_tensors = [Tensor(name='unpack:0', source_op=['unpack'],
                                dest_op=['identity'])]
        output_tensors = [Tensor(name='identity:0', source_op=['identity'], dest_op=[])]
        identity_node.construct('identity', 'Identity', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, transpose_node, unpack_node,
                                                    identity_node])
        graph = StartEndLogits()(graph)
        self.assertEqual(1, len(graph.nodes))


if __name__ == "__main__":
    unittest.main()
