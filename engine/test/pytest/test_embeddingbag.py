import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.embeddingbag import EmbeddingBag
import numpy as np


class TestEmbeddingBag(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_embeddingbag_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        squeeze_node = OPERATORS['Squeeze']()
        input_tensors = [Tensor(data=np.array(1)), Tensor(data=np.array(1)), 
                            Tensor(data=np.array(1))]
        output_tensors = [Tensor(name='matmul:0', source_op=['matmul'], 
                                    dest_op=['relu'])]
        squeeze_node.construct('matmul', 'Squeeze', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'src1_perm': '1,0'}))
        
        embeddingbag_node = OPERATORS['EmbeddingBag']()
        input_tensors = [Tensor(name='matmul:0', source_op=['matmul'], 
                                    dest_op=['relu']),
                         Tensor(name='matmul1:0', source_op=['matmul'], 
                                    dest_op=['relu']),
                         Tensor(name='matmul2:0', source_op=['matmul'], 
                                    dest_op=['relu'])]
        output_tensors = [Tensor(name='relu:0', source_op=['relu'],
                                dest_op=[])]
        embeddingbag_node.construct('relu', 'EmbeddingBag', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, squeeze_node, embeddingbag_node])
        graph = EmbeddingBag()(graph)
        self.assertEqual(3, len(graph.nodes))


if __name__ == "__main__":
    unittest.main()
