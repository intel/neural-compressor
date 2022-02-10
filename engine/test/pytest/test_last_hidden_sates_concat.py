import unittest
from collections import OrderedDict
import numpy as np
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.last_hidden_states_concat import LastHiddenStatesConcat


class TestLastHiddenStatesConcat(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_last_hidden_states_concat(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        unsqueeze_1_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'], 
                                dest_op=['concat'])]
        unsqueeze_1_node.construct('unsqueeze_1', 'Unsqueeze', input_tensors=input_tensors,
                                    output_tensors=output_tensors)
        
        unsqueeze_2_node = OPERATORS['Unsqueeze']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='unsqueeze_2:0', source_op=['unsqueeze_2'], 
                                dest_op=['concat'])]
        unsqueeze_2_node.construct('unsqueeze_2', 'Unsqueeze', input_tensors=input_tensors,
                                    output_tensors=output_tensors)
        
        concat_node = OPERATORS['Concat']()
        input_tensors = [Tensor(name='unsqueeze_1:0', source_op=['unsqueeze_1'], 
                                dest_op=['concat']), Tensor(name='unsqueeze_2:0', 
                                source_op=['unsqueeze_2'], dest_op=['concat'])]
        output_tensors = [Tensor(name='concat:0', source_op=['concat'],
                                dest_op=['reduce_mean'])]
        concat_node.construct('concat', 'Concat', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        reduce_mean_node = OPERATORS['ReduceMean']()
        input_tensors = [Tensor(name='concat:0', source_op=['concat'],
                                dest_op=['reduce_mean'])]
        output_tensors = [Tensor(name='reduce_mean:0', source_op=['reduce_mean'],
                                dest_op=['mat'])]
        reduce_mean_node.construct('reduce_mean', 'ReduceMean', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        mat_node = OPERATORS['MatMulWithBias']()
        input_tensors = [Tensor(name='reduce_mean:0', source_op=['reduce_mean'],
                                dest_op=['mat']), Tensor(data=np.array(1), shape=[256,3]), 
                                Tensor(data=np.array(1), shape=[3])]
        output_tensors = [Tensor(name='mat:0', source_op=['mat'], dest_op=[])]
        mat_node.construct('mat', 'MatMulWithBias', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, unsqueeze_1_node, unsqueeze_2_node,
                                            concat_node, reduce_mean_node, mat_node])
        graph = LastHiddenStatesConcat()(graph)
        self.assertEqual(8, len(graph.nodes))


if __name__ == "__main__":
    unittest.main()
