import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.gelu import Gelu


class TestGelu(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_gelu_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        div_node = OPERATORS['Div']()
        input_tensors = [Tensor()]
        output_tensors = [Tensor(name='div:0', source_op=['div'], dest_op=['erf'])]
        div_node.construct('div', 'Div', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        erf_node = OPERATORS['Erf']()
        input_tensors = [Tensor(name='div:0', source_op=['div'], dest_op=['erf'])]
        output_tensors = [Tensor(name='erf:0', source_op=['erf'], dest_op=['add'])]
        erf_node.construct('erf', 'Erf', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        add_node = OPERATORS['Add']()
        input_tensors = [Tensor(name='erf:0', source_op=['erf'], dest_op=['add'])]
        output_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=['mul_1'])]
        add_node.construct('add', 'Add', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mul_1_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='add:0', source_op=['add'], dest_op=['mul_1'])]
        output_tensors = [Tensor(name='mul_1:0', source_op=['mul_1'], dest_op=['mul_2'])]
        mul_1_node.construct('mul_1', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        mul_2_node = OPERATORS['Mul']()
        input_tensors = [Tensor(name='mul_1:0', source_op=['mul_1'], dest_op=['mul_2'])]
        output_tensors = [Tensor(name='mul_2:0', source_op=['mul_2'], dest_op=[])]
        mul_2_node.construct('mul_2', 'Mul', input_tensors=input_tensors, 
                                output_tensors=output_tensors)
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, div_node, erf_node, add_node,
                                                mul_1_node, mul_2_node])
        graph = Gelu()(graph)
        self.assertEqual(2, len(graph.nodes))
        self.assertEqual('mul_2', graph.nodes[-1].name)
        self.assertEqual('Gelu', graph.nodes[-1].op_type)


if __name__ == "__main__":
    unittest.main()
