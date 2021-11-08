import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.layer_norm_with_reduce_mean import LayerNormWithReduceMean


class TestLayerNormWithReduceMean(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_layer_norm_with_reduce_mean_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        output_tensors = [Tensor(), Tensor(), Tensor()]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors, 
                                output_tensors=output_tensors)

        ln_node = OPERATORS['LayerNorm']()
        input_tensors = [Tensor(), Tensor(shape=[768]), Tensor(shape=[768])]
        output_tensors = [Tensor(name='layer_norm:0', source_op=['layer_norm'], 
                                    dest_op=['reduce_mean'])]
        ln_node.construct('layer_norm', 'LayerNorm', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict({
                                    'epsilon': 0.009}))
        
        reduce_mean_node = OPERATORS['ReduceMean']()
        input_tensors = [Tensor(name='layer_norm:0', source_op=['layer_norm'], 
                                    dest_op=['reduce_mean'])]
        output_tensors = [Tensor(name='reduce_mean:0', source_op=['reduce_mean'],
                                dest_op=[])]
        reduce_mean_node.construct('reduce_mean', 'ReduceMean', input_tensors=input_tensors, 
                                output_tensors=output_tensors, attr=OrderedDict(
                                    {'axis': 1, 'keep_dims': False}))
        
        graph.insert_nodes(len(graph.nodes), [input_data_node, ln_node, reduce_mean_node])
        graph = LayerNormWithReduceMean()(graph)
        self.assertEqual(5, len(graph.nodes))
        self.assertEqual('-1,-1,768', graph.nodes[2].attr['dst_shape'])
        self.assertEqual('reducemean_after_reshape', graph.nodes[3].name)
        self.assertEqual(1, graph.nodes[3].attr['axis'])
        self.assertEqual('-1,768', graph.nodes[4].attr['dst_shape'])


if __name__ == "__main__":
    unittest.main()
