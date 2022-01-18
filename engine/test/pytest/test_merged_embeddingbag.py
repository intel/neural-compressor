import unittest
from collections import OrderedDict
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor
from engine.compile.graph import Graph
from engine.compile.sub_graph.merged_embeddingbag import MergedEmbeddingbag
import numpy as np


class TestMergedEmbeddingbag(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_merged_embeddingbag_1(self):
        graph = Graph()
        input_data_node = OPERATORS['Input']()
        input_tensors = []
        offsets = Tensor(name='offsets', source_op=[],
                                    dest_op=['input_data'])
        indices = Tensor(name='indices', source_op=[],
                                    dest_op=['input_data'])
        input_output_tensors = [offsets, indices]
        input_data_node.construct('input_data', 'Input', input_tensors=input_tensors,
                                output_tensors=input_output_tensors)

        # 0
        split_0_node = OPERATORS['Split']()
        split_0_output_tensors = [Tensor(name='split_0_ouput', source_op=['split_0'],
                                    dest_op=['squeeze_0'])]
        split_0_node.construct('split_0', 'Split', input_tensors=[indices],
                                output_tensors=split_0_output_tensors)

        # 1
        squeeze_0_node = OPERATORS['Squeeze']()
        squeeze_0_output_tensors = [Tensor(name='squeeze_0_ouput', source_op=['squeeze_0'],
                                    dest_op=['shape_0'])]
        squeeze_0_node.construct('squeeze_0', 'Squeeze', input_tensors=split_0_output_tensors,
                                output_tensors=squeeze_0_output_tensors)

        # 2
        shape_0_node = OPERATORS['Shape']()
        shape_0_output_tensors = [Tensor(name='shape_0_ouput', source_op=['shape_0'],
                                    dest_op=['gather_0'])]
        shape_0_node.construct('shape_0', 'Shape', input_tensors=squeeze_0_output_tensors,
                                output_tensors=shape_0_output_tensors)

        # 3
        gather_0_node = OPERATORS['Gather']()
        gather_0_output_tensors = [Tensor(name='gather_0_ouput', source_op=['gather_0'],
                                    dest_op=['unsqueeze_0'])]
        gather_0_node.construct('gather_0', 'Gather', input_tensors=shape_0_output_tensors,
                                output_tensors=gather_0_output_tensors)

        # 4
        gather_2_node = OPERATORS['Gather']()
        gather_2_output_tensors = [Tensor(name='gather_2_ouput', source_op=['gather_2'],
                                    dest_op=['concat_0'])]
        gather_2_node.construct('gather_2', 'Gather', input_tensors=[offsets],
                                output_tensors=gather_2_output_tensors)

        # 5
        unsqueeze_0_node = OPERATORS['Unsqueeze']()
        unsqueeze_0_output_tensors = [Tensor(name='unsqueeze_0_ouput',
                                    source_op=['unsqueeze_0'],
                                    dest_op=['concat_0'])]
        unsqueeze_0_node.construct('unsqueeze_0', 'Unsqueeze',
                                input_tensors=gather_0_output_tensors,
                                output_tensors=unsqueeze_0_output_tensors)

        # 6
        concat_0_node = OPERATORS['Concat']()
        concat_0_output_tensors = [Tensor(name='concat_0_ouput', source_op=['concat_0'],
                                    dest_op=['slice_0'])]
        concat_0_node.construct('concat_0', 'Concat',
                                input_tensors=[gather_2_output_tensors[0],
                                unsqueeze_0_output_tensors[0]],
                                output_tensors=concat_0_output_tensors)

        # 7
        slice_0_node = OPERATORS['Slice']()
        slice_0_output_tensors = [Tensor(name='slice_0_ouput', source_op=['slice_0'],
                                    dest_op=['shape_1'])]
        slice_0_node.construct('slice_0', 'Slice', input_tensors=concat_0_output_tensors,
                                output_tensors=slice_0_output_tensors)

        # 8
        shape_1_node = OPERATORS['Shape']()
        shape_1_output_tensors = [Tensor(name='shape_1_ouput', source_op=['shape_1'],
                                    dest_op=['gather_1'])]
        shape_1_node.construct('shape_1', 'Shape', input_tensors=slice_0_output_tensors,
                                output_tensors=shape_1_output_tensors)

        # 9
        gather_1_node = OPERATORS['Gather']()
        gather_1_output_tensors = [Tensor(name='gather_1_ouput', source_op=['gather_1'],
                                    dest_op=['loop_0'])]
        gather_1_node.construct('gather_1', 'Gather',
                                input_tensors=shape_1_output_tensors,
                                output_tensors=gather_1_output_tensors)

        # 10
        loop_0_node = OPERATORS['Loop']()
        loop_0_weight = Tensor(name='loop_0_weight', source_op=[],
                            dest_op=['loop_0'], data=np.array([1]))
        loop_0_output_tensors = [Tensor(name='loop_0_ouput', source_op=['loop_0'],
                                    dest_op=['output_data'])]
        loop_0_node.construct('loop_0', 'Loop', input_tensors=[loop_0_weight,
                                gather_1_output_tensors[0]],
                                output_tensors=loop_0_output_tensors)

        # 11
        output_data_node = OPERATORS['Output']()
        output_data_node.construct('output_data', 'Output',
                                input_tensors=loop_0_output_tensors,
                                output_tensors=[])

        graph.insert_nodes(len(graph.nodes), [input_data_node,
                split_0_node, squeeze_0_node, shape_0_node,
                gather_0_node, gather_2_node, unsqueeze_0_node,
                concat_0_node, slice_0_node, shape_1_node,
                gather_1_node, loop_0_node, output_data_node])

        graph = MergedEmbeddingbag()(graph)
        self.assertEqual(3, len(graph.nodes))

if __name__ == "__main__":
    unittest.main()
