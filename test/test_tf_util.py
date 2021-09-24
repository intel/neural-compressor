import os
import unittest
import numpy as np
import tensorflow as tf
from neural_compressor.adaptor.tf_utils.util import get_graph_def
from neural_compressor.adaptor.tf_utils.util import collate_tf_preds
from neural_compressor.adaptor.tf_utils.util import fix_ref_type_of_graph_def
from neural_compressor.adaptor.tf_utils.util import disable_random
from tensorflow.core.framework import graph_pb2
from neural_compressor.adaptor.tf_utils.graph_rewriter.graph_util import GraphRewriterHelper as Helper
from tensorflow.python.framework import dtypes

def build_fake_graphdef():
    graph_def = graph_pb2.GraphDef()
    constant_1_name = 'moving_1/switch_input_const'
    constant_1 = Helper.create_constant_node(
            constant_1_name,
            value=0.,
            dtype=dtypes.float32)

    constant_3_name = 'moving_1/switch_input_const/read'
    constant_3 = Helper.create_constant_node(
            constant_3_name,
            value=[1],
            dtype=dtypes.float32)

    constant_2_name = 'switch_input_const2'
    constant_2 = Helper.create_constant_node(
            constant_2_name,
            value=2.,
            dtype=dtypes.float32)
    equal_name = 'equal'
    equal = Helper.create_node("Equal", equal_name, [constant_1_name, constant_2_name])
    Helper.set_attr_dtype(equal, 'T', dtypes.float32)

    refswitch_name = 'refswitch'
    refswitch_node = Helper.create_node("RefSwitch", refswitch_name,
                                                    [constant_1_name ,equal_name])
    Helper.set_attr_dtype(refswitch_node, 'T', dtypes.float32)

    variable_name = 'variable'
    variable_node = Helper.create_node("VariableV2", variable_name,
                                                    [])
    Helper.set_attr_dtype(variable_node, 'T', dtypes.float32)

    assign_name = 'assign'
    assign_node = Helper.create_node("Assign", assign_name,
                                                    [variable_name,refswitch_name])
    Helper.set_attr_bool(assign_node, 'use_locking', True)
    Helper.set_attr_bool(assign_node, 'validate_shape', True)
    Helper.set_attr_dtype(assign_node, 'T', dtypes.float32)

    assignsub_name = 'assignsub'
    assignsub_node = Helper.create_node("AssignSub", assignsub_name,
                                                    [assign_name,constant_1_name])
    Helper.set_attr_bool(assignsub_node, 'use_locking', True)
    Helper.set_attr_dtype(assignsub_node, 'T', dtypes.float32)

    assignadd_name = 'assignadd'
    assignadd_node = Helper.create_node("AssignAdd", assignadd_name,
                                                    [assignsub_name,constant_2_name])
    Helper.set_attr_bool(assignadd_node, 'use_locking', True)
    Helper.set_attr_dtype(assignadd_node, 'T', dtypes.float32)

    graph_def.node.extend([
        constant_1,
        constant_2,
        constant_3,
        equal,
        refswitch_node,
        variable_node,
        assign_node,
        assignsub_node,
        assignadd_node
    ])
    return graph_def

class TestTFutil(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        os.remove('test.pb')

    @disable_random()
    def test_fix_ref_type(self):
        graph_def = build_fake_graphdef()
        new_graph_def = fix_ref_type_of_graph_def(graph_def)
        f = tf.io.gfile.GFile('./test.pb', 'wb')
        f.write(new_graph_def.SerializeToString())
        find_Assign_prefix = False
        for node in new_graph_def.node:
            if 'Assign' in node.op:
                find_Assign_prefix = True
        self.assertFalse(find_Assign_prefix, False)

    @disable_random()
    def test_collate_tf_preds(self):
        results = [[1],[np.array([2])]]
        data = collate_tf_preds(results)
        self.assertEqual(data,[1,np.array([2])])

    @disable_random()
    def test_get_graph_def(self):
        graphdef = get_graph_def('./test.pb', outputs="assignadd")
        self.assertIsInstance(graphdef, tf.compat.v1.GraphDef)

if __name__ == "__main__":
    unittest.main()