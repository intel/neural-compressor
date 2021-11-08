import unittest
from tensorflow.core.framework import node_def_pb2 
import engine.compile.tf_utils as util 


class TestTfUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass
    
    def test_create_tf_node(self):
        test_node = util.create_tf_node('Reshape', 'test_name', ['input_0'])
        self.assertEqual('Reshape', test_node.op)
        self.assertEqual('test_name', test_node.name)
        self.assertSequenceEqual(['input_0'], test_node.input)


if __name__ == "__main__":
    unittest.main()
