import unittest
import engine.compile as compile
import numpy as np
from engine.compile.ops.op import OPERATORS, Operator
from engine.compile.ops.tensor import Tensor


class TestOnnxUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_change_num_name(self):
        out = compile.onnx_utils.change_num_name(1)
        self.assertEqual(1, out)
        
    def test_change_num_namei_same(self):
        out = compile.onnx_utils.change_num_name('1')
        self.assertEqual('1_tensor', out)
    
    def test_bias_to_int32_if1(self):
        fake_input_tensors = [
          Tensor(data=np.array([[1,2],[3,4]], dtype=np.int8)),
          Tensor(data=np.array(0.1, dtype=np.float32)),
          Tensor(data=np.array(0.2, dtype=np.float32)),
          Tensor(data=None),
          Tensor(data=None),
          Tensor(data=None),
        ]
        fake_bias_node = OPERATORS['Add']()
        fake_bias_node.construct('bias_add', 'Add', 
                                    input_tensors=fake_input_tensors)
        out = compile.onnx_utils.bias_to_int32(fake_bias_node, 0.3, 0.4)
        golden_out = np.array([[1,2],[2,3]])
        self.assertSequenceEqual(golden_out.tolist(), out.tolist())
    
    def test_bias_to_int32_else(self):
        fake_input_tensors = [
          Tensor(data=None, source_op=[None]),
          Tensor(data=None, source_op=[None]),
          Tensor(data=None, source_op=[None]),
          Tensor(data=np.array([[1,2],[3,4]], dtype=np.int8)),
          Tensor(data=np.array(0.1, dtype=np.float32)),
          Tensor(data=np.array(0.2, dtype=np.float32)),
        ]
        fake_bias_node = OPERATORS['Add']()
        fake_bias_node.construct('bias_add', 'Add', 
                                    input_tensors=fake_input_tensors)
        out = compile.onnx_utils.bias_to_int32(fake_bias_node, 0.3, 0.4)
        golden_out = np.array([[1,2],[2,3]])
        self.assertSequenceEqual(golden_out.tolist(), out.tolist())
    
    def test_bias_to_int32_if2(self):
        fake_input_tensors = [
          Tensor(data=np.array([[1,2],[3,4]], dtype=np.int64)),
          Tensor(data=np.array(0.1, dtype=np.float32)),
          Tensor(data=np.array(0.2, dtype=np.float32)),
          Tensor(data=None),
          Tensor(data=None),
          Tensor(data=None),
        ]
        fake_bias_node = OPERATORS['Add']()
        fake_bias_node.construct('bias_add', 'Add', 
                                    input_tensors=fake_input_tensors)
        out = compile.onnx_utils.bias_to_int32(fake_bias_node, 0.3, 0.4)
        self.assertEqual(None, out)
        


if __name__ == "__main__":
    unittest.main()
