import os
import shutil
import unittest
import copy
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper, onnx_pb
from onnxruntime.quantization.quant_utils import QuantizationMode
from lpot.adaptor.ox_utils.onnx_quantizer import ONNXQuantizer
import onnxruntime as ort


class TestAdaptorONNXRT(unittest.TestCase):

    qlinear_backend = QuantizationMode.QLinearOps
    integer_backend = QuantizationMode.IntegerOps
    q_config = {"weight":{'dtype': 3, 
                          'algorithm': 'minmax', 
                          'scheme':'sym', 
                          'granularity': 'per_tensor'},
                'activation':{'dtype': 2, 
                              'algorithm': 'minmax', 
                              'scheme':'asym', 
                              'granularity':'per_tensor'}
                }

    @classmethod
    def setUpClass(cls):
        os.makedirs('./onnxrt_test')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./onnxrt_test", ignore_errors=True)

    def static_test(self, model, q_config, quantize_params, quantizable_op_types):
        quantizer = ONNXQuantizer(copy.deepcopy(model),
            q_config,
            self.qlinear_backend,
            True,
            quantize_params,
            quantizable_op_types)
        quantizer.quantize_model()
        assert quantizer.model.model

    def dynamic_test(self, model, q_config, quantize_params, quantizable_op_types):
        quantizer = ONNXQuantizer(copy.deepcopy(model),
            q_config,
            self.integer_backend,
            False,
            quantize_params,
            quantizable_op_types)
        quantizer.quantize_model()
        assert quantizer.model.model

    def test_conv(self):
        for op in ['Conv', 'FusedConv']:
            A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
            C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 1])
            D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 1])
            conv_node = onnx.helper.make_node(op, ['A', 'B', 'C'], ['D'], 
                                              name=op, 
                                              kernel_shape=[3, 3], 
                                              pads=[1, 1, 1, 1])
            graph = helper.make_graph([conv_node], 'test_graph_1', [A, B, C], [D])
            model = helper.make_model(graph)
            q_config = {op: self.q_config},
            quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                               "B": [np.float32(10.), np.uint8(0)],
                               "C": [np.float32(10.), np.uint8(0)],
                               "D": [np.float32(10.), np.uint8(0)]}       
            quantizable_op_types = [op]
            self.static_test(model, q_config, quantize_params, quantizable_op_types)

    def test_matmul(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 5, 1])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 1])
        matmul_node = onnx.helper.make_node('MatMul', ['A', 'B'], ['C'], name='Matmul')
        graph = helper.make_graph([matmul_node], 'test_graph_1', [A, B], [C])
        model = helper.make_model(graph)
        q_config = {"Matmul": self.q_config}
        quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                           "B": [np.float32(10.), np.uint8(0)],
                           "C": [np.float32(10.), np.uint8(0)]}
        quantizable_op_types = ["Matmul"]
        self.static_test(model, q_config, quantize_params, quantizable_op_types)
        self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)
        q_config = {"Matmul": {"weight":{'dtype': 3,
                               'algorithm': 'minmax',
                               'scheme':'sym',
                               'granularity': 'per_tensor'},
                     'activation':{'dtype': 3,
                                   'algorithm': 'minmax',
                                   'scheme':'asym',
                                   'granularity':'per_tensor'}}}
        quantize_params = {}
        self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)

    def test_attention(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 5, 5])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 5])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
        node = onnx.helper.make_node('Attention', ['A', 'B', 'C'], ['D'], name='Attention')
        graph = helper.make_graph([node], 'test_graph_1', [A, B, C], [D])
        model = helper.make_model(graph)
        q_config = {"Attention": self.q_config}
        quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                           "B": [np.float32(10.), np.uint8(0)],
                           "C": [np.float32(10.), np.uint8(0)],
                           "D": [np.float32(10.), np.uint8(0)]}
        quantizable_op_types = ["Attention"]
        self.static_test(model, q_config, quantize_params, quantizable_op_types)
        self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)

    def test_gather(self):
        a_value = np.random.randn(100, 4).astype(np.float32)
        A_init = helper.make_tensor('A', TensorProto.FLOAT, [100, 4], 
                                    a_value.reshape(400).tolist())
        b_value = np.random.randint(2, size=(1, 10)).astype(np.int32)
        B_init = helper.make_tensor('B', TensorProto.INT32, [1, 10],
                                    b_value.reshape(10).tolist())
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [100, 4])
        B = helper.make_tensor_value_info('B', TensorProto.INT32, [1, 10])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [10, 4])
        node = onnx.helper.make_node('Gather', ['A', 'B'], ['C'], name='Gather')
        graph = helper.make_graph([node], 'test_graph_1', [A, B], [C], [A_init, B_init])
        model = helper.make_model(graph)
        q_config = {'Gather': {"weight":{'dtype': 3,
                                         'algorithm': 'minmax',
                                         'scheme':'sym',
                                         'granularity': 'per_tensor'},
                              'activation':{'dtype': 2,
                                         'algorithm': 'minmax',
                                         'scheme':'asym',
                                         'granularity':'per_tensor'}
                  }} 
        quantize_params = {"A": [np.float32(10.), np.uint8(0)]}
        quantizable_op_types = ["Gather"]
        self.static_test(model, q_config, quantize_params, quantizable_op_types)
        self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)
        graph = helper.make_graph([node], 'test_graph_1', [A, B], [C])
        model = helper.make_model(graph)
        q_config = {'Gather': {"weight":{'dtype': 3,
                                         'algorithm': 'minmax',
                                         'scheme':'sym',
                                         'granularity': 'per_tensor'},
                              'activation':{'dtype': 2,
                                         'algorithm': 'minmax',
                                         'scheme':'asym',
                                         'granularity':'per_tensor'}
                  }}
        quantize_params = {}
        self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)

    def test_binary(self):
        for op in ['Mul', 'Add']:
            A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 10])
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1])
            C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 10])
            node = onnx.helper.make_node(op, ['A', 'B'], ['C'], name=op)
            graph = helper.make_graph([node], 'test_graph_1', [A, B], [C])
            model = helper.make_model(graph)
            q_config = {op: self.q_config}
            quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                               "B": [np.float32(10.), np.uint8(0)],
                               "C": [np.float32(10.), np.uint8(0)]}
            quantizable_op_types = [op]
            self.static_test(model, q_config, quantize_params, quantizable_op_types)
    
    def test_relu(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'], 
                                          name='Conv', 
                                          kernel_shape=[3, 3], 
                                          pads=[1, 1, 1, 1])
        relu_node = onnx.helper.make_node('Relu', ['C'], ['D'], name='Relu')
        graph = helper.make_graph([conv_node, relu_node], 'test_graph_1', [A, B], [D])
        model = helper.make_model(graph)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "./onnxrt_test/optimized_model.onnx"  
        session = ort.InferenceSession(model.SerializeToString(), sess_options)
        model = onnx.load(sess_options.optimized_model_filepath)
 
        q_config = {"Conv": self.q_config, "Relu": self.q_config}
        quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                           "B": [np.float32(10.), np.uint8(0)],
                           "C": [np.float32(10.), np.uint8(0)],
                           "D": [np.float32(10.), np.uint8(0)]}
        quantizable_op_types = ["Conv", "Relu"]
        self.static_test(model, q_config, quantize_params, quantizable_op_types)
        
    def test_clip(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'],
                                          name='Conv',
                                          kernel_shape=[3, 3],
                                          pads=[1, 1, 1, 1])
        clip_node = onnx.helper.make_node('Clip', ['C'], ['D'], name='Clip')
        graph = helper.make_graph([conv_node, clip_node], 'test_graph_1', [A, B], [D])
        model = helper.make_model(graph)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "./onnxrt_test/optimized_model.onnx"
        session = ort.InferenceSession(model.SerializeToString(), sess_options)
        model = onnx.load(sess_options.optimized_model_filepath)

        q_config = {"Conv": self.q_config, "Clip": self.q_config}
        quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                           "B": [np.float32(10.), np.uint8(0)],
                           "C": [np.float32(10.), np.uint8(0)],
                           "D": [np.float32(10.), np.uint8(0)]}
        quantizable_op_types = ["Conv", "Clip"]
        self.static_test(model, q_config, quantize_params, quantizable_op_types)

    def test_activation(self):
        for op in ["Relu", "LeakyRelu", "Sigmoid"]:
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 10])
            A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 10])
            node = onnx.helper.make_node(op, ['A'], ['B'], name=op)
            graph = helper.make_graph([node], 'test_graph_1', [A], [B])
            model = helper.make_model(graph)
            q_config = {op: self.q_config}
            quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                               "B": [np.float32(10.), np.uint8(0)]}
            quantizable_op_types = [op]
            self.static_test(model, q_config, quantize_params, quantizable_op_types)

            a_value = np.random.randn(1, 10).astype(np.float32)
            A_init = helper.make_tensor('A', TensorProto.FLOAT, [1, 10],
                                        a_value.reshape(10).tolist())
            graph = helper.make_graph([node], 'test_graph_1', [A], [B], [A_init])
            model = helper.make_model(graph)
            self.static_test(model, q_config, quantize_params, quantizable_op_types)

    def test_pooling(self):
        for op in ["MaxPool", "GlobalAveragePool"]:
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 5, 5])
            A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
            a_value = np.random.randn(1, 1, 5, 5).astype(np.float32)
            A_init = helper.make_tensor('A', TensorProto.FLOAT, [1, 1, 5, 5],
                                    a_value.reshape(25).tolist())
            node = onnx.helper.make_node(op, ['A'], ['B'], 
                                         name=op,
                                         kernel_shape=[3, 3],
                                         pads=[1, 1, 1, 1])
            graph = helper.make_graph([node], 'test_graph_1', [A], [B], [A_init]) 
            model = helper.make_model(graph)
            q_config = {op: self.q_config}
            quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                               "B": [np.float32(10.), np.uint8(0)]}
            quantizable_op_types = [op]
            self.static_test(model, q_config, quantize_params, quantizable_op_types)

            A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
            D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
            conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'],
                                              name='Conv',
                                              kernel_shape=[3, 3],
                                              pads=[1, 1, 1, 1])
            pool_node = onnx.helper.make_node(op, ['C'], ['D'], name=op)
            graph = helper.make_graph([conv_node, pool_node], 'test_graph_1', [A, B], [D])
            model = helper.make_model(graph)
 
            q_config = {"Conv": self.q_config, op: self.q_config}
            quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                               "B": [np.float32(10.), np.uint8(0)],
                               "C": [np.float32(10.), np.uint8(0)],
                               "D": [np.float32(10.), np.uint8(0)]}
            quantizable_op_types = ["Conv", op]
            self.static_test(model, q_config, quantize_params, quantizable_op_types)

    def test_exclude_node(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'],
                                          name='Conv',
                                          kernel_shape=[3, 3],
                                          pads=[1, 1, 1, 1])
        pool_node = onnx.helper.make_node("MaxPool", ['C'], ['D'], name="MaxPool")
        graph = helper.make_graph([conv_node, pool_node], 'test_graph_1', [A, B], [D])
        model = helper.make_model(graph)

        q_config = {"Conv": self.q_config, "MaxPool": "fp32"}
        quantize_params = {"A": [np.float32(10.), np.uint8(0)],
                           "B": [np.float32(10.), np.uint8(0)],
                           "C": [np.float32(10.), np.uint8(0)],
                           "D": [np.float32(10.), np.uint8(0)]}
        quantizable_op_types = ["Conv", "MaxPool"]
        self.static_test(model, q_config, quantize_params, quantizable_op_types)

if __name__ == "__main__":
    unittest.main()
