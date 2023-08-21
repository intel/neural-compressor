import os
import shutil
import unittest
import copy
import onnx
import numpy as np
from collections import Counter
from onnx import helper, TensorProto, numpy_helper, onnx_pb
from neural_compressor.adaptor.ox_utils.operators import QOPERATORS
from neural_compressor.adaptor.ox_utils.quantizer import Quantizer
from neural_compressor.adaptor.ox_utils.util import QuantizedInitializer, QuantizedValue, QuantizationMode
import onnxruntime as ort
from neural_compressor.config import ONNXQlinear2QDQConfig
from neural_compressor.utils.utility import CpuInfo

def build_model():
    initializers = []
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 15, 15])
    output = helper.make_tensor_value_info('add_out_2', TensorProto.FLOAT, [88, 11])
    
    add_node = onnx.helper.make_node('Add', ['input', 'add_init'], ['add_out'], name='add')

    conv1_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [3, 3, 3, 3]).astype(np.float32), name='conv1_weight')
    conv1_node = helper.make_node('Conv', ['add_out', 'conv1_weight'], ['conv1_output'], name='conv1')

    conv2_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [5, 3, 3, 3]).astype(np.float32), name='conv2_weight')
    conv2_node = helper.make_node('Conv', ['add_out', 'conv2_weight'], ['conv2_output'], name='conv2')

    # 1, 8, 13, 13
    concat_node = helper.make_node('Concat', ['conv1_output', 'conv2_output'], [
                                        'concat_output'], name='Concat', axis=1)
    # 1, 8, 11, 11
    avg_args = {'kernel_shape': [3, 3]}
    avgpool_node = helper.make_node('AveragePool', ['concat_output'], ['avg_output'], name='AveragePool', **avg_args)
    reshape_node = onnx.helper.make_node('Reshape', ['avg_output', 'shape'], ['reshape_output'], name='Reshape')

    add_node_2 = onnx.helper.make_node('Add', ['reshape_output', 'add_init_2'], ['add_out_2'], name='add_2')

    initializers = [conv1_weight_initializer, conv2_weight_initializer]
    initializers.append(onnx.numpy_helper.from_array(np.array([88, 11], dtype=np.int64), name='shape'))
    initializers.append(onnx.numpy_helper.from_array(np.zeros((1, 3, 15, 15)).astype('float32'), name='add_init'))
    initializers.append(onnx.numpy_helper.from_array(np.zeros((88, 11)).astype('float32'), name='add_init_2'))
 
    graph = helper.make_graph([conv1_node, conv2_node, concat_node, avgpool_node, reshape_node, add_node, add_node_2],
                              'test', [input], [output], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model

class TestAdaptorONNXRT(unittest.TestCase):

    qlinear_backend = QuantizationMode.QLinearOps
    qdq_backend = 'qdq'
    integer_backend = QuantizationMode.IntegerOps
    static_q_config = {"weight":{'dtype': 3, 
                          'algorithm': 'minmax', 
                          'scheme':'sym', 
                          'granularity': 'per_tensor'},
                       'activation':{'dtype': 2, 
                                     'algorithm': 'minmax', 
                                     'scheme':'asym', 
                                     'granularity':'per_tensor',
                                     'quant_mode': 'static'}
                       }
    dynamic_q_config = {"weight":{'dtype': 3, 
                          'algorithm': 'minmax', 
                          'scheme':'sym', 
                          'granularity': 'per_tensor'},
                        'activation':{'dtype': 2, 
                                      'algorithm': 'minmax', 
                                      'scheme':'asym', 
                                      'granularity':'per_tensor',
                                      'quant_mode': 'dynamic'}
                        }

    @classmethod
    def setUpClass(cls):
        os.makedirs('./onnxrt_test')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./onnxrt_test", ignore_errors=True)

    def qlinear_test(self, model, q_config, quantize_params, quantizable_op_types, **kwargs):
        quantizer = Quantizer(copy.deepcopy(model),
            q_config,
            self.qlinear_backend,
            True,
            quantize_params,
            quantizable_op_types,
            **kwargs)
        quantizer.quantize_model()
        assert quantizer.model.model
        return quantizer.model

    def qdq_test(self, model, q_config, quantize_params, quantizable_op_types, **kwargs):
        quantizer = Quantizer(copy.deepcopy(model),
            q_config,
            self.qdq_backend,
            True,
            quantize_params,
            quantizable_op_types,
            **kwargs)
        quantizer.quantize_model()
        assert quantizer.model.model
        return quantizer.model

    def dynamic_test(self, model, q_config, quantize_params, quantizable_op_types):
        quantizer = Quantizer(copy.deepcopy(model),
            q_config,
            self.integer_backend,
            False,
            quantize_params,
            quantizable_op_types)
        quantizer.quantize_model()
        assert quantizer.model.model
        return quantizer.model

    def test_resize(self):
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 26, 42])

        conv_weight_arr = np.random.randint(-1, 2, [3, 2, 3, 3]).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name='conv1_weight')
        conv_node = onnx.helper.make_node('Conv', ['input', 'conv1_weight'], ['conv_output'], name='conv_node')

        initializers = [conv_weight_initializer]

        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 48, 80])
        resize_inputs = ['conv_output']  # resize_roi_name, resize_scales_name, resize_sizes_name]
        resize_attrs = {'coordinate_transformation_mode': 'asymmetric', 'mode': 'nearest', 'nearest_mode': 'floor'}
        resize_node = helper.make_node('Resize', resize_inputs, ['output'], name='resize_node', **resize_attrs)
        resize_roi = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        resize_roi_name = 'resize_roi'
        resize_roi_initializer = helper.make_tensor(resize_roi_name, TensorProto.FLOAT, [len(resize_roi)], resize_roi)
        initializers.extend([resize_roi_initializer])
        resize_node.input.extend([resize_roi_name])

        resize_scales = [1.0, 1.0, 2.0, 2.0]
        resize_scales_name = 'resize_scales'
        resize_scales_initializer = helper.make_tensor(resize_scales_name, TensorProto.FLOAT, [
                                                       len(resize_scales)], resize_scales)
        initializers.extend([resize_scales_initializer])
        resize_node.input.extend([resize_scales_name])

        graph = helper.make_graph([conv_node, resize_node], 'TestOpQuantizerResize_test_model',
                                  [input_tensor], [output_tensor], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7 # use stable onnx ir version
        
        q_config = {'conv_node': self.static_q_config,
                    'resize_node': self.static_q_config}
        quantize_params = {'input': [np.uint8(0), np.float32(10.)],
                           'conv1_weight': [np.uint8(0), np.float32(10.)],
                           'conv_output': [np.uint8(0), np.float32(10.)],
                           'output': [np.uint8(0), np.float32(10.)],
                           }
        
        q_model = self.qlinear_test(model, q_config, quantize_params, ['Resize', 'Conv'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        
        q_model = self.qdq_test(model, q_config, quantize_params, ['Resize', 'Conv'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 4)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 3)

        # test opset version 10
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
        model.ir_version = 7 # use stable onnx ir version

        q_model = self.qlinear_test(model, q_config, quantize_params, ['Resize', 'Conv'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)

        q_model = self.qdq_test(model, q_config, quantize_params, ['Resize', 'Conv'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)
        
    def test_argmax(self):
        input_name = "input"
        output_name = "output"
        input_shape = [1, 256, 128, 128]
        output_shape = [1, 32, 128]
        initializers = []

        # make Conv node
        conv_weight_name = "conv_weight"
        conv_weight_arr = np.random.randint(-1, 2, [32, 256, 1, 1]).astype(np.float32)
        conv_weight_initializer = onnx.numpy_helper.from_array(conv_weight_arr, name=conv_weight_name)
        conv_output_name = "conv_output"
        conv_inputs = [input_name, conv_weight_name]
        conv_outputs = [conv_output_name]
        conv_name = "conv_node"
        conv_node = onnx.helper.make_node(
            "Conv",
            conv_inputs,
            conv_outputs,
            dilations=[1, 1],
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            name=conv_name,
        )

        # make ArgMax node
        argmax_inputs = [conv_output_name]
        argmax_outputs = [output_name]
        argmax_name = "argmax_node"
        argmax_node = onnx.helper.make_node(
            "ArgMax",
            argmax_inputs,
            argmax_outputs,
            axis=3,
            keepdims=0,
            name=argmax_name,
        )

        initializers = [conv_weight_initializer]

        # make graph
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.INT64, output_shape)
        graph_name = "ArgMax_Quant_Test"
        graph = helper.make_graph(
            [conv_node, argmax_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version
        q_config = {'conv_node': self.static_q_config,
                    'argmax_node': self.static_q_config}
        quantize_params = {'input': [np.uint8(0), np.float32(10.)],
                           'conv_weight': [np.uint8(0),np.float32(10.)],
                           'conv_output': [np.uint8(0), np.float32(10.)],
                           'output': [np.uint8(0), np.float32(10.)],
                           }
        q_model = self.qlinear_test(model, q_config, quantize_params, ['Conv', 'ArgMax'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)

    def test_gemm(self):
        input_name = "input"
        output_name = "output"
        initializers = []
        weight_shape = [100, 10]
        weight_name = "linear1.weight"
        bias_shape = [100]
        bias_name = "linear1.bias"
        node_name = "gemm"

        weight_data = np.random.normal(0, 0.1, weight_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))

        bias_data = np.random.normal(0, 0.1, bias_shape).astype(np.float32)
        initializers.append(onnx.numpy_helper.from_array(bias_data, name=bias_name))

        gemm1_node =  onnx.helper.make_node(
            "Gemm",
            [input_name, weight_name, bias_name],
            [output_name],
            alpha=1.0,
            beta=1.0,
            transB=1,
            name=node_name
        )

        gemm1_output_name = "gemm1_output"
        input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [-1, 10])
        output_tensor = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [-1, 100])
        graph_name = "gemm_test"
        graph = helper.make_graph(
            [gemm1_node],
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7  # use stable onnx ir version
        q_config = {'gemm': self.static_q_config}
        quantize_params = {'input': [np.uint8(0), np.float32(10.)],
                           'linear1.weight': [np.uint8(0), np.float32(10.)],
                           'linear1.bias': [np.uint8(0), np.float32(10.)],
                           'output': [np.uint8(0), np.float32(10.)],
                           }
        q_model = self.qlinear_test(model, q_config, quantize_params, ['Gemm'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        q_model = self.qdq_test(model, q_config, quantize_params, ['Gemm'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 4)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)

        # test gemm with non-constant bias
        bias_tensor = helper.make_tensor_value_info(bias_name, TensorProto.FLOAT, [100])
        gemm2_node =  onnx.helper.make_node(
            "Gemm",
            [input_name, weight_name, bias_name],
            [output_name],
            alpha=1.0,
            beta=1.0,
            transB=1,
            name=node_name
        )
        initializers = []
        initializers.append(onnx.numpy_helper.from_array(weight_data, name=weight_name))
        graph_name = "gemm_test"
        graph = helper.make_graph(
            [gemm2_node],
            graph_name,
            [input_tensor, bias_tensor],
            [output_tensor],
            initializer=initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        q_model = self.qlinear_test(model, q_config, quantize_params, ['Gemm'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 0)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 0)
        q_model = self.qdq_test(model, q_config, quantize_params, ['Gemm'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)

    def test_embed(self):
        input_ids_shape = [1, 4]
        input_ids_tensor = helper.make_tensor_value_info('input_ids', TensorProto.INT32, input_ids_shape)

        segment_ids_shape = [1, 4]
        segment_ids_tensor = helper.make_tensor_value_info('segment_ids', TensorProto.INT32, segment_ids_shape)

        # EmbedLayerNormalization Node Constants and Weights:
        word_embed_shape = [32, 4]
        word_embed_weights = np.random.random_sample(word_embed_shape).astype(dtype='float32')
        word_embed_initializer = onnx.numpy_helper.from_array(word_embed_weights, name='word_embed')

        pos_embed_shape = [16, 4]
        pos_embed_weights = np.random.random_sample(pos_embed_shape).astype(dtype='float32')
        pos_embed_initializer = onnx.numpy_helper.from_array(pos_embed_weights, name='pos_embed')

        seg_embed_shape = [2, 4]
        seg_embed_weights = np.random.random_sample(seg_embed_shape).astype(dtype='float32')
        seg_embed_initializer = onnx.numpy_helper.from_array(seg_embed_weights, name='seg_embed')

        gamma_shape = [4]
        gamma = np.random.random_sample(gamma_shape).astype(dtype='float32')
        gamma_initializer = onnx.numpy_helper.from_array(gamma, name='gamma')

        beta_shape = [4]
        beta = np.random.random_sample(beta_shape).astype(dtype='float32')
        beta_initializer = onnx.numpy_helper.from_array(beta, name='beta')

        # EmbedLayerNormalization Outputs:
        layernorm_out_shape = [1, 4, 4]
        layernorm_out_tensor = helper.make_tensor_value_info('layernorm_out', TensorProto.FLOAT, layernorm_out_shape)

        mask_index_out_shape = [1]
        mask_index_out_tensor = helper.make_tensor_value_info('mask_index_out', TensorProto.INT32, mask_index_out_shape)

        # EmbedLayerNormalization Node:
        embed_layer_norm_inputs = [
            'input_ids', 'segment_ids', 'word_embed', 'pos_embed', 'seg_embed', 'gamma', 'beta'
        ]
        embed_layer_norm_outputs = ['layernorm_out', 'mask_index_out']
        embed_layer_norm_node = helper.make_node('EmbedLayerNormalization',
                                                 embed_layer_norm_inputs,
                                                 embed_layer_norm_outputs,
                                                 domain='com.microsoft',
                                                 name='Embed')

        # Construct the Graph and Model:
        nodes = [embed_layer_norm_node]
        graph_name = 'embed_layernorm_graph'
        inputs = [input_ids_tensor, segment_ids_tensor]
        outputs = [layernorm_out_tensor, mask_index_out_tensor]
        initializers = [
            word_embed_initializer, pos_embed_initializer, seg_embed_initializer, gamma_initializer, beta_initializer
        ]

        graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)
        model = helper.make_model(graph, 
            opset_imports=[helper.make_opsetid("com.microsoft", 14), helper.make_opsetid("ai.onnx", 14)])
        model.ir_version = 7 # use stable onnx ir version
        
        q_config = {'Embed': self.static_q_config}
        quantize_params = {'word_embed': [np.uint8(10.), np.float32(0)],
                           'pos_embed': [np.uint8(10.), np.float32(0)],
                           'seg_embed': [np.uint8(10.), np.float32(0)],
                           'gamma': [np.uint8(10.), np.float32(0)],
                           'beta': [np.uint8(10.), np.float32(0)],
                           'layernorm_out': [np.uint8(10.), np.float32(0)],
                           'mask_index_out': [np.uint8(10.), np.float32(0)],
                           'input_ids': [np.uint8(10.), np.float32(0)],
                           } 
        q_model = self.qlinear_test(model, q_config, quantize_params, ['EmbedLayerNormalization'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QEmbedLayerNormalization'], 1)
        converter = QOPERATORS["QEmbedLayerNormalization"]([i for i in q_model.nodes() if i.op_type == "QEmbedLayerNormalization"][0],
                None, q_model.initializer())
        done, add_node, init = converter.convert()
        self.assertTrue("EmbedLayerNormalization" in [i.op_type for i in add_node])

        q_model = self.qdq_test(model, q_config, quantize_params, ['EmbedLayerNormalization'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 5)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['EmbedLayerNormalization'], 1)

    def test_LSTM(self):
        input_shape = [1, 1, 200]
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)

        w_shape = [2, 400, 200]
        w_weights = np.random.random_sample(w_shape).astype(dtype='float32')
        w_init = onnx.numpy_helper.from_array(w_weights, name='w')

        r_shape = [2, 400, 100]
        r_weights = np.random.random_sample(r_shape).astype(dtype='float32')
        r_init = onnx.numpy_helper.from_array(r_weights, name='r')

        b_shape = [2, 800]
        b_weights = np.random.random_sample(b_shape).astype(dtype='float32')
        b_init = onnx.numpy_helper.from_array(b_weights, name='b')

        out_shape = [1, 2, 1, 100]
        out_tensor = helper.make_tensor_value_info('out', TensorProto.FLOAT, out_shape)

        kwargs = {}
        kwargs['direction'] = "bidirectional"
        kwargs['activations'] = ["Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"]
        kwargs['hidden_size'] = 100
        kwargs['input_forget'] = 0

        lstm_node = helper.make_node('LSTM',
                                     ['input', 'w', 'r', 'b'],
                                     ['out'],
                                     name='lstm',
                                     domain='',
                                     **kwargs)
        graph = helper.make_graph([lstm_node], 'test', [input_tensor], [out_tensor], initializer=[w_init, r_init, b_init])
        model = helper.make_model(graph, 
            opset_imports=[helper.make_opsetid("", 11)])
        model.ir_version = 7 # use stable onnx ir version
        
        q_config = {'lstm': self.dynamic_q_config}
        q_model = self.dynamic_test(model, q_config, None, ['LSTM'])
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DynamicQuantizeLSTM'], 1)
        
    def test_concat_reshape_pooling(self):
        model = build_model()
 
        q_config = {'Reshape':self.static_q_config, 'conv1':self.static_q_config, 'conv2':self.static_q_config, \
                    'Concat':self.static_q_config, 'AveragePool':self.static_q_config, 'add':self.static_q_config}
        quantize_params = {'input': [np.uint8(10.), np.float32(0)],
                           'conv1_weight': [np.uint8(10.), np.float32(0)],
                           'conv1_output': [np.uint8(10.), np.float32(0)],
                           'conv2_weight': [np.uint8(10.), np.float32(0)],
                           'conv2_output': [np.uint8(10.), np.float32(0)],
                           'concat_output': [np.uint8(10.), np.float32(0)],
                           'avg_output': [np.uint8(10.), np.float32(0)],
                           'add_out': [np.uint8(10.), np.float32(0)],
                           'add_init': [np.uint8(10.), np.float32(0)],
                           'shape': [np.uint8(10.), np.float32(0)],
                           'reshape_output': [np.uint8(10.), np.float32(0)],
                           'add_init_2': [np.uint8(10.), np.float32(0)],
                           'add_out_2': [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ['Reshape', 'Conv', 'Concat', 'AveragePool', 'Add']
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types, **{'dedicated_qdq_pair': True})
        self.assertEqual(len(q_model.model.graph.node), 9)

        q_model.export('test.onnx', ONNXQlinear2QDQConfig())
        export_model = onnx.load('test.onnx')
        self.assertEqual(len(export_model.graph.node), 20)
        os.remove('test.onnx')
 
        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types, **{'dedicated_qdq_pair': True})
        self.assertEqual(len(q_model.model.graph.node), 23)

        q_config = {'Reshape':self.static_q_config, 'conv1':'fp32', 'conv2':self.static_q_config, \
                    'Concat':self.static_q_config, 'AveragePool':self.static_q_config}
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 9)
        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 12)

        q_config = {'Reshape':self.static_q_config, 'conv1':'fp32', 'conv2':'fp32', \
                    'Concat':self.static_q_config, 'AveragePool':self.static_q_config}
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 7)
        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 7)

        q_config = {'Reshape':self.static_q_config, 'conv1':self.static_q_config, 'conv2':self.static_q_config, \
                    'Concat':self.static_q_config, 'AveragePool':'fp32'}
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 9)
        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 17)
 
        quantize_params = {'input': [np.uint8(10.), np.float32(0)],
                           'conv1_weight': [np.uint8(10.), np.float32(0)],
                           'conv1_output': [np.uint8(10.), np.float32(0)],
                           'conv2_weight': [np.uint8(10.), np.float32(0)],
                           'conv2_output': [np.uint8(10.), np.float32(0)],
                           'concat_output': [np.uint8(10.), np.float32(0)],
                           'avg_output': [np.uint8(10.), np.float32(0)],
                           'shape': [np.uint8(10.), np.float32(0)],
                           'add_out': [np.uint8(10.), np.float32(0)],
                           'add_init': [np.uint8(10.), np.float32(0)],
                           'reshape_output': [np.uint8(10.), np.float32(0)]}
        q_config = {'Reshape':self.static_q_config, 'conv1':self.static_q_config, 'conv2':self.static_q_config, \
                    'Concat':self.static_q_config, 'AveragePool':self.static_q_config}
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 9)
        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 21)
 
    def test_conv(self):
        for op in ['Conv', 'FusedConv']:
            A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5, 1])
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 3, 3, 1])
            C = helper.make_tensor('C', TensorProto.FLOAT, [1, 5, 5, 1], np.random.random((1, 5 ,5, 1)).reshape(25).tolist())
            D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 1])
            conv_node = onnx.helper.make_node(op, ['A', 'B', 'C'], ['D'], 
                                              name=op, 
                                              kernel_shape=[3, 3], 
                                              pads=[1, 1, 1, 1])
            initializers = [C]
            graph = helper.make_graph([conv_node], 'test_graph_1', [A, B], [D], initializer=initializers)
            model = helper.make_model(graph)
            q_config = {op: self.static_q_config}
            quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                               "B": [np.uint8(10.), np.float32(0)],
                               "C": [np.uint8(10.), np.float32(0)],
                               "D": [np.uint8(10.), np.float32(0)]}       
            quantizable_op_types = [op]
            q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)
            q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 4)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 3)

    def test_matmul(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B_init = helper.make_tensor('B', TensorProto.FLOAT, [1, 1, 5, 1], np.random.random((1, 1, 5, 1)).reshape(5).tolist())
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 1])
        matmul_node = onnx.helper.make_node('MatMul', ['A', 'B'], ['C'], name='Matmul')
        graph = helper.make_graph([matmul_node], 'test_graph_1', [A], [C], [B_init])
        model = helper.make_model(graph)
        q_config = {"Matmul": self.static_q_config}
        quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                           "B": [np.uint8(10.), np.float32(0)],
                           "C": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ["Matmul"]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)

        q_config = {"Matmul": self.dynamic_q_config}
        q_model = self.dynamic_test(model, q_config, None, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DynamicQuantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['MatMulInteger'], 1)

        quantize_params = {"A": [np.float32(10.)],
                           "B": [np.float32(10.)],
                           "C": [np.float32(10.)]}
        with self.assertRaises(ValueError):
            self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        with self.assertRaises(ValueError):
            self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
 
        q_config = {"Matmul": {"weight":{'dtype': 3,
                               'algorithm': 'minmax',
                               'scheme':'sym',
                               'granularity': 'per_tensor'},
                     'activation':{'dtype': 2,
                                   'algorithm': 'minmax',
                                   'scheme':'asym',
                                   'granularity':'per_tensor',
                                   'quant_mode': 'dynamic'}}}
        quantize_params = {}
        q_model = self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DynamicQuantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['MatMulInteger'], 1)

    def test_attention(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 5, 5])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 1, 5, 5])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
        node = onnx.helper.make_node('Attention', ['A', 'B', 'C'], ['D'], name='Attention')
        graph = helper.make_graph([node], 'test_graph_1', [A, B, C], [D])
        model = helper.make_model(graph)
        q_config = {"Attention": self.static_q_config}
        quantize_params = {"A": [np.uint8(0), np.float32(0.5)],
                           "B": [np.uint8(0), np.float32(0.5)],
                           "C": [np.uint8(0), np.float32(0.5)],
                           "D": [np.uint8(0), np.float32(0.5)]}
        quantizable_op_types = ["Attention"]

        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QAttention'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)

        converter = QOPERATORS["QAttention"]([i for i in q_model.nodes() if i.op_type == "QAttention"][0],
                None, q_model.initializer())
        done, add_node, init = converter.convert()
        self.assertTrue("Attention" in [i.op_type for i in add_node])

        self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        q_config = {"Attention": self.dynamic_q_config}
        q_model = self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DynamicQuantizeLinear'], 3)

        E = helper.make_tensor_value_info('E', TensorProto.INT32, [1, 1, 5, 5])
        F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [1, 1, 5, 5])
        node = onnx.helper.make_node('Attention', ['A', 'B', 'C', 'F', 'E'], ['D'], name='Attention')
        graph = helper.make_graph([node], 'test_graph_1', [A, B, C, F, E], [D])
        model = helper.make_model(graph)
        q_config = {"Attention": self.static_q_config}
        quantize_params = {"A": [np.uint8(0), np.float32(0.5)],
                           "B": [np.uint8(0), np.float32(0.5)],
                           "C": [np.uint8(0), np.float32(0.5)],
                           "D": [np.uint8(0), np.float32(0.5)]}
        quantizable_op_types = ["Attention"]

        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 3)

        q_config = {"Attention": self.dynamic_q_config}
        q_model = self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DynamicQuantizeLinear'], 3)

    def test_gather(self):
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2])

        matmul_weight = helper.make_tensor('matmul_weight', TensorProto.FLOAT, [2, 3], np.random.random((2, 3)).reshape(6).tolist())
        matmul_output = helper.make_tensor_value_info('matmul_output', TensorProto.FLOAT, [3, 3])
        matmul_node = onnx.helper.make_node('MatMul', ['input', 'matmul_weight'], ['matmul_output'], name='MatMul')

        gather_indices = helper.make_tensor('gather_indices', TensorProto.INT64, [1, 2], [0, 2])
        gather_output = helper.make_tensor_value_info('gather_output', TensorProto.FLOAT, [1, 2, 3])
        gather_node = onnx.helper.make_node('Gather', ['matmul_output', 'gather_indices'], ['gather_output'], name='Gather')

        initializers = [matmul_weight, gather_indices]
        graph = helper.make_graph([matmul_node, gather_node], 
                                  'TestGather_test_model',
                                  [input_tensor], [gather_output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        
        q_config = {'Gather': self.static_q_config, "MatMul": self.static_q_config} 
        quantize_params = {"input": [np.uint8(10.), np.float32(0)],
                           "matmul_weight": [np.uint8(10.), np.float32(0)],
                           "matmul_output": [np.uint8(10.), np.float32(0)],
                           "gather_output": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ["Gather", "MatMul"]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 4)

        q_config = {'Gather': self.dynamic_q_config, "MatMul": self.dynamic_q_config} 
        q_model = self.dynamic_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 6)

    def test_split(self):
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [100, 2])
        e_value = np.random.randn(2, 2).astype(np.float32)
        E_init = helper.make_tensor('E', TensorProto.FLOAT, [2, 2],
                                    e_value.reshape(4).tolist())
 
        matmul_node = onnx.helper.make_node('MatMul', ['D', 'E'], ['A'], name='Matmul')
 
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [50, 2])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [50, 2])
        node = onnx.helper.make_node('Split', ['A'], ['B', 'C'], name='Split', **{'num_outputs': 2})
        graph = helper.make_graph([matmul_node, node], 'test_graph_1', [D], [B, C], [E_init])
        model = helper.make_model(graph)
        q_config = {'Split': {"weight":{'dtype': 3,
                                         'algorithm': 'minmax',
                                         'scheme':'sym',
                                         'granularity': 'per_tensor'},
                              'activation':{'dtype': 2,
                                         'algorithm': 'minmax',
                                         'scheme':'asym',
                                         'granularity':'per_tensor',
                                         'quant_mode': 'static'}
                        },
                    'Matmul': {"weight":{'dtype': 3,
                                         'algorithm': 'minmax',
                                         'scheme':'sym',
                                         'granularity': 'per_tensor'},
                              'activation':{'dtype': 2,
                                         'algorithm': 'minmax',
                                         'scheme':'asym',
                                         'granularity':'per_tensor',
                                         'quant_mode': 'static'}
                        },
 
                    }
        quantize_params = {"A": [np.uint8(0), np.float32(0.5)],
                           "B": [np.uint8(0), np.float32(0.5)],
                           "C": [np.uint8(0), np.float32(0.5)],
                           "D": [np.uint8(0), np.float32(0.5)],
                           "E": [np.uint8(0), np.float32(0.5)],
                           }
        quantizable_op_types = ["Split", "MatMul"]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 2)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)

        q_model.export('test.onnx', ONNXQlinear2QDQConfig())
        export_model = onnx.load('test.onnx')
        self.assertEqual(len(export_model.graph.node), 11)
        os.remove('test.onnx')

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 5)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 4)
 
    def test_pad(self):
        b_value = np.array([0, 1, 1, 0, 1, 1]).astype(np.int64)
        B_init = helper.make_tensor('B', TensorProto.INT64, [6],
                                    b_value.reshape(6).tolist())
        B = helper.make_tensor_value_info('B', TensorProto.INT64, [6])
        C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 7, 7])

        d_value = np.random.randn(1).astype(np.float32)
        D_init = helper.make_tensor('D', TensorProto.FLOAT, [1],
                                    d_value.reshape(1).tolist())
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1])

        e_value = np.random.randn(1, 5, 5).astype(np.float32)
        E_init = helper.make_tensor('E', TensorProto.FLOAT, [1, 1, 5, 5],
                                    e_value.reshape(25).tolist())
        E = helper.make_tensor_value_info('E', TensorProto.FLOAT, [1, 1, 5, 5])
        f_value = np.random.randn(1, 3, 3).astype(np.float32)
        F_init = helper.make_tensor('F', TensorProto.FLOAT, [1, 1, 3, 3],
                                    f_value.reshape(9).tolist())
        F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [1, 1, 3, 3])
        for mode in ["constant", "edge", "reflect", "constant_value", "constant_value_wo_init"]:
            conv_node = onnx.helper.make_node('Conv', ['E', 'F'], ['A'], 
                                              name='Conv', 
                                              kernel=[3, 3],
                                              padding=[1, 1, 1, 1])
            if mode == "constant_value":
                node = onnx.helper.make_node('Pad', ['A', 'B', 'D'], ['C'], name='Pad', mode="constant")
                graph = helper.make_graph([conv_node, node], 'test_graph_1', [E, F, B, D], [C], [E_init, F_init, B_init, D_init])
            elif mode == "constant_value_wo_init":
                node = onnx.helper.make_node('Pad', ['A', 'B', 'D'], ['C'], name='Pad', mode="constant")
                graph = helper.make_graph([conv_node, node], 'test_graph_1', [E, F, B, D], [C], [E_init, F_init, B_init])
            else:
                node = onnx.helper.make_node('Pad', ['A', 'B'], ['C'], name='Pad', mode=mode)
                graph = helper.make_graph([conv_node, node], 'test_graph_1', [E, F, B], [C], [E_init, F_init, B_init])
            model = helper.make_model(graph)
            pad_config = {"weight":{'dtype': 3,
                                    'algorithm': 'minmax',
                                    'scheme':'sym',
                                    'granularity': 'per_tensor'},
                         'activation':{'dtype': 2,
                                    'algorithm': 'minmax',
                                    'scheme':'asym',
                                    'granularity':'per_tensor',
                                    'quant_mode': 'static'}}
            conv_config = {"weight":{'dtype': 3,
                                    'algorithm': 'minmax',
                                    'scheme':'sym',
                                    'granularity': 'per_channel'},
                         'activation':{'dtype': 2,
                                    'algorithm': 'minmax',
                                    'scheme':'asym',
                                    'granularity':'per_tensor',
                                    'quant_mode': 'static'}}
            q_config = {'Conv': conv_config,
                        'Pad': pad_config}
            quantize_params = {"A": [np.uint8(10.), np.float32(1)],
                               "C": [np.uint8(10.), np.float32(1)],
                               "D": [np.uint8(10.), np.float32(1)],
                               "E": [np.uint8(10.), np.float32(1)],
                               "F": [np.uint8(10.), np.float32(1)]}
            quantizable_op_types = ["Conv", "Pad"]
            q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
            q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types, **{'dedicated_qdq_pair': True})
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 4)

        node = onnx.helper.make_node('Pad', ['E', 'B', 'D'], ['C'], name='Pad', mode="constant")
        graph = helper.make_graph([node], 'test_graph_1', [E, B, D], [C], [E_init, B_init, D_init])
        model = helper.make_model(graph)
        q_config = {'Pad': {'activation':{'dtype': 2,
                                         'algorithm': 'minmax',
                                         'scheme':'asym',
                                         'granularity':'per_tensor',
                                         'quant_mode': 'static'}
                  }}
        quantize_params = {"C": [np.uint8(10.), np.float32(0)],
                           "E": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ["Pad"]
        self.qlinear_test(model, pad_config, quantize_params, quantizable_op_types)
        self.qdq_test(model, pad_config, quantize_params, quantizable_op_types)

    def test_binary(self):
        for op in ['Mul', 'Add']:
            A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 10])
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1])
            C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 10])
            node = onnx.helper.make_node(op, ['A', 'B'], ['C'], name=op)
            graph = helper.make_graph([node], 'test_graph_1', [A, B], [C])
            model = helper.make_model(graph)
            q_config = {op: self.static_q_config}
            quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                               "B": [np.uint8(10.), np.float32(0)],
                               "C": [np.uint8(10.), np.float32(0)]}
            quantizable_op_types = [op]
            self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            self.qlinear_test(model, q_config, {}, quantizable_op_types)
            self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
            self.qdq_test(model, q_config, {}, quantizable_op_types)
    
    def test_relu(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 5, 5])
        E = helper.make_tensor_value_info('E', TensorProto.FLOAT, [1, 1, 5, 5])
        F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [1, 1, 5, 5])
  
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'], 
                                          name='Conv', 
                                          kernel_shape=[3, 3], 
                                          pads=[1, 1, 1, 1])
        relu_node = onnx.helper.make_node('Relu', ['C'], ['D'], name='Relu')
        add_node = onnx.helper.make_node('Add', ['D', 'E'], ['F'], name='Add')
        graph = helper.make_graph([conv_node, relu_node], 'test_graph_1', [A, B], [D])
        model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "./onnxrt_test/optimized_model.onnx"  
        session = ort.InferenceSession(model.SerializeToString(), sess_options, providers=ort.get_available_providers())
        tmp_model = onnx.load(sess_options.optimized_model_filepath)
 
        q_config = {"Conv": self.static_q_config, "Relu": self.static_q_config}
        quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                           "B": [np.uint8(10.), np.float32(0)],
                           "C": [np.uint8(10.), np.float32(0)],
                           "D": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ["Conv", "Relu"]
        q_model = self.qlinear_test(tmp_model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 4)
        q_model = self.qdq_test(tmp_model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 7)
        
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(model.SerializeToString(), sess_options, providers=ort.get_available_providers())
        tmp_model = onnx.load(sess_options.optimized_model_filepath)
        q_model = self.qlinear_test(tmp_model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 5)
        q_model = self.qdq_test(tmp_model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 8)

        graph = helper.make_graph([conv_node, relu_node, add_node], 'test_graph_2', [A, B, E], [F])
        model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(model.SerializeToString(), sess_options, providers=ort.get_available_providers())
        tmp_model = onnx.load(sess_options.optimized_model_filepath)
        q_model = self.qlinear_test(tmp_model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 5)
        q_model = self.qdq_test(tmp_model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(len(q_model.model.graph.node), 8)

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
        model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = "./onnxrt_test/optimized_model.onnx"
        session = ort.InferenceSession(model.SerializeToString(), sess_options, providers=ort.get_available_providers())
        model = onnx.load(sess_options.optimized_model_filepath)

        q_config = {"Conv": self.static_q_config, "Clip": self.static_q_config}
        quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                           "B": [np.uint8(10.), np.float32(0)],
                           "C": [np.uint8(10.), np.float32(0)],
                           "D": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ["Conv", "Clip"]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)
        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 3)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 3)

    def test_activation(self):
        for op in ["Relu", "LeakyRelu", "Sigmoid"]:
            B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 10])
            A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 10])
            node = onnx.helper.make_node(op, ['A'], ['B'], name=op)
            graph = helper.make_graph([node], 'test_graph_1', [A], [B])
            model = helper.make_model(graph)
            q_config = {op: self.static_q_config}
            quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                               "B": [np.uint8(10.), np.float32(0)]}
            quantizable_op_types = [op]
            self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            self.qdq_test(model, q_config, quantize_params, quantizable_op_types)

            a_value = np.random.randn(1, 10).astype(np.float32)
            A_init = helper.make_tensor('A', TensorProto.FLOAT, [1, 10],
                                        a_value.reshape(10).tolist())
            graph = helper.make_graph([node], 'test_graph_1', [A], [B], [A_init])
            model = helper.make_model(graph)
            self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
            self.qlinear_test(model, q_config, {}, quantizable_op_types)
            self.qdq_test(model, q_config, {}, quantizable_op_types)

    def test_pooling(self):
        op = "MaxPool"
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 5, 5, 1])
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5, 1])
        node = onnx.helper.make_node(op, ['A'], ['B'], 
                                     name=op,
                                     kernel_shape=[3, 3],
                                     pads=[1, 1, 1, 1])
        graph = helper.make_graph([node], 'test_graph_1', [A], [B]) 
        q_config = {op: self.static_q_config}
        quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                           "B": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = [op]
        for opset_version in [12, 13]:
            opset = onnx.OperatorSetIdProto()
            opset.version = opset_version
            model = helper.make_model(graph, opset_imports=[opset])
            self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            self.qdq_test(model, q_config, quantize_params, quantizable_op_types)

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
 
        q_config = {"Conv": self.static_q_config, op: self.static_q_config}
        quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                           "B": [np.uint8(10.), np.float32(0)],
                           "C": [np.uint8(10.), np.float32(0)],
                           "D": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ["Conv", op]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)
        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 4)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 4)

        op = "GlobalAveragePool"
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 5, 1, 1])
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5, 1])
        node = onnx.helper.make_node(op, ['A'], ['B'], 
                                     name=op,
                                     kernel_shape=[3, 3],
                                     pads=[1, 1, 1, 1])
        graph = helper.make_graph([node], 'test_graph_1', [A], [B]) 
        q_config = {op: self.static_q_config}
        quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                           "B": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = [op]
        for opset_version in [12, 13]:
            opset = onnx.OperatorSetIdProto()
            opset.version = opset_version
            model = helper.make_model(graph, opset_imports=[opset])
            q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
            q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 2)
            self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)

        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 1, 5, 5])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [1, 1, 3, 3])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 1, 1])
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'],
                                          name='Conv',
                                          kernel_shape=[3, 3],
                                          pads=[1, 1, 1, 1])
        pool_node = onnx.helper.make_node(op, ['C'], ['D'], name=op)
        graph = helper.make_graph([conv_node, pool_node], 'test_graph_1', [A, B], [D])
        model = helper.make_model(graph)
 
        q_config = {"Conv": self.static_q_config, op: self.static_q_config}
        quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                           "B": [np.uint8(10.), np.float32(0)],
                           "C": [np.uint8(10.), np.float32(0)],
                           "D": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ["Conv", op]
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 2)

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 4)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 4)


    def test_exclude_node(self):
        A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5, 1])
        B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3, 3, 1, 1])
        D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 1, 3, 3])
        conv_node = onnx.helper.make_node('Conv', ['A', 'B'], ['C'],
                                          name='Conv',
                                          kernel_shape=[3, 3],
                                          pads=[1, 1, 1, 1])
        pool_node = onnx.helper.make_node("MaxPool", ['C'], ['D'], name="MaxPool")
        graph = helper.make_graph([conv_node, pool_node], 'test_graph_1', [A, B], [D])
        model = helper.make_model(graph)

        q_config = {"Conv": self.static_q_config, "MaxPool": "fp32"}
        quantize_params = {"A": [np.uint8(10.), np.float32(0)],
                           "B": [np.uint8(10.), np.float32(0)],
                           "C": [np.uint8(10.), np.float32(0)],
                           "D": [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ["Conv", "MaxPool"]
        self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
    
    def test_more_direct8bit_nodes(self):
        # test direct q8 nodes: MatMul-Flatten-Abs-Sign-ShrinK-MatMul
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 32])

        matmul1_weight = helper.make_tensor('matmul1_weight', TensorProto.FLOAT, [32, 64], np.random.random((32, 64)).reshape(2048).tolist())
        matmul1_output = helper.make_tensor_value_info('matmul1_output', TensorProto.FLOAT, [1, 64])
        matmul1_node = onnx.helper.make_node('MatMul', ['input', 'matmul1_weight'], ['matmul1_output'], name='Matmul_0')

        flatten_output = helper.make_tensor_value_info('flatten_output', TensorProto.FLOAT, [1, 64])
        flatten_node = onnx.helper.make_node("Flatten", inputs=["matmul1_output"], outputs=["flatten_output"], axis=1, name='Flatten_1')

        abs_output = helper.make_tensor_value_info('abs_output', TensorProto.FLOAT, [1, 64])
        abs_node = onnx.helper.make_node("Abs", inputs=["flatten_output"], outputs=["abs_output"], name='Abs_2')

        sign_output = helper.make_tensor_value_info('sign_output', TensorProto.FLOAT, [1, 64])
        sign_node = onnx.helper.make_node("Sign", inputs=["abs_output"], outputs=["sign_output"], name="Sign_3")

        shrink_output = helper.make_tensor_value_info('shrink_output', TensorProto.FLOAT, [1, 64])
        shrink_node = onnx.helper.make_node("Shrink", inputs=["sign_output"], outputs=["shrink_output"], name="Shrink_4")

        matmul2_weight = helper.make_tensor('matmul2_weight', TensorProto.FLOAT, [64, 2], np.random.random((64, 2)).reshape(128).tolist())
        matmul2_output = helper.make_tensor_value_info('matmul2_output', TensorProto.FLOAT, [1, 2])
        matmul2_node = onnx.helper.make_node('MatMul', ['shrink_output', 'matmul2_weight'], ['matmul2_output'], name='Matmul_5')

        initializers = [matmul1_weight, matmul2_weight]
        graph = helper.make_graph([matmul1_node, flatten_node, abs_node, sign_node, shrink_node, matmul2_node], 
                                  'TestMoreDirect8_test_model',
                                  [input_tensor], [matmul2_output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7

        q_config = {'Matmul_0':self.static_q_config, 'Flatten_1':self.static_q_config, 'Abs_2':self.static_q_config, \
            'Sign_3':self.static_q_config, 'Shrink_4':self.static_q_config, 'Matmul_5':self.static_q_config}
        quantize_params = {'input': [np.uint8(10.), np.float32(0)],
                           'matmul1_weight': [np.uint8(10.), np.float32(0)],
                           'matmul1_output': [np.uint8(10.), np.float32(0)],
                           'flatten_output': [np.uint8(10.), np.float32(0)],
                           'abs_output': [np.uint8(10.), np.float32(0)],
                           'sign_output': [np.uint8(10.), np.float32(0)],
                           'shrink_output': [np.uint8(10.), np.float32(0)],
                           'matmul2_weight': [np.uint8(10.), np.float32(0)],
                           'matmul2_output': [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ['MatMul', 'Flatten', 'Abs', 'Sign', 'Shrink']
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)
        

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        q_model.save('qdq.onnx')
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 9)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 7)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)


    def test_expand(self):
        # test expand nodes: MatMul-Expand-MatMul
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2])

        matmul1_weight = helper.make_tensor('matmul1_weight', TensorProto.FLOAT, [2, 1], np.random.random((2, 1)).reshape(2).tolist())
        matmul1_output = helper.make_tensor_value_info('matmul1_output', TensorProto.FLOAT, [3, 1])
        matmul1_node = onnx.helper.make_node('MatMul', ['input', 'matmul1_weight'], ['matmul1_output'], name='Matmul_0')

        expand_new_shape = helper.make_tensor('expand_new_shape', TensorProto.INT64, [2], [3, 4])
        expand_output = helper.make_tensor_value_info('expand_output', TensorProto.FLOAT, [3, 4])
        expand_node = onnx.helper.make_node('Expand', ['matmul1_output', 'expand_new_shape'], ['expand_output'], name='Expand_1')

        matmul2_weight = helper.make_tensor('matmul2_weight', TensorProto.FLOAT, [4, 2], np.random.random((4, 2)).reshape(8).tolist())
        matmul2_output = helper.make_tensor_value_info('matmul2_output', TensorProto.FLOAT, [3, 2])
        matmul2_node = onnx.helper.make_node('MatMul', ['expand_output', 'matmul2_weight'], ['matmul2_output'], name='Matmul_2')

        initializers = [matmul1_weight, matmul2_weight, expand_new_shape]
        graph = helper.make_graph([matmul1_node, expand_node, matmul2_node], 
                                  'TestExpand_test_model',
                                  [input_tensor], [matmul2_output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        
        q_config = {'Matmul_0':self.static_q_config, 'Expand_1':self.static_q_config, 'Matmul_2':self.static_q_config}
        quantize_params = {'input': [np.uint8(10.), np.float32(0)],
                           'matmul1_weight': [np.uint8(10.), np.float32(0)],
                           'matmul1_output': [np.uint8(10.), np.float32(0)],
                           'matmul2_weight': [np.uint8(10.), np.float32(0)],
                           'matmul2_output': [np.uint8(10.), np.float32(0)],
                           'expand_output': [np.uint8(10.), np.float32(0)],
                           }
        quantizable_op_types = ['MatMul', 'Expand']
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 6)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 4)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)

    def test_slice(self):
        # test slice nodes: MatMul-Slice-MatMul
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [5, 4, 1])

        matmul1_weight = helper.make_tensor('matmul1_weight', TensorProto.FLOAT, [1, 3], np.random.random((1, 3)).reshape(3).tolist())
        matmul1_output = helper.make_tensor_value_info('matmul1_output', TensorProto.FLOAT, [5, 4, 3])
        matmul1_node = onnx.helper.make_node('MatMul', ['input', 'matmul1_weight'], ['matmul1_output'], name='Matmul_0')

        slice_starts = helper.make_tensor('slice_starts', TensorProto.INT64, [2], [0, 0])
        slice_ends = helper.make_tensor('slice_ends', TensorProto.INT64, [2], [3, 4])
        slice_axes = helper.make_tensor('slice_axes', TensorProto.INT64, [2], [0, 1])
        slice_steps = helper.make_tensor('slice_steps', TensorProto.INT64, [2], [1, 1])
        slice_output = helper.make_tensor_value_info('slice_output', TensorProto.FLOAT, [3, 4, 3])
        slice_node = onnx.helper.make_node('Slice', 
                                            ['matmul1_output', 'slice_starts', 'slice_ends', 'slice_axes', 'slice_steps'], 
                                            ['slice_output'], name='Slice_1')

        matmul2_weight = helper.make_tensor('matmul2_weight', TensorProto.FLOAT, [3, 2], np.random.random((3, 2)).reshape(6).tolist())
        matmul2_output = helper.make_tensor_value_info('matmul2_output', TensorProto.FLOAT, [3, 4, 2])
        matmul2_node = onnx.helper.make_node('MatMul', ['slice_output', 'matmul2_weight'], ['matmul2_output'], name='Matmul_2')

        initializers = [matmul1_weight, matmul2_weight, slice_starts, slice_ends, slice_axes, slice_steps]
        graph = helper.make_graph([matmul1_node, slice_node, matmul2_node], 
                                  'TestSlice_test_model',
                                  [input_tensor], [matmul2_output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7

        q_config = {'Matmul_0':self.static_q_config, 'Slice_1':self.static_q_config, 'Matmul_2':self.static_q_config}
        quantize_params = {'input': [np.uint8(10.), np.float32(0)],
                           'matmul1_weight': [np.uint8(10.), np.float32(0)],
                           'matmul1_output': [np.uint8(10.), np.float32(0)],
                           'matmul2_weight': [np.uint8(10.), np.float32(0)],
                           'matmul2_output': [np.uint8(10.), np.float32(0)],
                           'slice_output': [np.uint8(10.), np.float32(0)],
                           }
        quantizable_op_types = ['MatMul', 'Slice']
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 6)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 4)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)

    def test_mod(self):
        # test mode nodes: MatMul-Mod-MatMul
        #                  MatMul-/
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

        matmul1_weight = helper.make_tensor('matmul1_weight', TensorProto.FLOAT, [3, 4], np.random.random((3, 4)).reshape(12).tolist())
        matmul1_output = helper.make_tensor_value_info('matmul1_output', TensorProto.FLOAT, [2, 4])
        matmul1_node = onnx.helper.make_node('MatMul', ['input', 'matmul1_weight'], ['matmul1_output'], name='Matmul_0')

        matmul2_weight = helper.make_tensor('matmul2_weight', TensorProto.FLOAT, [3, 4], np.random.random((3, 4)).reshape(12).tolist())
        matmul2_output = helper.make_tensor_value_info('matmul2_output', TensorProto.FLOAT, [2, 4])
        matmul2_node = onnx.helper.make_node('MatMul', ['input', 'matmul2_weight'], ['matmul2_output'], name='Matmul_1')

        mod_output = helper.make_tensor_value_info('mod_output', TensorProto.FLOAT, [2, 4])
        mod_node = onnx.helper.make_node('Mod', ['matmul1_output', 'matmul2_output'], ['mod_output'], name='Mod_2')

        matmul3_weight = helper.make_tensor('matmul3_weight', TensorProto.FLOAT, [4, 2], np.random.random((4, 2)).reshape(8).tolist())
        matmul3_output = helper.make_tensor_value_info('matmul3_output', TensorProto.FLOAT, [2, 2])
        matmul3_node = onnx.helper.make_node('MatMul', ['mod_output', 'matmul3_weight'], ['matmul3_output'], name='Matmul_3')

        initializers = [matmul1_weight, matmul2_weight, matmul3_weight]
        graph = helper.make_graph([matmul1_node, matmul2_node, mod_node, matmul3_node], 
                                  'TestMod_test_model',
                                  [input_tensor], [matmul3_output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 7

        q_config = {'Matmul_0':self.static_q_config, 'Matmul_1':self.static_q_config, 'Mod_2':self.static_q_config,
                    'Matmul_3':self.static_q_config}
        quantize_params = {'input': [np.uint8(10.), np.float32(0)],
                           'matmul1_weight': [np.uint8(10.), np.float32(0)],
                           'matmul1_output': [np.uint8(10.), np.float32(0)],
                           'matmul2_weight': [np.uint8(10.), np.float32(0)],
                           'matmul2_output': [np.uint8(10.), np.float32(0)],
                           'mod_output': [np.uint8(10.), np.float32(0)],
                           'matmul3_weight': [np.uint8(10.), np.float32(0)],
                           'matmul3_output': [np.uint8(10.), np.float32(0)],
                           }
        quantizable_op_types = ['MatMul', 'Mod']
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)


        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 8)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 5)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)

    def test_reducemin_reducemax(self):
        # MatMul-ReduceMin-MatMul
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 3])

        matmul1_weight = helper.make_tensor('matmul1_weight', TensorProto.FLOAT, [3, 2], np.random.random((3, 2)).reshape(6).tolist())
        matmul1_output = helper.make_tensor_value_info('matmul1_output', TensorProto.FLOAT, [3, 2, 2])
        matmul1_node = onnx.helper.make_node('MatMul', ['input', 'matmul1_weight'], ['matmul1_output'], name='Matmul_0')

        reducemin_output = helper.make_tensor_value_info('reducemin_output', TensorProto.FLOAT, [3, 1, 2])
        reducemin_node = onnx.helper.make_node("ReduceMin", inputs=['matmul1_output'], 
                                               outputs=['reducemin_output'], axes=[1], keepdims=1, name='Reducemin_1')
        
        matmul2_weight = helper.make_tensor('matmul2_weight', TensorProto.FLOAT, [2, 3], np.random.random((2, 3)).reshape(6).tolist())
        matmul2_output = helper.make_tensor_value_info('matmul2_output', TensorProto.FLOAT, [3, 1, 3])
        matmul2_node = onnx.helper.make_node('MatMul', ['reducemin_output', 'matmul2_weight'], ['matmul2_output'], name='Matmul_2')

        initializers = [matmul1_weight, matmul2_weight]
        graph = helper.make_graph([matmul1_node, reducemin_node, matmul2_node], 
                                  'TestReduceMin_test_model',
                                  [input_tensor], [matmul2_output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7

        q_config = {'Matmul_0':self.static_q_config, 'Reducemin_1':self.static_q_config, 'Matmul_2':self.static_q_config}
        quantize_params = {'input': [np.uint8(10.), np.float32(0)],
                           'matmul1_weight': [np.uint8(10.), np.float32(0)],
                           'matmul1_output': [np.uint8(10.), np.float32(0)],
                           'reducemin_output': [np.uint8(10.), np.float32(0)],
                           'matmul2_weight': [np.uint8(10.), np.float32(0)],
                           'matmul2_output': [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ['MatMul', 'ReduceMin']
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 6)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 4)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        self.assertIsNotNone(session)

        # MatMul-ReduceMax-MatMul
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2, 3])

        matmul1_weight = helper.make_tensor('matmul1_weight', TensorProto.FLOAT, [3, 2], np.random.random((3, 2)).reshape(6).tolist())
        matmul1_output = helper.make_tensor_value_info('matmul1_output', TensorProto.FLOAT, [3, 2, 2])
        matmul1_node = onnx.helper.make_node('MatMul', ['input', 'matmul1_weight'], ['matmul1_output'], name='Matmul_0')

        reducemax_output = helper.make_tensor_value_info('reducemax_output', TensorProto.FLOAT, [3, 1, 2])
        reducemax_node = onnx.helper.make_node("ReduceMax", inputs=['matmul1_output'], 
                                               outputs=['reducemax_output'], axes=[1], keepdims=1, name='Reducemax_1')
        
        matmul2_weight = helper.make_tensor('matmul2_weight', TensorProto.FLOAT, [2, 3], np.random.random((2, 3)).reshape(6).tolist())
        matmul2_output = helper.make_tensor_value_info('matmul2_output', TensorProto.FLOAT, [3, 1, 3])
        matmul2_node = onnx.helper.make_node('MatMul', ['reducemax_output', 'matmul2_weight'], ['matmul2_output'], name='Matmul_2')

        initializers = [matmul1_weight, matmul2_weight]
        graph = helper.make_graph([matmul1_node, reducemax_node, matmul2_node], 
                                  'TestReduceMax_test_model',
                                  [input_tensor], [matmul2_output], initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7

        q_config = {'Matmul_0':self.static_q_config, 'Reducemax_1':self.static_q_config, 'Matmul_2':self.static_q_config}
        quantize_params = {'input': [np.uint8(10.), np.float32(0)],
                           'matmul1_weight': [np.uint8(10.), np.float32(0)],
                           'matmul1_output': [np.uint8(10.), np.float32(0)],
                           'reducemax_output': [np.uint8(10.), np.float32(0)],
                           'matmul2_weight': [np.uint8(10.), np.float32(0)],
                           'matmul2_output': [np.uint8(10.), np.float32(0)]}
        quantizable_op_types = ['MatMul', 'ReduceMax']
        q_model = self.qlinear_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 1)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 1)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])

        q_model = self.qdq_test(model, q_config, quantize_params, quantizable_op_types)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['DequantizeLinear'], 6)
        self.assertEqual(Counter([node.op_type for node in q_model.model.graph.node])['QuantizeLinear'], 4)
        session = ort.InferenceSession(q_model.model.SerializeToString(), providers=['CPUExecutionProvider'])
        
class TestCastONNXRT(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def build_model(self, inps, outs, weights, node_infos):
        inputs = []
        for name, type, shape in inps:
            inputs.append(helper.make_tensor_value_info(name, type, shape))
        outputs = []
        for name, type, shape in outs:
            outputs.append(helper.make_tensor_value_info(name, type, shape))
        inits = []
        if len(weights) > 0:
            for name, type, shape, data in weights:
                inits.append(helper.make_tensor(name, type, shape, data.tolist()))
        nodes = []
        if len(node_infos) > 0 and len(node_infos[0]) == 4:
            for name, in_name, out_name, type in node_infos:
                nodes.append(onnx.helper.make_node(type, in_name, out_name, name=name))
        if len(node_infos) > 0 and len(node_infos[0]) == 5:
            for name, in_name, out_name, type, domain in node_infos:
                nodes.append(onnx.helper.make_node(type, in_name, out_name, name=name, domain=domain))

        graph = helper.make_graph(nodes, 'test', inputs, outputs, inits)
        model = helper.make_model(graph)
        model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 15)]})
        return model

    def build_test_data(self, names, shapes, dtypes):
        ort_input = {}
        for name, shape, dtype in zip(names, shapes, dtypes):
            ort_input[name] = np.random.random(shape).astype(dtype)
        return ort_input

    def get_fp16_mixed_precision_model(self, model):
        from neural_compressor import MixedPrecisionConfig
        from neural_compressor.mix_precision import fit
        config = MixedPrecisionConfig(backend='onnxrt_cuda_ep', device='gpu', precision='fp16')
        converted_model = fit(model, config)
        return converted_model

    @unittest.skipIf('CUDAExecutionProvider' not in ort.get_all_providers(),
        "skip since CUDAExecutionProvider is not supported")
    def test_fp16(self):
        optypes = ['Sum', 'Sub', 'Div', 'Pow', 'Add']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,2)]]
            weights = [['input2', TensorProto.FLOAT, (1,2), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.BOOL, (1,2)]]
            weights = [['input2', TensorProto.FLOAT, (1,2), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['Abs', 'Exp', 'Log', 'Round', 'Sqrt', 'Softmax', 'Exp', 'Tanh', 'Sigmoid', 'LeakyRelu', 'Round']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,2)]]
            node_infos = [['test', ['input1'], ['output'], optype]]
            model = self.build_model(inps, outs, [], node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['ReduceMean', 'ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp', 'ReduceMax', 'ReduceProd', \
                   'ReduceSum', 'ReduceSumSquare']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,1)]]
            node_infos = [['test', ['input1'], ['output'], optype]]
            model = self.build_model(inps, outs, [], node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['Gelu']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,2)]]
            node_infos = [['test', ['input1'], ['output'], optype, 'com.microsoft']]
            model = self.build_model(inps, outs, [], node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['BiasGelu', 'FastGelu']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, [2]]]
            outs = [['output', TensorProto.FLOAT, [2]]]
            weights = [['input2', TensorProto.FLOAT, [2], np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype, 'com.microsoft']]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)


        optypes = ['MatMul']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,1)]]
            weights = [['input2', TensorProto.FLOAT, (2,1), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['FusedMatMul']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,1)]]
            weights = [['input2', TensorProto.FLOAT, (2,1), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype, 'com.microsoft']]
            model = self.build_model(inps, outs, weights, node_infos)
            ort.InferenceSession(model.SerializeToString())
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['Gemm']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,2)]]
            weights = [['input2', TensorProto.FLOAT, (2,1), np.random.random((2))],
                        ['input3', TensorProto.FLOAT, (1,2), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2', 'input3'], ['output'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['LayerNormalization']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output1', TensorProto.FLOAT, (1,2)], ['output2', TensorProto.FLOAT, (1,2)], ['output3', TensorProto.FLOAT, (1,2)]]
            weights = [['input2', TensorProto.FLOAT, (2,1), np.random.random((2))],
                        ['input3', TensorProto.FLOAT, (2,1), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2', 'input3'], ['output1', 'output2', 'output3'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['BatchNormalization']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, [1, 2]]]
            outs = [['output1', TensorProto.FLOAT, [1, 2]]]
            weights = [['input2', TensorProto.FLOAT, [2], np.random.random((2))],
                        ['input3', TensorProto.FLOAT, [2], np.random.random((2))],
                        ['input4', TensorProto.FLOAT, [2], np.random.random((2))],
                        ['input5', TensorProto.FLOAT, [2], np.random.random((2))],]
            node_infos = [['test', ['input1', 'input2', 'input3', 'input4', 'input5'], ['output1'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            ort.InferenceSession(model.SerializeToString())
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_fp16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(10 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['CUDAExecutionProvider'])
            outputs = session.run(None, input_data)

    def get_bf16_mixed_precision_model(self, model):
        from neural_compressor import MixedPrecisionConfig
        from neural_compressor.mix_precision import fit
        config = MixedPrecisionConfig(backend='onnxrt_dnnl_ep', precision='bf16')
        converted_model = fit(model, config)
        return converted_model

    @unittest.skipIf(not CpuInfo().bf16 or 'DnnlExecutionProvider' not in ort.get_all_providers(),
        "skip since DnnlExecutionProvider is not supported")
    def test_bf16(self):
        optypes = ['Sum', 'Sub', 'Div', 'Pow', 'Add']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,2)]]
            weights = [['input2', TensorProto.FLOAT, (1,2), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['Equal', 'Greater', 'GreaterOrEqual', 'Less', 'LessOrEqual']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.BOOL, (1,2)]]
            weights = [['input2', TensorProto.FLOAT, (1,2), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['Abs', 'Exp', 'Log', 'Round', 'Sqrt', 'Softmax', 'Exp', 'Tanh', 'Sigmoid', 'LeakyRelu', 'Round']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,2)]]
            node_infos = [['test', ['input1'], ['output'], optype]]
            model = self.build_model(inps, outs, [], node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['ReduceMean', 'ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceLogSumExp', 'ReduceMax', 'ReduceProd', \
                   'ReduceSum', 'ReduceSumSquare']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,1)]]
            node_infos = [['test', ['input1'], ['output'], optype]]
            model = self.build_model(inps, outs, [], node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['Gelu']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,2)]]
            node_infos = [['test', ['input1'], ['output'], optype, 'com.microsoft']]
            model = self.build_model(inps, outs, [], node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['BiasGelu', 'FastGelu']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, [2]]]
            outs = [['output', TensorProto.FLOAT, [2]]]
            weights = [['input2', TensorProto.FLOAT, [2], np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype, 'com.microsoft']]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)


        optypes = ['MatMul']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,1)]]
            weights = [['input2', TensorProto.FLOAT, (2,1), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['FusedMatMul']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,1)]]
            weights = [['input2', TensorProto.FLOAT, (2,1), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2'], ['output'], optype, 'com.microsoft']]
            model = self.build_model(inps, outs, weights, node_infos)
            ort.InferenceSession(model.SerializeToString())
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['Gemm']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output', TensorProto.FLOAT, (1,2)]]
            weights = [['input2', TensorProto.FLOAT, (2,1), np.random.random((2))],
                        ['input3', TensorProto.FLOAT, [], np.random.random((1))]]
            node_infos = [['test', ['input1', 'input2', 'input3'], ['output'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['LayerNormalization']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, (1,2)]]
            outs = [['output1', TensorProto.FLOAT, (1,2)], ['output2', TensorProto.FLOAT, (1,2)], ['output3', TensorProto.FLOAT, (1,2)]]
            weights = [['input2', TensorProto.FLOAT, (2,1), np.random.random((2))],
                        ['input3', TensorProto.FLOAT, (2,1), np.random.random((2))]]
            node_infos = [['test', ['input1', 'input2', 'input3'], ['output1', 'output2', 'output3'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)

        optypes = ['BatchNormalization']
        for optype in optypes:
            inps = [['input1', TensorProto.FLOAT, [1, 2]]]
            outs = [['output1', TensorProto.FLOAT, [1, 2]]]
            weights = [['input2', TensorProto.FLOAT, [2], np.random.random((2))],
                        ['input3', TensorProto.FLOAT, [2], np.random.random((2))],
                        ['input4', TensorProto.FLOAT, [2], np.random.random((2))],
                        ['input5', TensorProto.FLOAT, [2], np.random.random((2))],]
            node_infos = [['test', ['input1', 'input2', 'input3', 'input4', 'input5'], ['output1'], optype]]
            model = self.build_model(inps, outs, weights, node_infos)
            ort.InferenceSession(model.SerializeToString())
            input_data = self.build_test_data(['input1'], [(1,2)], ['float32'])
            convert_model = self.get_bf16_mixed_precision_model(model)
            self.assertTrue('Cast' in set([i.op_type for i in convert_model.nodes()]))
            self.assertTrue(16 in set([i.attribute[0].i for i in convert_model.nodes() if i.op_type == 'Cast']))
            session = ort.InferenceSession(convert_model.model.SerializeToString(), providers=['DnnlExecutionProvider'])
            outputs = session.run(None, input_data)


if __name__ == "__main__":
    unittest.main()
