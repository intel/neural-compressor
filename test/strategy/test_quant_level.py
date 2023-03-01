"""Tests for optimization level & conservative strategy"""

import shutil
import unittest
import numpy as np
from copy import deepcopy

from onnx import helper, TensorProto, numpy_helper
from neural_compressor.data import Datasets, DATALOADERS
from neural_compressor.utils import logger
from neural_compressor import PostTrainingQuantConfig
from neural_compressor.quantization import fit

def build_conv_model():
    initializers = []
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    conv1_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [3, 3, 3, 3]).astype(np.float32), name='conv1_weight')
    conv1_node = helper.make_node('Conv', ['input', 'conv1_weight'], ['conv1_output'], name='conv1')

    conv2_weight_initializer = numpy_helper.from_array(
        np.random.randint(-1, 2, [5, 3, 3, 3]).astype(np.float32), name='conv2_weight')
    conv2_node = helper.make_node('Conv', ['conv1_output', 'conv2_weight'], ['conv2_output'], name='conv2')

    avg_args = {'kernel_shape': [3, 3]}
    avgpool_node = helper.make_node('AveragePool', ['conv1_output'], ['avg_output'], name='AveragePool', **avg_args)

    concat_node = helper.make_node('Concat', ['avg_output', 'conv2_output'], 
        ['concat_output'], name='Concat', axis=1)
    output = helper.make_tensor_value_info('concat_output', TensorProto.FLOAT, [1, 8, 220, 220])
    initializers = [conv1_weight_initializer, conv2_weight_initializer]
    graph = helper.make_graph([conv1_node, conv2_node, concat_node, avgpool_node],
        'test', [input], [output], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model

def build_ort_data():
    datasets = Datasets('onnxrt_qlinearops')
    cv_dataset = datasets['dummy'](shape=(10, 3, 224, 224), low=0., high=1., label=True)
    cv_dataloader = DATALOADERS['onnxrt_qlinearops'](cv_dataset)
    return cv_dataloader


def export_onnx_model(model, path, opset=12):
    import torch
    x = torch.randn(100, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    path,                      # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=opset,          # the ONNX version to export the model to, please ensure at least 11.
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ["input"],   # the model"s input names
                    output_names = ["output"], # the model"s output names
                    dynamic_axes={"input" : {0 : "batch_size"},    # variable length axes
                                  "output" : {0 : "batch_size"}})

def build_resnet18():
    import onnx
    import torchvision
    rn18_model = torchvision.models.resnet18()
    rn18_export_path = "rn18.onnx"
    export_onnx_model(rn18_model, rn18_export_path, 12)
    rn18_model = onnx.load(rn18_export_path)
    return rn18_model


def build_fake_model():
    import tensorflow as tf
    try:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)).astype(np.float32), name='y')
            z = tf.constant(np.random.random((1,1,1,1)).astype(np.float32), name='z')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')
            op2 = tf.nn.conv2d(input=op, filters=z, strides=[1,1,1,1], padding='VALID', )
            last_identity = tf.identity(op2, name='op2_to_store')
            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op2_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    except:
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape=(1,3,3,1), name='x')
            y = tf.constant(np.random.random((2,2,1,1)).astype(np.float32), name='y')
            z = tf.constant(np.random.random((1,1,1,1)).astype(np.float32), name='z')
            op = tf.nn.conv2d(input=x, filters=y, strides=[1,1,1,1], padding='VALID', name='op_to_store')
            op2 = tf.nn.conv2d(input=op, filters=z, strides=[1,1,1,1], padding='VALID')
            last_identity = tf.identity(op2, name='op2_to_store')

            sess.run(tf.compat.v1.global_variables_initializer())
            constant_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['op2_to_store'])

        graph_def.ParseFromString(constant_graph.SerializeToString())
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
    return graph


class TestQuantLevel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        self.ort_cv_model = build_conv_model()
        self.ort_cv_dataloader = build_ort_data()
        self.ort_resnet18 = build_resnet18()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('saved', ignore_errors=True)
        shutil.rmtree('nc_workspace', ignore_errors=True)
        
    def test_quant_level_auto(self):
        from neural_compressor import PostTrainingQuantConfig
        from neural_compressor.quantization import fit
                
        acc_lst = [1.0, 0.9, 1.1, 0.8]
        def fake_eval(model):
            result = acc_lst[0]
            del acc_lst[0]
            return result       

        conf = PostTrainingQuantConfig(approach='static')
        
        q_model = fit(model=self.ort_cv_model, conf=conf, \
            calib_dataloader=self.ort_cv_dataloader, eval_func=fake_eval)
        node_names = [i.name for i in q_model.nodes()]
        # All conv will be quantized
        for node_name in node_names:
            if 'conv' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
    
    def test2_quant_level_auto(self):
        # All conv will be quantized but matmul not
        acc_lst = [1.0, 0.9, 1.1, 0.8]
        def fake_eval(model):
            result = acc_lst[0]
            del acc_lst[0]
            return result
        
        conf = PostTrainingQuantConfig(approach='static')
        q_model = fit(model=deepcopy(self.ort_resnet18), conf=conf, \
            calib_dataloader=self.ort_cv_dataloader, eval_func=fake_eval)
        node_names = [i.name for i in q_model.nodes()]
        
        for node_name in node_names:
            if 'conv' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
            if 'MatMul' in node_name:
                self.assertTrue('quant' not in node_name and 'Quant' not in node_name)

    def test3_quant_level_auto(self):
        # All conv/matmul will be quantized
        acc_lst = [1.0, 0.9, 1.1, 1.2]
        def fake_eval3(model):
            result = acc_lst[0]
            del acc_lst[0]
            return result
        conf = PostTrainingQuantConfig(approach='static')
        q_model = fit(model=deepcopy(self.ort_resnet18), conf=conf, \
            calib_dataloader=self.ort_cv_dataloader, eval_func=fake_eval3)
        node_names = [i.name for i in q_model.nodes()]
        
        for node_name in node_names:
            if 'conv' in node_name or 'MatMul' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
                
    def test4_quant_level_auto(self):
        # All matmul will be quantized but conv not
        acc_lst = [1.0, 0.9, 0.8, 1.1]
        def fake_eval4(model):
            result = acc_lst[0]
            del acc_lst[0]
            return result

        conf = PostTrainingQuantConfig(approach='static')
        q_model = fit(model=deepcopy(self.ort_resnet18), conf=conf, \
            calib_dataloader=self.ort_cv_dataloader, eval_func=fake_eval4)
        node_names = [i.name for i in q_model.nodes()]
        
        for node_name in node_names:
            if 'MatMul' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
            if 'conv' in node_name:
                self.assertTrue('quant' not in node_name and 'Quant' not in node_name)
                
    def test5_quant_level_auto(self):
        # All matmul and conv will be quantized, return with all int8.
        acc_lst = [1.0, 1.2, 0.8, 1.1]
        def fake_eval5(model):
            result = acc_lst[0]
            del acc_lst[0]
            return result

        conf = PostTrainingQuantConfig(approach='static')
        q_model = fit(model=deepcopy(self.ort_resnet18), conf=conf, \
            calib_dataloader=self.ort_cv_dataloader, eval_func=fake_eval5)
        node_names = [i.name for i in q_model.nodes()]
        
        for node_name in node_names:
            if 'MatMul' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
            if 'conv' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
                
    def test6_quant_level_auto(self):
        # start with basic
        acc_lst = [1.0, 0.7, 0.8, 0.9, 1.1]
        def fake_eval6(model):
            result = acc_lst[0]
            del acc_lst[0]
            return result

        conf = PostTrainingQuantConfig(approach='static')
        q_model = fit(model=deepcopy(self.ort_resnet18), conf=conf, \
            calib_dataloader=self.ort_cv_dataloader, eval_func=fake_eval6)
        node_names = [i.name for i in q_model.nodes()]
        
        for node_name in node_names:
            if 'MatMul' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
            if 'conv' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
                
    def test7_quant_level_auto(self):
        # start with basic and return at the 3th of basic stage
        acc_lst = [1.0, 0.7, 0.8, 0.9, 0.95, 0.98, 1.1]
        def fake_eval7(model):
            result = acc_lst[0]
            del acc_lst[0]
            return result

        conf = PostTrainingQuantConfig(approach='static')
        q_model = fit(model=deepcopy(self.ort_resnet18), conf=conf, \
            calib_dataloader=self.ort_cv_dataloader, eval_func=fake_eval7)
        node_names = [i.name for i in q_model.nodes()]
        
        for node_name in node_names:
            if 'MatMul' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)
            if 'conv' in node_name:
                self.assertTrue('quant' in node_name or 'Quant' in node_name)

    def test_pt_opt_level_0(self):
        logger.info("*** Test: quantization level is auto with pytorch model.")
        import torchvision
        from neural_compressor.data import Datasets, DATALOADERS
        from neural_compressor import PostTrainingQuantConfig
        from neural_compressor.quantization import fit
        
        resnet18 = torchvision.models.resnet18()
        acc_lst =  [2.0, 1.0, 2.1, 2.2, 2.3]
        def _fake_eval(model):
            result = acc_lst[0]
            del acc_lst[0]
            return result

        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)
        conf = PostTrainingQuantConfig()
        q_model = fit(model=resnet18, conf=conf, calib_dataloader= dataloader, \
                      eval_dataloader=dataloader, eval_func=_fake_eval)
        self.assertIsNotNone(q_model)
        fc_layer = q_model._model.fc
        self.assertTrue('Quant' in str(fc_layer))

if __name__ == "__main__":
    unittest.main()
