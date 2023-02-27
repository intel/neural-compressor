import copy
import os
import shutil
import unittest
import tensorflow as tf
import numpy as np
import torchvision

def build_fake_model():
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

class Test_MSEV2Strategy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tf_model = build_fake_model()
        self.torch_model = torchvision.models.resnet18()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('./saved', ignore_errors=True)
        shutil.rmtree('nc_workspace', ignore_errors=True)

    def test_quantization_saved_tf(self):
        i = [0] # use a mutable type (list) to wrap the int object
        def fake_eval_func(_):
            #               1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            eval_list = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
            i[0] += 1
            return eval_list[i[0]]
        
        from neural_compressor.quantization import fit
        from neural_compressor.config import TuningCriterion, PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS['tensorflow'](dataset)
        
        conf = PostTrainingQuantConfig(
            approach="static",
            quant_level=1,
            tuning_criterion=TuningCriterion(strategy="mse_v2"))
        
        q_model = fit(
            model=self.tf_model,
            conf=conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_func=fake_eval_func)
        self.assertIsNotNone(q_model)
        
    def test_quantization_saved_tf_with_confidence_batches(self):
        i = [0] # use a mutable type (list) to wrap the int object
        def fake_eval_func(_):
            #               1, 2, 3, 4, 5, 6, 7, 8, 9, 10
            eval_list = [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
            i[0] += 1
            return eval_list[i[0]]
        
        from neural_compressor.quantization import fit
        from neural_compressor.config import TuningCriterion, PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS['tensorflow'](dataset)
        
        conf = PostTrainingQuantConfig(
            approach="static",
            quant_level=1,
            tuning_criterion=TuningCriterion(
                strategy="mse_v2",
                strategy_kwargs={
                    "confidence_batches": 5,
                }))
        
        q_model = fit(
            model=self.tf_model,
            conf=conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_func=fake_eval_func)
        self.assertIsNotNone(q_model)
        
    def test_quantization_saved_torch(self):
        i = [0]
        def fake_eval_func(model):
            acc_lst = [1, 1, 0, 0, 0, 0, 1, 1.1, 1.5, 1.1]
            i[0] += 1
            return acc_lst[i[0]]
        
        from neural_compressor.quantization import fit
        from neural_compressor.config import TuningCriterion, PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS
        dataset = Datasets("pytorch")["dummy"](((1, 3, 224, 224)))
        dataloader = DATALOADERS['pytorch'](dataset)
        
        conf = PostTrainingQuantConfig(
            approach="static",
            quant_level=1,
            tuning_criterion=TuningCriterion(strategy="mse_v2"))
        
        q_model = fit(
            model=self.torch_model,
            conf=conf,
            calib_dataloader=dataloader,
            eval_dataloader=dataloader,
            eval_func=fake_eval_func)
        self.assertIsNotNone(q_model)
        
if __name__ == "__main__":
    unittest.main()
