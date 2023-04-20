"""Tests for quantization"""
import numpy as np
import unittest
import shutil
import os

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

class TestBasicTuningStrategy(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()
        self.workspace = os.path.join(os.getcwd(), 'nc_workspace')

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('saved', ignore_errors=True)
        shutil.rmtree(self.workspace)
        
    def test_run_basic_one_trial_new_api(self):
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS
        
        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)
        
        def fake_eval(model):
            return 1
        
        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig()
        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader= dataloader, eval_func=fake_eval)
        self.assertIsNotNone(q_model)


    def test_diagnosis(self):
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS
        
        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)
        
        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(diagnosis=True)
        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader= dataloader,\
                eval_func=lambda model: 1)
        self.assertEqual(os.path.exists(os.path.join(os.getcwd(), './nc_workspace/inspect_saved/fp32/inspect_result.pkl')), True)
        self.assertEqual(os.path.exists(os.path.join(os.getcwd(), './nc_workspace/inspect_saved/quan/inspect_result.pkl')), True)

        

    def test_run_create_eval_from_metric_and_dataloader(self):
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS
        
        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)
        from neural_compressor.metric import METRICS
        metrics = METRICS('tensorflow')
        top1 = metrics['topk']()
        
        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig()
        q_model = fit(model=self.constant_graph, conf=conf, calib_dataloader= dataloader,\
            eval_dataloader=dataloader, eval_metric=top1)

    def test_no_tuning(self):
        import torchvision
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS
        conf = PostTrainingQuantConfig()
        conf.performance_only = True
        # test performance_only without eval_func
        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((1, 3, 224, 224)))
        dataloader = DATALOADERS["pytorch"](dataset)
        # model
        model = torchvision.models.resnet18()
        #tuning and accuracy criterion
        conf = PostTrainingQuantConfig(quant_level=1)
        # fit
        q_model = fit(model=model, conf=conf, calib_dataloader=dataloader)
        self.assertIsNotNone(q_model)

if __name__ == "__main__":
    unittest.main()
