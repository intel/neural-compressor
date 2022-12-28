"""Tests for optimization level & conservative strategy"""

import shutil
import unittest
import time

import numpy as np

from neural_compressor.utils import logger

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


class TestOptimizationLevel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.constant_graph = build_fake_model()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree('saved', ignore_errors=True)
        shutil.rmtree('nc_workspace', ignore_errors=True)

    def test_tf_opt_level_0(self):
        logger.info("*** Test: optimization level 0 with tensorflow model.")
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS

        # fake evaluation function
        def _fake_eval(model):
            return 1

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        quant_level = 0
        conf = PostTrainingQuantConfig(quant_level=0)

        # fit
        q_model = fit(model=self.constant_graph,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        self.assertIsNotNone(q_model)

    def test_tf_opt_level_1(self):
        logger.info("*** Test: optimization level 1 with tensorflow model.")
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS

        # fake evaluation function
        self._fake_acc = 10
        def _fake_eval(model):
            self._fake_acc -= 1
            return self._fake_acc

        # dataset and dataloader
        dataset = Datasets("tensorflow")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["tensorflow"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig()

        # fit
        q_model = fit(model=self.constant_graph,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        self.assertIsNone(q_model)

    def test_pt_opt_level_0(self):
        logger.info("*** Test: optimization level 0 with pytorch model.")
        from neural_compressor.quantization import fit
        from neural_compressor.config import PostTrainingQuantConfig
        from neural_compressor.data import Datasets, DATALOADERS
        import torchvision

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        acc_lst =  [2.0, 1.0, 2.1, 2.2, 2.3]
        perf_lst = [2.0, 1.5, 1.0, 0.5, 0.1]
        self.test_pt_opt_level_0_index = -1
        def _fake_eval(model):
            self.test_pt_opt_level_0_index += 1
            perf = perf_lst[self.test_pt_opt_level_0_index]
            time.sleep(perf)
            return acc_lst[self.test_pt_opt_level_0_index]

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        quant_level = 0
        conf = PostTrainingQuantConfig(quant_level=quant_level)

        # fit
        q_model = fit(model=resnet18,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        self.assertIsNotNone(q_model)

if __name__ == "__main__":
    unittest.main()
