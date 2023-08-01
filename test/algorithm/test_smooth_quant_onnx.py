import onnx
from onnx import helper, TensorProto, numpy_helper
import copy
import unittest
import numpy as np
import shutil
from neural_compressor.data import Datasets, DATALOADERS
from neural_compressor.adaptor.ox_utils.smooth_quant import ORTSmoothQuant
import logging
logger = logging.getLogger("neural_compressor")

def check_model_is_same(model_proto1, model_proto2):
    # Compare if both models have the same number of nodes
    if len(model_proto1.graph.node) != len(model_proto2.graph.node):
        return False

    # Compare individual nodes in both models
    for node1, node2 in zip(model_proto1.graph.node, model_proto2.graph.node):
        print(node1.name, node2.name)
        # Check node name, input, output, and op_type
        if node1.name != node2.name or \
            node1.op_type != node2.op_type or \
            node1.input != node2.input or \
            node1.output != node2.output:
            return False

        # Check node attribure
        if len(node1.attribute) != len(node2.attribute):
            return False

        for attr1, attr2 in zip(node1.attribute, node2.attribute):
            if attr1.name == attr2.name:
                if attr1.type == onnx.AttributeProto.FLOATS:
                    # Compare float attributes using numpy.allclose
                    if not attr1.floats == attr2.floats:
                        return False
                elif attr1.type == onnx.AttributeProto.INTS:
                    # Compare int attributes
                    if attr1.ints != attr2.ints:
                        return False
    # Compare initializer
    init1 = {init.name: init for init in model_proto1.graph.initializer}
    init2 = {init.name: init for init in model_proto2.graph.initializer}
    for name in init1.keys():
        if name not in init2 or \
            not (numpy_helper.to_array(init1[name]) == numpy_helper.to_array(init2[name])).all():
            return False

    # Compare model inputs and outputs
    if model_proto1.graph.input != model_proto2.graph.input or \
       model_proto1.graph.output != model_proto2.graph.output:
        return False
    
    return True


def build_onnx_model():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 5, 2])
    H = helper.make_tensor_value_info('H', TensorProto.FLOAT, [1, 5, 2])

    g_value = np.random.uniform(low=0.001, high=0.5, size=(25)).astype(np.float32)
    G_init = helper.make_tensor('G', TensorProto.FLOAT, [5, 5], g_value.reshape(25).tolist())
    matmul_node = onnx.helper.make_node('MatMul', ['A', 'G'], ['C'], name='Matmul')

    b_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    B_init = helper.make_tensor('B', TensorProto.FLOAT, [5, 2], b_value.reshape(10).tolist())
    matmul_node2 = onnx.helper.make_node('MatMul', ['C', 'B'], ['I'], name='Matmul2')

    e_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    E_init = helper.make_tensor('E', TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    matmul_node3 = onnx.helper.make_node('MatMul', ['C', 'E'], ['K'], name='Matmul3')

    add = onnx.helper.make_node('Add', ['I', 'E'], ['D'], name='add')

    f_value = np.random.uniform(low=0.001, high=0.5, size=(10)).astype(np.float32)
    F_init = helper.make_tensor('F', TensorProto.FLOAT, [5, 2], f_value.reshape(10).tolist())
    add2 = onnx.helper.make_node('Add', ['D', 'F'], ['H'], name='add2')

    graph = helper.make_graph([matmul_node, matmul_node2, matmul_node3, add, add2], 'test_graph_1', [A], [H], [B_init, E_init, F_init, G_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    return model


class TestORTSq(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model = build_onnx_model()
        dataset = Datasets("onnxrt_qdq")["dummy_v2"]((5,5), (5,1))
        self.dataloader = DATALOADERS['onnxrt_qlinearops'](dataset)
        fixed_dataset = Datasets("onnxrt_qdq")['dummy'](shape=(5,5,5), label=True)
        self.fixed_dataloader = DATALOADERS['onnxrt_qlinearops'](fixed_dataset)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./nc_workspace", ignore_errors=True)

    def test_sq(self):
        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, scales_per_op=False)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 1)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, folding=False, scales_per_op=False)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 2)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, folding=False, scales_per_op=True)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, scales_per_op=True)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, scales_per_op=True, alpha='auto')
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 3)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)


        sq = ORTSmoothQuant(copy.deepcopy(self.model), self.dataloader)
        model = sq.transform(calib_iter=5, alpha='auto', scales_per_op=False)
        self.assertEqual(len([i for i in model.model.graph.node if i.op_type == 'Mul']), 1)
        sq.recover()
        self.assertEqual(len(sq.model.nodes()), len(self.model.graph.node))
        for init in self.model.graph.initializer:
            tensor = numpy_helper.to_array(init)
            sq_tensor = numpy_helper.to_array(sq.model.get_initializer(init.name))
            self.assertAlmostEqual(tensor[0][0], sq_tensor[0][0], 4)

    def _test_sq_tune_alpha_common(self, eval_func, alpha=np.arange(0.1, 0.2, 0.05).tolist(), quant_level=1):
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        tuning_criterion = TuningCriterion(max_trials=8)

        fp32_model = self.model
        conf = PostTrainingQuantConfig(
            quant_level=quant_level,
            tuning_criterion=tuning_criterion,
            calibration_sampling_size=4,
            recipes={"smooth_quant": True, 
                     "smooth_quant_args": {"alpha": alpha}
                     }
        )
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=self.fixed_dataloader,
            eval_func=eval_func,
        )
        self.assertIsNotNone(q_model)
        return q_model

    def test_tune_sq_alpha(self):
        from functools import partial
        def fake_eval(model, eval_result_lst):
            acc = eval_result_lst.pop(0)
            return acc
        
        # test for quantized models generated by int alpha and list alpha are the same
        partial_fake_eval = partial(fake_eval, eval_result_lst = [1, 1.1] )
        q_model_without_tune = self._test_sq_tune_alpha_common(partial_fake_eval, alpha=0.5)
        partial_fake_eval = partial(fake_eval, eval_result_lst = [1, 0.8, 1.1] )
        q_model_with_tune = self._test_sq_tune_alpha_common(partial_fake_eval, alpha=[0.4, 0.5])
        self.assertTrue(check_model_is_same(q_model_without_tune.model, q_model_with_tune.model))

        # test for alpha is a list
        for eval_result_lst, note in [
                ([1, 0.8, 1.1, 0.7, 1.1], "Expect tuning ends at 2nd trial with alpha is 0.15"),
                ([1, 0.8, 0.9, 0.7, 1.1], "Expect tuning ends at 4th trial with alpha is 0.15"),
                ([1, 0.9, 0.8, 0.7, 1.1], "Expect tuning ends at 4th trial with alpha is 0.10")
                ]:
            logger.info(f"test_sq_tune_alpha_common with eval_result_lst: {eval_result_lst}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst = eval_result_lst )
            self._test_sq_tune_alpha_common(partial_fake_eval)

        # test for various alphas
        for eval_result_lst, alpha, note in [
                ([1, 0.8, 1.1, 0.7, 1.1], 0.5 ,"Expect tuning ends at 2nd trial with alpha is 0.5 and not tune sq's alpha."),
                ([1, 0.8, 0.9, 0.7, 1.1], [0.5], "Expect tuning ends at 4th trial with alpha is  0.5 and not tune sq's alpha."),
                ([1, 0.9, 0.8, 0.7, 1.1], [0.5, 0.7, 0.9] ,"Expect tuning ends at 4th trial with alpha is 0.5")
                ]:
            logger.info(f"test_sq_tune_alpha_common with eval_result_lst: {eval_result_lst}, alpha: {alpha}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst=eval_result_lst)
            self._test_sq_tune_alpha_common(partial_fake_eval, alpha=alpha)

        # test for quant_level is auto or 0
        for eval_result_lst, alpha, quant_level, note in [
                (
                    [1, 0.8, 1.1, 0.7, 1.1], 
                    np.arange(0.1, 0.2, 0.05).tolist(), 
                    "auto", 
                    "Expect tuning ends at 2nd trial with alpha is 0.15."
                    ),
                (
                    [1, 0.8, 0.9, 0.7, 1.1],
                    np.arange(0.1, 0.2, 0.05).tolist(),
                    "auto",
                    "Expect tuning ends at 4th trial with alpha is  0.15 at basic strategy."
                    ),
                (
                    [1, 1.1, 0.8, 0.7, 1.1], 
                    np.arange(0.1, 0.2, 0.05).tolist(),
                    0,
                    "Expect tuning ends at 1th trial with alpha is 0.1")
                ]:
            logger.info(f"test_sq_tune_alpha_common with ")
            logger.info(f"eval_result_lst: {eval_result_lst}, alpha: {alpha}, quant_level: {quant_level}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst=eval_result_lst)
            self._test_sq_tune_alpha_common(partial_fake_eval, alpha=alpha, quant_level=quant_level)
        
if __name__ == '__main__':
    unittest.main()
