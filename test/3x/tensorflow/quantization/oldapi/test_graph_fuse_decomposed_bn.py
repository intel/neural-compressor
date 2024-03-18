import unittest

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import constant_op, dtypes, importer, ops, test_util
from tensorflow.python.ops import math_ops  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops, gen_math_ops, nn_ops
from tensorflow.python.platform import test
from tensorflow.python.tools import optimize_for_inference_lib

from neural_compressor.tensorflow.quantization.utils.graph_rewriter.generic.fuse_decomposed_bn import (
    FuseDecomposedBNOptimizer,
)


class OptimizeForInferenceTest(unittest.TestCase):
    def count_batchnorm_relavant_ops(self, graph_def):
        """Return the count of FusedBatchNorm op and the count of primitive
        ops which may make up batchnorm computation in a given graph."""
        batchnorm_count = 0
        decompose_count = 0
        for node in graph_def.node:
            if node.op == "FusedBatchNorm":
                batchnorm_count += 1
            if node.op in ["Add", "Rsqrt", "Mul", "Sub"]:
                decompose_count += 1
        return batchnorm_count, decompose_count

    @test_util.run_deprecated_v1
    def create_base_for_fuse_batchnorm(self, pattern_match_mode="MATCH_ALL", use_reshape=True):
        """Create testing graph and compute the result from original graph.
        Args:
            pattern_match_mode: A label string to indicate which batchnorm composition
            pattern to create in the resulting graph.
            "MATCH_ALL" - Create a graph matching the decomposed batchnorm pattern
                            with full set of primitive ops.
            "MATCH_NO_GAMMA" - Create a graph matching the decomposed batchnorm
                                pattern when gamma factor is 1 and multiplication
                                with gamma is omitted.
            "NO_MATCH" - Create a graph with same set of primitive ops which makes
                        up the decomposed batchnorm, but not matching the pattern.
        Returns:
            A GraphDef as original graph to run the decomposed batchnorm test cases.
            Computation result from executing the original graph defined by GraphDef.
        """
        ops.reset_default_graph()
        with tf.Session() as sess:
            inputs = [1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6]
            input_op = constant_op.constant(np.array(inputs), shape=[1, 1, 6, 2], dtype=dtypes.float32)
            weights = [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4]
            weights_op = constant_op.constant(np.array(weights), shape=[1, 2, 2, 2], dtype=dtypes.float32)
            conv_op = nn_ops.conv2d(input_op, weights_op, [1, 1, 1, 1], padding="SAME", name="conv_op")

            const_op_1 = constant_op.constant(np.array([0.25, 0.5]), shape=[2], dtype=dtypes.float32)
            if use_reshape:
                const_op_1 = array_ops.reshape(const_op_1, shape=[1, 1, 1, 2])
            const_op_2 = constant_op.constant(0.00001, dtype=dtypes.float32)
            const_op_3 = constant_op.constant(np.array([10, 20]), shape=[2], dtype=dtypes.float32)
            if use_reshape:
                const_op_3 = array_ops.reshape(const_op_3, shape=[1, 1, 1, 2])
            const_op_4 = constant_op.constant(np.array([0.1, 0.6]), shape=[2], dtype=dtypes.float32)
            if use_reshape:
                const_op_4 = array_ops.reshape(const_op_4, shape=[1, 1, 1, 2])

            add_op_1 = gen_math_ops.add(const_op_1, const_op_2)
            rsqrt_op = math_ops.rsqrt(add_op_1)

            variable_op = None
            if pattern_match_mode == "MATCH_NO_GAMMA":
                variable_op = rsqrt_op
            else:
                const_op_5 = constant_op.constant(np.array([1.0, 2.0]), shape=[2], dtype=dtypes.float32)
                if use_reshape:
                    const_op_5 = array_ops.reshape(const_op_5, shape=[1, 1, 1, 2])
                variable_op = math_ops.multiply(rsqrt_op, const_op_5)

            mul_op_1 = math_ops.multiply(conv_op, variable_op)

            mul_op_2 = None
            if pattern_match_mode == "NO_MATCH":
                const_op_6 = constant_op.constant(np.array([0.2, 0.5]), shape=[2], dtype=dtypes.float32)
                mul_op_2 = math_ops.multiply(const_op_3, const_op_6)
            else:
                mul_op_2 = math_ops.multiply(const_op_3, variable_op)

            sub_op = math_ops.subtract(const_op_4, mul_op_2)
            gen_math_ops.add(mul_op_1, sub_op, name="output")

            test_util.set_producer_version(ops.get_default_graph(), 8)

            original_graph = sess.graph_def
            original_result = sess.run(["output:0"])

            return original_graph, original_result

    def assertAllClose(self, first, second, rtol=1e-7, atol=0):
        first_array = np.array(first)
        second_array = np.array(second)
        np.testing.assert_allclose(first_array, second_array, rtol, atol)

    @test_util.run_deprecated_v1
    def testFuseDecomposedBatchNorm_MatchAll(self):
        for test_rehape in [False, True]:
            original_graph_def, original_result = self.create_base_for_fuse_batchnorm("MATCH_ALL", test_rehape)

            # Test correctness of fusing individual ops to FusedBatchNorm
            optimized_graph_def = FuseDecomposedBNOptimizer(original_graph_def).do_transformation()

            batchnorm_count, decompose_count = self.count_batchnorm_relavant_ops(optimized_graph_def)
            self.assertEqual(batchnorm_count, 1)
            self.assertEqual(decompose_count, 0)

            with tf.Session() as sess:
                _ = importer.import_graph_def(optimized_graph_def, input_map={}, name="optimized")
                optimized_result = sess.run(["optimized/output:0"])

            self.assertAllClose(original_result, optimized_result)

            # Test correctness of fusing individual ops to FusedBatchNorm followed by
            # folding FusedBatchNorm
            optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(optimized_graph_def)
            for node in optimized_graph_def.node:
                self.assertNotEqual("FusedBatchNorm", node.op)

            with tf.Session() as sess:
                _ = importer.import_graph_def(optimized_graph_def, input_map={}, name="optimized2")
                optimized_result = sess.run(["optimized2/output:0"])

            self.assertAllClose(original_result, optimized_result, rtol=1e-04, atol=1e-06)

    @test_util.run_deprecated_v1
    def testFuseDecomposedBatchNorm_MatchNoGamma(self):
        for test_rehape in [False, True]:
            original_graph_def, original_result = self.create_base_for_fuse_batchnorm("MATCH_NO_GAMMA", test_rehape)

            # Test correctness of fusing individual ops to FusedBatchNorm
            optimized_graph_def = FuseDecomposedBNOptimizer(original_graph_def).do_transformation()

            batchnorm_count, decompose_count = self.count_batchnorm_relavant_ops(optimized_graph_def)
            self.assertEqual(batchnorm_count, 1)
            self.assertEqual(decompose_count, 0)

            with tf.Session() as sess:
                _ = importer.import_graph_def(optimized_graph_def, input_map={}, name="optimized")
                optimized_result = sess.run(["optimized/output:0"])

            self.assertAllClose(original_result, optimized_result)

            # Test correctness of fusing individual ops to FusedBatchNorm followed by
            # folding FusedBatchNorm
            optimized_graph_def = optimize_for_inference_lib.fold_batch_norms(optimized_graph_def)
            for node in optimized_graph_def.node:
                self.assertNotEqual("FusedBatchNorm", node.op)

            with tf.Session() as sess:
                _ = importer.import_graph_def(optimized_graph_def, input_map={}, name="optimized2")
                optimized_result = sess.run(["optimized2/output:0"])

            self.assertAllClose(original_result, optimized_result, rtol=1e-04, atol=1e-06)

    @test_util.run_deprecated_v1
    def testFuseDecomposedBatchNorm_NonMatchCase(self):
        for test_rehape in [False, True]:
            original_graph_def, original_result = self.create_base_for_fuse_batchnorm("NO_MATCH", test_rehape)

            # Test for not to fuse ops if graph has same types of ops but pattern mismatch
            optimized_graph_def = FuseDecomposedBNOptimizer(original_graph_def).do_transformation()

            batchnorm_count, math_op_count = self.count_batchnorm_relavant_ops(optimized_graph_def)
            self.assertEqual(batchnorm_count, 0)
            self.assertEqual(math_op_count, 7)

            with tf.Session() as sess:
                _ = importer.import_graph_def(optimized_graph_def, input_map={}, name="optimized")
                optimized_result = sess.run(["optimized/output:0"])

            self.assertAllClose(original_result, optimized_result)


if __name__ == "__main__":
    unittest.main()
