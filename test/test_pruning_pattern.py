import random
import copy

import unittest
import numpy as np

from neural_compressor.experimental.pruning_recipes.patterns import patterns

class TestPruningPattern(unittest.TestCase):

    tensor_4d = np.random.random([1,100,2240,2240])
    tensor_2d = np.random.random([5120,2560])

    def test_tile_pattern(self):
        for tensor in [self.tensor_2d, self.tensor_4d]:
            shape = list(tensor.shape)

            pattern_1x1 = patterns['tile_pattern_1x1']()
            shape_1x1 = shape[:-2] + [shape[-2] / 1] + [shape[-1] / 1]
            sparse_tensor = self.sparsify_tensor(tensor, [1,1], 0.8)
            self.assertEqual(list(pattern_1x1.reduce(sparse_tensor).shape), shape_1x1)
            self.assertAlmostEqual(pattern_1x1.compute_sparsity(sparse_tensor), 0.8, delta=0.01)

            pattern_2x2 = patterns['tile_pattern_2x2']()
            shape_2x2 = shape[:-2] + [shape[-2] / 2] + [shape[-1] / 2]
            sparse_tensor = self.sparsify_tensor(tensor, [2,2], 0.7)
            self.assertEqual(list(pattern_2x2.reduce(sparse_tensor).shape), shape_2x2)
            self.assertAlmostEqual(pattern_2x2.compute_sparsity(sparse_tensor), 0.7, delta=0.01)

            pattern_1x16 = patterns['tile_pattern_1x16']()
            shape_1x16 = shape[:-2] + [shape[-2] / 1] + [shape[-1] / 16]
            sparse_tensor = self.sparsify_tensor(tensor, [1,16], 0.5)
            self.assertEqual(list(pattern_1x16.reduce(sparse_tensor).shape), shape_1x16)
            self.assertAlmostEqual(pattern_1x16.compute_sparsity(sparse_tensor), 0.5, delta=0.02)

    def sparsify_tensor(self, tensor, mask_shape, ratio):
        tensor = copy.deepcopy(tensor)
        for i in range(tensor.shape[-2]//mask_shape[-2]):
            for j in range(tensor.shape[-1]//mask_shape[-1]):
                if random.random() < ratio:
                    tensor[..., i*mask_shape[-2]:(i+1)*mask_shape[-2], j*mask_shape[-1]:(j+1)*mask_shape[-1]] = 0
        return tensor


if __name__ == "__main__":
    unittest.main()
