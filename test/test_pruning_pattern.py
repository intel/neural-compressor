import random
import copy

import unittest
import numpy as np

from neural_compressor.experimental.pruning_recipes.patterns import patterns

class TestPruningPattern(unittest.TestCase):

    tensor_4d = np.random.random([2240,2240, 3, 3])
    tensor_2d = np.random.random([5120,2560])

    def test_tile_pattern(self):
        for tensor in [self.tensor_2d, self.tensor_4d]:
            shape = list(tensor.shape)
            size = tensor.size

            for mask_shape in [(1, 1), (2, 2), (1, 16), (4, 1), (1, 2)]:
                m0 = mask_shape[0]
                m1 = mask_shape[1]
                pattern = patterns['tile_pattern_{}x{}'.format(m0, m1)]()
                new_shape = [shape[0] / m0] + [size // shape[0] / m1]
                sparse_tensor = self.sparsify_tensor(tensor, [m0,m1], 0.8)
                reduced_tensor = pattern.reduce(sparse_tensor)
                self.assertEqual(list(reduced_tensor.shape), new_shape)
                self.assertAlmostEqual(pattern.compute_sparsity(sparse_tensor), 0.8, delta=0.01)
                mask = reduced_tensor == 0
                repeat_mask = pattern.repeat_mask(mask, ori_shape=tensor.shape)
                self.assertEqual(repeat_mask.shape, tensor.shape)

    def sparsify_tensor(self, tensor, mask_shape, ratio):
        tensor = copy.deepcopy(tensor)
        for i in range(tensor.shape[0]//mask_shape[0]):
            for j in range(tensor.shape[1]//mask_shape[1]):
                if random.random() < ratio:
                    tensor[i*mask_shape[0]:(i+1)*mask_shape[0], j*mask_shape[1]:(j+1)*mask_shape[1], ...] = 0
        return tensor


if __name__ == "__main__":
    unittest.main()
