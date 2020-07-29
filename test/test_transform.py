"""Tests for the transform module."""
import numpy as np
import unittest
import os
from ilit.data import TRANSFORMS
class TestMetrics(unittest.TestCase):
    def setUp(self):
        pass
    def test_tensorflow_2(self):
        image = np.ones([1, 256, 256, 1])
        resize_kwargs = {"size":[224, 224]}
        transforms = TRANSFORMS(framework="tensorflow", process="preprocess")
        resize = transforms['resize'](**resize_kwargs)
        random_crop_kwargs = {"size": [1, 128, 128, 1]}
        random_crop = transforms['random_crop'](**random_crop_kwargs)
        transform_list = [resize, random_crop]
        compose = transforms['Compose'](transform_list)
        image_result = compose(image)
        self.assertEqual(image_result.shape, (1, 128, 128, 1))

if __name__ == "__main__":
    unittest.main()
