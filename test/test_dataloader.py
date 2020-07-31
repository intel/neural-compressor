"""Tests for the dataloader module."""
import numpy as np
import unittest
import os
from ilit.data import TRANSFORMS, Dataset, DATASETS, DataLoader, dataset_registry
import sys


class TestMetrics(unittest.TestCase):
    def setUp(self):
        pass

    def test_tensorflow_dummy(self):
        datasets = DATASETS('tensorflow')
        dataset = datasets['dummy'](shape=(4, 256, 256, 3))
        
        data_loader = DataLoader('tensorflow', dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))

    def test_tensorflow_list_dict(self):
        dataset = [{'a':1, 'b':2, 'c':3, 'd':4}, {'a':5, 'b':6, 'c':7, 'd':8}]
        data_loader = DataLoader('tensorflow', dataset)

        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {'a':[1], 'b':[2], 'c':[3], 'd':[4]})

        # test iterable consistent
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {'a':[1], 'b':[2], 'c':[3], 'd':[4]})

        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data, {'a':[1, 5], 'b':[2, 6], 'c':[3, 7], 'd':[4, 8]})

    def test_tensorflow2_dataset(self):
        dataset = ((1, 2, 3, 4), (5, 6, 7, 8))
        dataset = np.array(dataset)
        import tensorflow as tf
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        data_loader = DataLoader('tensorflow', dataset)
 
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data[0][1], 2)
 
    def test_pytorch_dummy(self):
        datasets = DATASETS('pytorch')
        dataset = datasets['dummy'](shape=(4, 256, 256, 3))
        
        data_loader = DataLoader('pytorch', dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))
 
 
    def test_mxnet_dummy(self):
        datasets = DATASETS('mxnet')
        dataset = datasets['dummy'](shape=(4, 256, 256, 3))
        
        data_loader = DataLoader('mxnet', dataset)
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (1, 256, 256, 3))
        # dynamic batching
        data_loader.batch(batch_size=2, last_batch='rollover')
        iterator = iter(data_loader)
        data = next(iterator)
        self.assertEqual(data.shape, (2, 256, 256, 3))


if __name__ == "__main__":
    unittest.main()
