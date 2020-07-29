"""Tests for the metrics module."""
import numpy as np
import unittest
import os
from ilit.metric import METRICS

class TestMetrics(unittest.TestCase):
    def setUp(self):
        pass

    def test_tensorflow_accuracy(self):
        metrics = METRICS('tensorflow')
        acc = metrics['Accuracy']()
        predicts = [1, 0, 1, 1]
        labels = [0, 1, 1, 1]
        acc.update(predicts, labels)
        acc_result = acc.result()
        self.assertEqual(acc_result, 0.5)

    def test_pytorch_accuracy(self):
        metrics = METRICS('pytorch')
        acc = metrics['Accuracy']()
        predicts = [1, 0, 1, 1]
        labels = [0, 1, 1, 1]
        acc.update(predicts, labels)
        acc_result = acc.result()
        self.assertEqual(acc_result, 0.5)

    def test_mxnet_accuracy(self):
        metrics = METRICS('mxnet')
        acc = metrics['Accuracy']()
        predicts = [1, 0, 1, 1]
        labels = [0, 1, 1, 1]
        acc.update(predicts, labels)
        acc_result = acc.result()
        self.assertEqual(acc_result, 0.5)

if __name__ == "__main__":
    unittest.main()
