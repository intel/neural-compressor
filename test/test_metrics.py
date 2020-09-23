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

    def test_tensorflow_topk(self):
        metrics = METRICS('tensorflow')
        top1 = metrics['topk']()
        top2 = metrics['topk'](k=2)
        top3 = metrics['topk'](k=3)

        predicts = [[0, 0.2, 0.9, 0.3], [0, 0.9, 0.8, 0]]
        single_predict = [0, 0.2, 0.9, 0.3]
       
        labels = [[0, 1, 0, 0], [0, 0, 1, 0]]
        sparse_labels = [2, 2]
        single_label = 2

        # test functionality of one-hot label
        top1.update(predicts, labels)
        top2.update(predicts, labels)
        top3.update(predicts, labels)
        self.assertEqual(top1.result(), 0.0)
        self.assertEqual(top2.result(), 0.5)
        self.assertEqual(top3.result(), 1)

        # test functionality of sparse label
        top1.update(predicts, sparse_labels)
        top2.update(predicts, sparse_labels)
        top3.update(predicts, sparse_labels)
        self.assertEqual(top1.result(), 0.25)
        self.assertEqual(top2.result(), 0.75)
        self.assertEqual(top3.result(), 1)

        # test functionality of single label
        top1.update(single_predict, single_label)
        top2.update(single_predict, single_label)
        top3.update(single_predict, single_label)
        self.assertEqual(top1.result(), 0.4)
        self.assertEqual(top2.result(), 0.8)
        self.assertEqual(top3.result(), 1)
        

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
