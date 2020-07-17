"""Tests for the metrics module."""
import numpy as np
import unittest
import sys
import os
from ilit import metric

class TestMetrics(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_TopK(self):
        acc_metric = metric.Topk({'topk':1})
        fake_data = np.array([[2, 3], [4, 5], [5, 5]])
        fake_predict = fake_data/np.sum(fake_data.astype(np.float32), axis=1)[:, None]
        fake_label = np.array([1,1,1])
        fake_wrong_label = np.array([1,0,1])
        acc = acc_metric.evaluate(fake_predict, fake_label)
        wrong_acc = acc_metric.evaluate(fake_predict, fake_wrong_label)
        self.assertEqual(acc, 1.0)
        self.assertEqual(wrong_acc, 5.0/6.0) #[1,1,1] [1,0,1]
        

        acc_metric = metric.Topk({'topk':5})
        fake_data = np.array([[1, 2, 3, 2, 6, 8], [4, 5, 6, 3, 1, 2], [5, 5, 5, 5, 5, 5]])
        fake_predict = fake_data/np.sum(fake_data.astype(np.float32), axis=1)[:, None] 
        fake_label_1 = np.array([1,1,1])
        fake_label_2 = np.array([3,5,4])
        fake_wrong_label = np.array([0,1,2])
        acc_1 = acc_metric.evaluate(fake_predict, fake_label_1) 
        acc_2 = acc_metric.evaluate(fake_predict, fake_label_2)
        wrong_acc = acc_metric.evaluate(fake_predict, fake_wrong_label) 
        self.assertEqual(acc_1, acc_2)
        self.assertEqual(wrong_acc, 8.0/9.0)
           
    
    def test_F1(self):
        test_f1_weight = metric.F1(f1={'average': 'weighted'})
        test_f1_micro = metric.F1(f1={'average': 'micro'})
        test_f1_macro = metric.F1(f1={'average': 'macro'})

        # binary test
        fake_data = np.array([[1, 2], [4, 5], [5, 5]])
        fake_predict = fake_data/np.sum(fake_data.astype(np.float32), axis=1)[:, None] #[1,1,0]
        fake_label = np.array([1,1,0])
        fake_wrong_label = np.array([0,1,0])

        acc = test_f1_weight.evaluate(fake_predict, fake_label)
        wrong_acc = test_f1_weight.evaluate(fake_predict, fake_wrong_label)
        self.assertEqual(acc, 1.0)
        self.assertEqual(wrong_acc, 2.0/3.0)

        # multi-label test
        fake_data = np.array([[3, 2, 1], [6, 5, 4], [2, 3, 2],[3, 2, 5], [3, 5, 4], [2, 3, 1],[3, 3, 5], [1, 5, 4], [2, 3, 6]])
        fake_predict = fake_data/np.sum(fake_data.astype(np.float32), axis=1)[:, None]
        fake_label = np.array([0,0,1,2,1,1,2,1,2])
        fake_wrong_label = np.array([0,0,0,0,1,1,1,2,2])
        
        acc_macro = test_f1_macro.evaluate(fake_predict, fake_label)
        self.assertEqual(acc_macro, 1.0)
        wrong_acc_macro = test_f1_macro.evaluate(fake_predict, fake_wrong_label)
        self.assertAlmostEqual(wrong_acc_macro, 2*(0.5/1.5+(0.5*2/3.0)/(0.5+2/3.0)+(0.5*1/3.0)/(0.5+1/3.0))/3) 

        acc_micro = test_f1_micro.evaluate(fake_predict, fake_label)
        self.assertEqual(acc_micro, 1.0)
        wrong_acc_micro = test_f1_micro.evaluate(fake_predict, fake_wrong_label)
        self.assertAlmostEqual(wrong_acc_micro, 2*(5/9.0 * 5/9.0)/(10/9.0))  

        acc_weight = test_f1_weight.evaluate(fake_predict, fake_label)
        self.assertEqual(acc_weight, 1.0)
        wrong_acc_weight = test_f1_weight.evaluate(fake_predict, fake_wrong_label)
        self.assertAlmostEqual(wrong_acc_weight, 2*(4*0.5/1.5+3*(0.5*2/3.0)/(0.5+2/3.0)+2*(0.5*1/3.0)/(0.5+1/3.0))/9)


if __name__ == "__main__":
    unittest.main()