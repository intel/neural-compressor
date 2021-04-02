"""Tests for the metrics module."""
import numpy as np
import unittest
from lpot.metric import METRICS
from lpot.experimental.metric.f1 import evaluate
from lpot.experimental.metric import bleu

class TestMetrics(unittest.TestCase):
    def testBLEU(self):
        metrics = METRICS('tensorflow')
        bleu = metrics['BLEU']()
        preds = ['Gutach: Mehr Sicherheit für Fußgänger']
        labels = ('Gutach: Noch mehr Sicherheit für Fußgänger',)
        bleu.update(preds, labels)
        self.assertAlmostEqual(bleu.result(), 51.1507809)
        bleu.reset()

        preds = ['Dies wurde auch von Peter Arnold vom Offenburg District Office bestätigt.']
        labels = ('Dies bestätigt auch Peter Arnold vom Landratsamt Offenburg.',)
        bleu.update(preds, labels)
        self.assertAlmostEqual(bleu.result(), 16.108992695)
        with self.assertRaises(ValueError):
            bleu.update(['a','b'], ('c',))

    def test_tensorflow_F1(self):
        metrics = METRICS('tensorflow')
        F1 = metrics['F1']()
        preds = [1, 1, 1, 1]
        labels = [0, 1, 1, 0]

        F1.update(preds, labels)
        self.assertEqual(F1.result(), 0.5)

    def test_squad_evaluate(self):
        label = [{'paragraphs':\
            [{'qas':[{'answers': [{'answer_start': 177, 'text': 'Denver Broncos'}, \
                                  {'answer_start': 177, 'text': 'Denver Broncos'}, \
                                  {'answer_start': 177, 'text': 'Denver Broncos'}], \
                      'question': 'Which NFL team represented the AFC at Super Bowl 50?', \
                      'id': '56be4db0acb8001400a502ec'}]}]}]
        preds = {'56be4db0acb8001400a502ec': 'Denver Broncos'}
        f1 = evaluate(preds, label)
        self.assertEqual(f1, 100.)

    def test_pytorch_F1(self):
        metrics = METRICS('pytorch')
        F1 = metrics['F1']()
        F1.reset()
        preds = [1, 1]
        labels = [2, 1, 1]

        F1.update(preds, labels)
        self.assertEqual(F1.result(), 0.8)

    def test_mxnet_F1(self):
        metrics = METRICS('mxnet')
        F1 = metrics['F1']()
        preds = [0, 1, 1, 1, 1, 0]
        labels = [0, 1, 1, 1]

        F1.update(preds, labels)
        self.assertEqual(F1.result(), 0.8)

    def test_onnx_topk(self):
        metrics = METRICS('onnxrt_qlinearops')
        top1 = metrics['topk']()
        top1.reset()
        self.assertEqual(top1.result(), 0)
        self.assertEqual(top1.result(), 0)
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


    def test_mxnet_topk(self):
        metrics = METRICS('mxnet')
        top1 = metrics['topk']()
        top1.reset()
        self.assertEqual(top1.result(), 0)
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

    def test_tensorflow_topk(self):
        metrics = METRICS('tensorflow')
        top1 = metrics['topk']()
        top1.reset()
        self.assertEqual(top1.result(), 0)
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
        
    def test_tensorflow_COCOmAP(self):
        import json
        import os
        metrics = METRICS('tensorflow')
        fake_dict = {
            'info': {},
            'licenses':{},
            'images':[{
                'file_name': '000000397133.jpg',
                'height': 100,
                'width': 100,
                'id': 397133
             }],
             'annotations':[{
                'category_id': 18,
                'id': 1768,
                'iscrowd': 0,
                'image_id': 397133,
                'bbox': [473.07, 395.93, 38.65, 28.67]
             }],
             'categories':[{
                'supercategory': 'animal',
                'id': 18,
                'name': 'dog'
             }]
        }
        fake_json = json.dumps(fake_dict)
        with open('anno.json', 'w') as f:
            f.write(fake_json)
        mAP = metrics['COCOmAP']('anno.json')
        self.assertEqual(mAP.category_map_reverse['dog'], 1)
        detection = [
            np.array([[5]]),
            np.array([[5]]),
            np.array([[[0.16117382, 0.59801614, 0.81511605, 0.7858219 ],
                        [0.5589304 , 0.        , 0.98301625, 0.520178  ],
                        [0.62706745, 0.35748824, 0.6892729 , 0.41513762],
                        [0.40032804, 0.01218696, 0.6924763 , 0.30341768],
                        [0.62706745, 0.35748824, 0.6892729 , 0.41513762]]]),
            np.array([[0.9267181 , 0.8510787 , 0.60418576, 0.35155892, 0.31158054]]),
            np.array([[ 1., 67., 51., 79., 47.]])
        ]
        ground_truth = [
            np.array([[[0.5633255 , 0.34003124, 0.69857144, 0.4009531 ],
                        [0.4763466 , 0.7769531 , 0.54334897, 0.9675937 ]]]),
            np.array([['a', 'b']]),            
            np.array([[]]),
            np.array([b'000000397133.jpg'])
        ]
        self.assertRaises(ValueError, mAP.update, detection, ground_truth)

        os.remove('anno.json')

        mAP = metrics['COCOmAP']()
        detection = [
            np.array([[[0.16117382, 0.59801614, 0.81511605, 0.7858219 ],
                        [0.5589304 , 0.        , 0.98301625, 0.520178  ],
                        [0.62706745, 0.35748824, 0.6892729 , 0.41513762],
                        [0.40032804, 0.01218696, 0.6924763 , 0.30341768],
                        [0.62706745, 0.35748824, 0.6892729 , 0.41513762]]]),
            np.array([[0.9267181 , 0.8510787 , 0.60418576, 0.35155892, 0.31158054]]),
            np.array([[ 1., 67., 51., 79., 47.]])
        ]
        detection_2 = [
            np.array([[8]]),
            np.array([[[0.82776225, 0.5865939 , 0.8927653 , 0.6302338 ],
                        [0.8375764 , 0.6424138 , 0.9055594 , 0.6921875 ],
                        [0.57902956, 0.39394334, 0.8342961 , 0.5577197 ],
                        [0.7949219 , 0.6513021 , 0.8472295 , 0.68427753],
                        [0.809729  , 0.5947042 , 0.8539927 , 0.62916476],
                        [0.7258591 , 0.08907133, 1.        , 0.86224866],
                        [0.43100086, 0.37782395, 0.8384069 , 0.5616918 ],
                        [0.32005906, 0.84334356, 1.        , 1.        ]]]),
            np.array([[0.86698544, 0.7562499 , 0.66414887, 0.64498234,\
                        0.63083494,0.46618757, 0.3914739 , 0.3094324 ]]),
            np.array([[55., 55., 79., 55., 55., 67., 79., 82.]])
        ]
        ground_truth = [
            np.array([[[0.5633255 , 0.34003124, 0.69857144, 0.4009531 ],
                        [0.56262296, 0.0015625 , 1.        , 0.5431719 ],
                        [0.16374707, 0.60728127, 0.813911  , 0.77823436],
                        [0.5841452 , 0.21182813, 0.65156907, 0.24670312],
                        [0.8056206 , 0.048875  , 0.90124124, 0.1553125 ],
                        [0.6729742 , 0.09317187, 0.7696956 , 0.21203125],
                        [0.3848478 , 0.002125  , 0.61522245, 0.303     ],
                        [0.61548007, 0.        , 0.7015925 , 0.097125  ],
                        [0.6381967 , 0.1865625 , 0.7184075 , 0.22534375],
                        [0.6274239 , 0.22104688, 0.71140516, 0.27134374],
                        [0.39566743, 0.24370313, 0.43578455, 0.284375  ],
                        [0.2673302 , 0.245625  , 0.3043794 , 0.27353126],
                        [0.7137705 , 0.15429688, 0.726815  , 0.17114063],
                        [0.6003747 , 0.25942189, 0.6438876 , 0.27320313],
                        [0.68845433, 0.13501562, 0.714637  , 0.17245312],
                        [0.69358313, 0.10959375, 0.7043091 , 0.12409375],
                        [0.493911  , 0.        , 0.72571427, 0.299     ],
                        [0.69576114, 0.15107812, 0.70714283, 0.16332813],
                        [0.4763466 , 0.7769531 , 0.54334897, 0.9675937 ]]]),
            np.array([[]]),
            np.array([[44, 67,  1, 49, 51, 51, 79, 1, 47, 47, 51, 51,\
                        56, 50, 56, 56, 79, 57, 81]]),            
            np.array([b'000000397133.jpg'])
        ]
        ground_truth_2 = [
            np.array([[[0.51508695, 0.2911648 , 0.5903478 , 0.31360796],
                        [0.9358696 , 0.07528409, 0.99891305, 0.25      ],
                        [0.8242174 , 0.3309659 , 0.93508697, 0.47301137],
                        [0.77413046, 0.22599432, 0.9858696 , 0.8179261 ],
                        [0.32582608, 0.8575    , 0.98426086, 0.9984659 ],
                        [0.77795655, 0.6268466 , 0.89930433, 0.73434657],
                        [0.5396087 , 0.39053977, 0.8483913 , 0.5615057 ],
                        [0.58473915, 0.75661933, 0.5998261 , 0.83579546],
                        [0.80391306, 0.6129829 , 0.8733478 , 0.66201705],
                        [0.8737391 , 0.6579546 , 0.943     , 0.7053693 ],
                        [0.775     , 0.6549716 , 0.8227391 , 0.6882955 ],
                        [0.8130869 , 0.58292615, 0.90526086, 0.62551135],
                        [0.7844348 , 0.68735796, 0.98182607, 0.83329546],
                        [0.872     , 0.6190057 , 0.9306522 , 0.6591761 ]]]),
            np.array([[]]),
            np.array([[64, 62, 62, 67, 82, 52, 79, 81, 55, 55, 55, 55, 62, 55]]),
            np.array([b'000000037777.jpg'])
        ]
        
        self.assertEqual(mAP.result(), 0)

        mAP.update(detection, ground_truth)
        
        mAP.update(detection, ground_truth)
        self.assertEqual(format(mAP.result(), '.5f'),
                            '0.14149')

        mAP.update(detection_2, ground_truth_2)
        self.assertEqual(format(mAP.result(), '.5f'),
                            '0.13366')
        mAP.reset()
        mAP.update(detection, ground_truth)
        self.assertEqual(format(mAP.result(), '.5f'),
                            '0.14149')

        ground_truth_1 = [
            np.array([[[0.51508695, 0.2911648 , 0.5903478 , 0.31360796],
                        [0.872     , 0.6190057 , 0.9306522 , 0.6591761 ]]]),
            np.array([[]]),
            np.array([[[64, 62]]]),
            np.array([b'000000037777.jpg'])
        ]
        self.assertRaises(ValueError, mAP.update, detection, ground_truth_1)
        ground_truth_2 = [
            np.array([[[0.51508695, 0.2911648 , 0.5903478 , 0.31360796],
                        [0.872     , 0.6190057 , 0.9306522 , 0.6591761 ]]]),
            np.array([[]]),
            np.array([[64]]),
            np.array([b'000000037700.jpg'])
        ]
        self.assertRaises(ValueError, mAP.update, detection, ground_truth_2)
        detection_1 = [
            np.array([[[0.16117382, 0.59801614, 0.81511605, 0.7858219 ],
                        [0.5589304 , 0.        , 0.98301625, 0.520178  ]]]),
            np.array([[0.9267181 , 0.8510787 , 0.60418576, 0.35155892, 0.31158054]]),
            np.array([[ 1., 67., 51., 79., 47.]])
        ]
        ground_truth_1 = [
            np.array([[[0.51508695, 0.2911648 , 0.5903478 , 0.31360796],
                        [0.872     , 0.6190057 , 0.9306522 , 0.6591761 ]]]),
            np.array([[]]),
            np.array([[64, 62]]),
            np.array([b'000000011.jpg'])
        ]
        self.assertRaises(ValueError, mAP.update, detection_1, ground_truth_1)
        ground_truth_2 = [
            np.array([[[0.51508695, 0.2911648 , 0.5903478 , 0.31360796],
                        [0.872     , 0.6190057 , 0.9306522 , 0.6591761 ]]]),
            np.array([[]]),
            np.array([[64, 62]]),
            np.array([b'000000012.jpg'])
        ]
        detection_2 = [
            np.array([[[0.16117382, 0.59801614, 0.81511605, 0.7858219 ],
                        [0.5589304 , 0.        , 0.98301625, 0.520178 ]]]),
            np.array([[0.9267181 , 0.8510787]]),
            np.array([[ 1., 67., 51., 79., 47.]])
        ]
        self.assertRaises(ValueError, mAP.update, detection_2, ground_truth_2)
 
    def test__accuracy(self):
        predicts1 = [1, 0, 1, 1]
        labels1 = [0, 1, 1, 1]

        predicts2 = [[0, 0], [0, 0]]
        labels2 = [[0, 1], [1, 1]]
        
        predicts3 = [[[0, 1], [0, 0], [0, 1]], [[0, 1], [0, 1], [0, 1]]]
        labels3 = [[[0, 1], [0, 1], [1, 0]], [[1, 0], [1, 0], [1, 0]]]

        predicts4 = [[0.2, 0.8], [0.1, 0.9], [0.3, 0.7], [0.4, 0.6]] #1,1,1,1
        labels4 = [0, 1, 0, 0]

        metrics = METRICS('pytorch')
        acc = metrics['Accuracy']()
        acc.update(predicts1, labels1)
        acc_result = acc.result()
        self.assertEqual(acc_result, 0.5)
        acc.reset()
        acc.update(predicts2, labels2)
        self.assertEqual(acc.result(), 0.25)
        acc.reset()
        acc.update(predicts3, labels3)
        self.assertEqual(acc.result(), 0.25)
        acc.reset()
        acc.update(predicts4, labels4)
        self.assertEqual(acc.result(), 0.25)

        metrics = METRICS('mxnet')
        acc = metrics['Accuracy']()
        acc.update(predicts1, labels1)
        acc_result = acc.result()
        self.assertEqual(acc_result, 0.5)
        acc.reset()
        acc.update(predicts2, labels2)
        self.assertEqual(acc.result(), 0.25)
        acc.reset()
        acc.update(predicts3, labels3)
        self.assertEqual(acc.result(), 0.25)
        acc.reset()
        acc.update(predicts4, labels4)
        self.assertEqual(acc.result(), 0.25)

        metrics = METRICS('onnxrt_qlinearops')
        acc = metrics['Accuracy']()
        acc.update(predicts1, labels1)
        acc_result = acc.result()
        self.assertEqual(acc_result, 0.5)
        acc.reset()
        acc.update(predicts2, labels2)
        self.assertEqual(acc.result(), 0.25)
        acc.reset()
        acc.update(predicts3, labels3)
        self.assertEqual(acc.result(), 0.25)
        acc.reset()
        acc.update(predicts4, labels4)
        self.assertEqual(acc.result(), 0.25)

        acc.reset()
        acc.update(1, 1)
        self.assertEqual(acc.result(), 1.0)
        
        wrong_predictions = [1, 0, 0]
        wrong_labels = [[0, 1, 1]]
        self.assertRaises(ValueError, acc.update, wrong_predictions, wrong_labels)
 

    def test_mxnet_accuracy(self):
        metrics = METRICS('mxnet')
        acc = metrics['Accuracy']()
        predicts = [1, 0, 1, 1]
        labels = [0, 1, 1, 1]
        acc.update(predicts, labels)
        acc_result = acc.result()
        self.assertEqual(acc_result, 0.5)

    def test_mse(self):
        predicts1 = [1, 0, 0, 1]
        labels1 = [0, 1, 0, 0]
        predicts2 = [1, 1, 1, 1]
        labels2 = [0, 1, 1, 0]

        metrics = METRICS('onnxrt_qlinearops')
        mse = metrics['MSE'](compare_label=False)
        mse.update(predicts1, labels1)
        mse_result = mse.result()
        self.assertEqual(mse_result, 1/(0.75+0.001))
        mse.update(predicts2, labels2)
        mse_result = mse.result()
        self.assertEqual(mse_result, 1/(0.625+0.001))
        
        metrics = METRICS('onnxrt_qlinearops')
        mse = metrics['MSE']()        
        mse.update(predicts1, labels1)
        mse_result = mse.result()
        self.assertEqual(mse_result, 0.75)
        mse.update(predicts2, labels2)
        mse_result = mse.result()
        self.assertEqual(mse_result, 0.625)
        
        metrics = METRICS('tensorflow')
        mse = metrics['MSE'](compare_label=False)
        mse.update(predicts1, labels1)
        mse_result = mse.result()
        self.assertEqual(mse_result, 1/(0.75+0.001))
        mse.update(predicts2, labels2)
        mse_result = mse.result()
        self.assertEqual(mse_result, 1/(0.625+0.001))

        metrics = METRICS('tensorflow')
        mse = metrics['MSE']()
        mse.update(predicts1, labels1)
        mse_result = mse.result()
        self.assertEqual(mse_result, 0.75)
        mse.update(predicts2, labels2)
        mse_result = mse.result()
        self.assertEqual(mse_result, 0.625)

        metrics = METRICS('mxnet')
        mse = metrics['MSE']()
        mse.update(predicts1, labels1)
        mse_result = mse.result()
        self.assertEqual(mse_result, 0.75)
        mse.update(predicts2, labels2)
        mse_result = mse.result()
        self.assertEqual(mse_result, 0.625)

        metrics = METRICS('pytorch')
        mse = metrics['MSE']()
        mse.update(predicts1, labels1)
        mse_result = mse.result()
        self.assertEqual(mse_result, 0.75)
        mse.update(predicts2, labels2)
        mse_result = mse.result()
        self.assertEqual(mse_result, 0.625)

    def test_mae(self):
        predicts1 = [1, 0, 0, 1]
        labels1 = [0, 1, 0, 0]
        predicts2 = [1, 1, 1, 1]
        labels2 = [1, 1, 1, 0]

        metrics = METRICS('tensorflow')
        mae = metrics['MAE']()
        mae.update(predicts1, labels1)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.75)
        mae.update(0, 1)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.8)
        mae.reset()
        mae.update(predicts2, labels2)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.25)

        metrics = METRICS('tensorflow')
        mae = metrics['MAE'](compare_label=False)
        mae.update(predicts1, labels1)
        mae_result = mae.result()
        self.assertEqual(mae_result, 1/(0.75+0.001))
        mae.update(0, 1)
        mae_result = mae.result()
        self.assertEqual(mae_result, 1/(0.8+0.001))
        mae.reset()
        mae.update(predicts2, labels2)
        mae_result = mae.result()
        self.assertEqual(mae_result, 1/(0.25+0.001))
        
        metrics = METRICS('pytorch')
        mae = metrics['MAE']()
        mae.update(predicts1, labels1)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.75)
        mae.update(predicts2, labels2)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.5)

        metrics = METRICS('mxnet')
        mae = metrics['MAE']()
        mae.update(predicts1, labels1)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.75)
        mae.update(predicts2, labels2)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.5)

        metrics = METRICS('onnxrt_qlinearops')
        mae = metrics['MAE']()
        mae.update(predicts1, labels1)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.75)
        mae.update(predicts2, labels2)
        mae_result = mae.result()
        self.assertEqual(mae_result, 0.5)

        metrics = METRICS('onnxrt_qlinearops')
        mae = metrics['MAE'](compare_label=False)
        mae.update(predicts1, labels1)
        mae_result = mae.result()
        self.assertEqual(mae_result, 1/(0.75+0.001))
        mae.update(predicts2, labels2)
        mae_result = mae.result()
        self.assertEqual(mae_result, 1/(0.5+0.001))

    def test_rmse(self):
        predicts1 = [1, 0, 0, 1]
        labels1 = [1, 0, 0, 0]
        predicts2 = [1, 1, 1, 1]
        labels2 = [1, 0, 0, 0]

        metrics = METRICS('tensorflow')
        rmse = metrics['RMSE']()
        rmse.update(predicts1, labels1)
        rmse_result = rmse.result()
        self.assertEqual(rmse_result, 0.5)
        rmse.reset()
        rmse.update(predicts2, labels2)
        rmse_result = rmse.result()
        self.assertAlmostEqual(rmse_result, np.sqrt(0.75))

        metrics = METRICS('tensorflow')
        rmse = metrics['RMSE'](compare_label=False)
        rmse.update(predicts1, labels1)
        rmse_result = rmse.result()
        self.assertEqual(rmse_result, np.sqrt(1/(0.25+0.001)))
        rmse.reset()
        rmse.update(predicts2, labels2)
        rmse_result = rmse.result()
        self.assertAlmostEqual(rmse_result, np.sqrt(1/(0.75+0.001)))

        metrics = METRICS('pytorch')
        rmse = metrics['RMSE']()
        rmse.update(predicts1, labels1)
        rmse_result = rmse.result()
        self.assertEqual(rmse_result, 0.5)
        rmse.update(predicts2, labels2)
        rmse_result = rmse.result()
        self.assertAlmostEqual(rmse_result, np.sqrt(0.5))

        metrics = METRICS('mxnet')
        rmse = metrics['RMSE']()
        rmse.update(predicts1, labels1)
        rmse_result = rmse.result()
        self.assertEqual(rmse_result, 0.5)
        rmse.update(predicts2, labels2)
        rmse_result = rmse.result()
        self.assertAlmostEqual(rmse_result, np.sqrt(0.5))

        metrics = METRICS('onnxrt_qlinearops')
        rmse = metrics['RMSE']()
        rmse.update(predicts1, labels1)
        rmse_result = rmse.result()
        self.assertEqual(rmse_result, 0.5)
        rmse.update(predicts2, labels2)
        rmse_result = rmse.result()
        self.assertAlmostEqual(rmse_result, np.sqrt(0.5))

        metrics = METRICS('onnxrt_qlinearops')
        rmse = metrics['RMSE'](compare_label=False)
        rmse.update(predicts1, labels1)
        rmse_result = rmse.result()
        self.assertEqual(rmse_result, np.sqrt(1/(0.25+0.001)))
        rmse.update(predicts2, labels2)
        rmse_result = rmse.result()
        self.assertAlmostEqual(rmse_result, np.sqrt(1/(0.5+0.001)))

    def test_loss(self):
        metrics = METRICS('pytorch')
        loss = metrics['Loss']()
        predicts = [1, 0, 0, 1]
        labels = [0, 1, 0, 0]
        loss.update(predicts, labels)
        loss_result = loss.result()
        self.assertEqual(loss_result, 0.5)
        predicts = [1, 1, 0, 1]
        labels = [0, 1, 0, 0]
        loss.update(predicts, labels)
        loss_result = loss.result()
        self.assertEqual(loss_result, 0.625)
        loss.reset()
        predicts = [1, 0, 0, 1]
        labels = [0, 1, 0, 0]
        loss.update(predicts, labels)
        self.assertEqual(loss.result(), 0.5)

       
        metrics = METRICS('onnxrt_qlinearops')
        loss = metrics['Loss']()
        predicts = [1, 0, 0, 1]
        labels = [0, 1, 0, 0]
        loss.update(predicts, labels)
        loss_result = loss.result()
        self.assertEqual(loss_result, 0.5)
        predicts = [1, 1, 0, 1]
        labels = [0, 1, 0, 0]
        loss.update(predicts, labels)
        loss_result = loss.result()
        self.assertEqual(loss_result, 0.625)
        loss.reset()
        predicts = [1, 0, 0, 1]
        labels = [0, 1, 0, 0]
        loss.update(predicts, labels)
        self.assertEqual(loss.result(), 0.5)

if __name__ == "__main__":
    unittest.main()
