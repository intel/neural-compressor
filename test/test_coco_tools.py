"""Tests for coco_tools. """
import unittest
import numpy as np
from lpot.experimental.metric.coco_tools import *

class TestCOCO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        groundtruth_annotations_list = [
            {
                'id': 1,
                'image_id': 1,
                'category_id': 1,
                'bbox': [387.99,97.43,84.99,81.29],
                'area': 2991.9213,
                'iscrowd': 0,
                'segmentation':[
                    [387.99,176.5,398.34,164.68,405.733,156.55,412.38,141.77,
                     419.77,136.6,424.94,125.51,432.33,116.64,434.55,102.6,
                     436.77,97.43,441.944,102.6,453.76,101.12,459.68,109.99,
                     457.46,115.9,463.37,124.03,470.76,128.47,472.98,137.34,
                     465.559,143.25,447.11,137.34,444.9,142.51,442.68,156.55,
                     444.9,163.2,446.37,176.5,444.9,178.72]
                 ]
            }
        ]
        image_list = [{'id': 1}]
        category_list = [{'id': 0, 'name': 'person'},
                         {'id': 1, 'name': 'cat'},
                         {'id': 2, 'name': 'dog'}]
        cls.groundtruth_dict = {
            'annotations': groundtruth_annotations_list,
            'images': image_list,
            'categories': category_list
        }
        cls.detections_list = [
            {
                'image_id': 1,
                'category_id': 1,
                'bbox': [387.99,97.43,84.99,81.29],
                'score': .8,
                'segmentation':[
                    [387.99,176.5,398.34,164.68,405.733,156.55,412.38,141.77,
                     419.77,136.6,424.94,125.51,432.33,116.64,434.55,102.6,
                     436.77,97.43,441.944,102.6,453.76,101.12,459.68,109.99,
                     457.46,115.9,463.37,124.03,470.76,128.47,472.98,137.34,
                     465.559,143.25,447.11,137.34,444.9,142.51,442.68,156.55,
                     444.9,163.2,446.37,176.5,444.9,178.72]
                 ]
 
            },
        ]

    def testCOCOWrapper(self):
        with self.assertRaises(ValueError):
            wrap = COCOWrapper(None, 'test')

        wrap = COCOWrapper(TestCOCO.groundtruth_dict)
        with self.assertRaises(ValueError):
            wrap.LoadAnnotations(TestCOCO.groundtruth_dict)
        wrong_detection = {
            'image_id': 'test',
            'category_id': 1,
            'bbox': [100., 100., 100., 100.],
            'score': .8
        }
        with self.assertRaises(ValueError):
            wrap.LoadAnnotations(wrong_detection)
        wrong_detection = [
            {
            'image_id': 'test',
            'category_id': 1,
            'bbox': [100., 100., 100., 100.],
            'score': .8
            }
        ]
        with self.assertRaises(ValueError):
            wrap.LoadAnnotations(wrong_detection)
 
        groundtruth = COCOWrapper(TestCOCO.groundtruth_dict)
        detections = groundtruth.LoadAnnotations(TestCOCO.detections_list)
        evaluator = COCOEvalWrapper(groundtruth, detections)
        self.assertEqual(evaluator.GetCategory(1)['name'], 'cat')
        self.assertTrue(not evaluator.GetAgnosticMode())
        self.assertEqual(evaluator.GetCategoryIdList(), [0, 1, 2])
        evaluator = COCOEvalWrapper(groundtruth, detections, agnostic_mode=True)
        self.assertTrue(evaluator.GetAgnosticMode())
        summary_metrics, _ = evaluator.ComputeMetrics()
        self.assertAlmostEqual(1.0, summary_metrics['Precision/mAP'])
        with self.assertRaises(ValueError):
            summary_metrics, _ = evaluator.ComputeMetrics(True, True)


    def testExportSingleImageDetectionBoxesToCoco(self):
        with self.assertRaises(ValueError):
            ExportSingleImageDetectionBoxesToCoco(None, None, None, 
                    np.array([0]), np.array([[0,0]]))
        with self.assertRaises(ValueError):
            ExportSingleImageDetectionBoxesToCoco(None, None, np.array([0,0]), 
                    np.array([0]), np.array([0]))
        with self.assertRaises(ValueError):
            ExportSingleImageDetectionBoxesToCoco(None, None, np.array([[0,0]]), 
                    np.array([0]), np.array([0]))
    
    def testExportSingleImageGroundtruthToCoco(self):
        with self.assertRaises(ValueError):
            ExportSingleImageGroundtruthToCoco(None, None, None, 
                    np.array([0,0]), np.array([0]))
        with self.assertRaises(ValueError):
            ExportSingleImageGroundtruthToCoco(None, None, None, 
                    np.array([[0,0]]), np.array([0]))
        with self.assertRaises(ValueError):
            ExportSingleImageGroundtruthToCoco(None, None, None, 
                np.array([[1,1,5,5]]), np.array([1]), np.array([[[1]]]), np.array([[1,0]]))
        ExportSingleImageGroundtruthToCoco(1, 2, [0,1,2], np.array([[1,1,5,5]]), 
                np.array([1]), np.array([[[1]]], dtype=np.uint8), np.array([1,0]))


    def testExportSingleImageDetectionMasksToCoco(self):
        with self.assertRaises(ValueError):
            ExportSingleImageDetectionMasksToCoco(None, None, None, 
                    np.array([0]), np.array([[0,0]]))
        with self.assertRaises(ValueError):
            ExportSingleImageDetectionMasksToCoco(None, None, np.array([0,0]), 
                    np.array([0]), np.array([0]))
        mask=[
            [387.99,176.5,398.34,164.68,405.733,156.55,412.38,141.77,
             419.77,136.6,424.94,125.51,432.33,116.64,434.55,102.6,
             436.77,97.43,441.944,102.6,453.76,101.12,459.68,109.99,
             457.46,115.9,463.37,124.03,470.76,128.47,472.98,137.34,
             465.559,143.25,447.11,137.34,444.9,142.51,442.68,156.55,
             444.9,163.2,446.37,176.5,444.9,178.72]
         ]
 
        result = ExportSingleImageDetectionMasksToCoco(
                    1, [0,1,2], mask, np.array([0.8]), np.array([1]))
        self.assertEqual(len(result), 1)

if __name__ == "__main__":
    unittest.main()
