"""Tests for coco_tools. """
import unittest
import numpy as np
from lpot.metric.coco_tools import *

class TestCOCO(unittest.TestCase):
    def testCOCOWrapper(self):
        with self.assertRaises(ValueError):
            wrap = COCOWrapper(None, 'test')

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

    def testExportSingleImageDetectionMasksToCoco(self):
        with self.assertRaises(ValueError):
            ExportSingleImageDetectionMasksToCoco(None, None, None, 
                    np.array([0]), np.array([[0,0]]))
        with self.assertRaises(ValueError):
            ExportSingleImageDetectionMasksToCoco(None, None, np.array([0,0]), 
                    np.array([0]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
