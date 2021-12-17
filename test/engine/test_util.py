import unittest
import numpy as np
from neural_compressor.adaptor.engine_utils.util import collate_preds

class TestUtil(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_collate_preds(self):
        fake_preds = np.random.randn(300, 32)
        res = collate_preds(fake_preds)
        self.assertEqual(int(res.shape[0]), 300*32)
    
if __name__ == "__main__":
    unittest.main()
