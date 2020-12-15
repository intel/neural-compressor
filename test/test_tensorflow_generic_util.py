#
#  -*- coding: utf-8 -*-
#
import unittest
import os
from lpot.adaptor.tf_utils.util import get_graph_def


class TestTFGenericUtil(unittest.TestCase):
    mb_model_url = 'https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_6/mobilenet_v1_1.0_224_frozen.pb'
    pb_path = '~/mobilenet_fp32.pb'

    @classmethod
    def setUpClass(self):
        self.saved_flag = True
        try:
            os.system("wget {} -O {} ".format(self.mb_model_url, self.pb_path))
        except Exception as e:
            self.saved_flag = False

    @classmethod
    def tearDownClass(self):
        if not self.saved_flag:
            os.system("rm -rf {}".format(self.pb_path))


    def test_ParsePbSavedUnderHomeDir(self):
        paresd_flag = False
        if self.saved_flag:
            try:
                get_graph_def(self.pb_path)
                paresd_flag = True
            except Exception as e:
                paresd_flag = False
        else:
            paresd_flag = True

        self.assertEqual(paresd_flag, True)


if __name__ == "__main__":
    unittest.main()
