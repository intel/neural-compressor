"""Tests for distributed tuning strategy"""

import shutil
import unittest
import time

import numpy as np

from neural_compressor.utils import logger

from mpi4py import MPI

from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.data import Datasets, DATALOADERS
import torchvision

class TestDistributedTuning(unittest.TestCase):
    """Run: mpirun -np 2 python test_distributed_tuning.py TestDistributedTuning.test_pt_stage_1_met"""
    @classmethod
    def setUpClass(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
    
    def test_pt_stage_1_met(self):
        logger.info("*** Test: distributed tuning testing test_pt_stage_1_met start.")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        acc_lst =  [2.0, 1.0, 2.1, 2.2, 2.3, 2.0]
        perf_lst = [2.0, 1.5, 1.0, 0.5, 0.1, 4.0]

        self.test_pt_stage_1_met_index = -1
        def _fake_eval(model):
            self.test_pt_stage_1_met_index += 1
            print("perf,acc index: {}".format(self.test_pt_stage_1_met_index))
            perf = perf_lst[self.test_pt_stage_1_met_index]
            time.sleep(perf)
            return acc_lst[self.test_pt_stage_1_met_index]

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(use_distributed_tuning=True)
        # fit
        q_model = fit(model=resnet18,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        if self.rank == 0:
            self.assertIsNotNone(q_model)

    def test_pt_stage_3_fp32_met(self):
        logger.info("*** Test: distributed tuning testing test_pt_stage_3_fp32_met start.")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        acc_lst =  [2.0] + [1.0] * 16 + [2.0]
        perf_lst = [2.0] + [1.0] * 16 + [1.0]

        self.test_pt_stage_3_fp32_met_index = -1
        def _fake_eval(model):
            self.test_pt_stage_3_fp32_met_index += 1
            print("perf,acc index: {}".format(self.test_pt_stage_3_fp32_met_index))
            perf = perf_lst[self.test_pt_stage_3_fp32_met_index]
            time.sleep(perf)
            return acc_lst[self.test_pt_stage_3_fp32_met_index]

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(use_distributed_tuning=True)
        # fit
        q_model = fit(model=resnet18,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        if self.rank == 0:
            self.assertIsNotNone(q_model)

    def test_pt_stage_4_fp32_met(self):
        logger.info("*** Test: distributed tuning testing test_pt_stage_3_met start.")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        acc_lst =  [2.0] + [1.0] * 37 + [2.0]
        perf_lst = [2.0] + [1.0] * 37 + [1.0]

        self.test_pt_stage_3_met_index = -1
        def _fake_eval(model):
            self.test_pt_stage_3_met_index += 1
            print("perf,acc index: {}".format(self.test_pt_stage_3_met_index))
            perf = perf_lst[self.test_pt_stage_3_met_index]
            time.sleep(perf)
            return acc_lst[self.test_pt_stage_3_met_index]

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(use_distributed_tuning=True)
        # fit
        q_model = fit(model=resnet18,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        if self.rank == 0:
            self.assertIsNotNone(q_model)

    def test_pt_stage_not_met(self):
        logger.info("*** Test: distributed tuning testing test_pt_stage_not_met start.")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        acc_lst =  [2.0] + [1.0] * 57 + [2.0]
        perf_lst = [2.0] + [1.0] * 57 + [1.0]

        self.test_pt_stage_not_met_index = -1
        def _fake_eval(model):
            self.test_pt_stage_not_met_index += 1
            print("perf,acc index: {}".format(self.test_pt_stage_not_met_index))
            perf = perf_lst[self.test_pt_stage_not_met_index]
            time.sleep(perf)
            return acc_lst[self.test_pt_stage_not_met_index]

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(use_distributed_tuning=True)
        # fit
        q_model = fit(model=resnet18,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        if self.rank == 0:
            self.assertIsNone(q_model)  # None of the tuning configs met the requirements!

    def test_pt_met_wait_before_no_met(self):
        pass

    def test_pt_met_wait_before_met(self):
        pass

if __name__ == "__main__":
    unittest.main()
