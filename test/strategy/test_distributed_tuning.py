"""Tests for distributed tuning strategy."""
import importlib
import os
import re
import shutil
import signal
import subprocess
import sys
import unittest

import cpuinfo

from neural_compressor.utils import logger

if importlib.util.find_spec("mpi4py") is None:
    CONDITION = True
else:
    # from mpi4py import MPI
    CONDITION = False


def build_fake_ut():
    fake_ut = """
import shutil
import unittest
import time
import os
import sys

import numpy as np

from neural_compressor.utils import logger
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.data import Datasets, DATALOADERS
import torchvision

import importlib
if importlib.util.find_spec("mpi4py") is None:
    CONDITION = True
else:
    from mpi4py import MPI
    CONDITION = False

def save_acc_perf_to_local(acc_lst, perf_lst, acc_perf_data_file_path):
    import json
    data = {'acc_lst': acc_lst, 'perf_lst': perf_lst}
    with open(acc_perf_data_file_path, 'w') as fp:
        json.dump(data, fp)
        logger.info(f"Save data to {acc_perf_data_file_path}")

def next_acc_and_perf(acc_perf_data_file_path):
    import json
    acc, perf = None, None
    with open(acc_perf_data_file_path, 'r') as fp:
        data = json.load(fp)
        acc = data['acc_lst'][0]
        perf = data['perf_lst'][0]
    new_acc_lst = data['acc_lst'][1:]
    new_perf_lst = data['perf_lst'][1:]
    save_acc_perf_to_local(new_acc_lst, new_perf_lst, acc_perf_data_file_path)
    return acc, perf

@unittest.skipIf(CONDITION , "missing the mpi4py package")
class TestDistributedTuning(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    @classmethod
    def tearDownClass(self):
        if self.rank == 0:
            if os.path.exists('test_pt_stage_1_met.json'):
                os.remove('test_pt_stage_1_met.json')
            if os.path.exists('test_pt_stage_3_fp32_met.json'):
                os.remove('test_pt_stage_3_fp32_met.json')
            if os.path.exists('test_pt_stage_stage_4_fp32_met.json'):
                os.remove('test_pt_stage_stage_4_fp32_met.json')
            if os.path.exists('test_pt_stage_not_met.json'):
                os.remove('test_pt_stage_not_met.json')
            if os.path.exists('test_pt_num_of_nodes_more_than_len_of_tune_cfg_lst_met.json'):
                os.remove('test_pt_num_of_nodes_more_than_len_of_tune_cfg_lst_met.json')

    def test_mpi4py_installation(self):
        logger.info(f"Test rank {self.rank} of {self.size} processes")
        self.assertGreater(self.size, 0)
        self.assertGreaterEqual(self.size, 0)

    def test_pt_stage_1_met(self):
        logger.info("*** Test: distributed tuning testing test_pt_stage_1_met start.")

        num_processes = 3
        logger.info(f"*** Test: distributed tuning testing test_pt_stage_1_met start. NP: {num_processes} (load acc and perf from local).")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        num_baseline = num_processes # TODO, replace num_baseline with 1 when evaluating baseline only once.
        acc_lst =  [2.0] * num_baseline + [1.0, 2.1, 2.2, 2.3, 2.0] #the tuning result (2.1)
        perf_lst = [2.0] * num_baseline + [2.5, 2.0, 1.5, 1.1, 5.0]

        # make sure this path can be accessed by all nodes
        acc_perf_data_file_path = 'test_pt_stage_1_met.json'
        save_acc_perf_to_local(acc_lst, perf_lst, acc_perf_data_file_path)

        def _fake_eval(model):
            acc, perf = next_acc_and_perf(acc_perf_data_file_path)
            logger.info(f"Current evaluate result: acc {acc}, perf: {perf}.")
            time.sleep(perf)
            return acc

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(quant_level=1)
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

        num_processes = 3
        logger.info(f"*** Test: distributed tuning testing test_pt_stage_1_met start. NP: {num_processes} (load acc and perf from local).")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        num_baseline = num_processes # TODO, replace num_baseline with 1 when evaluating baseline only once.
        acc_lst =  [2.0] * num_baseline + [1.0] * 16 + [2.0, 1.0, 1.0]
        perf_lst = [2.0] * num_baseline + [1.0] * 16 + [1.0, 1.0, 1.0]

        # make sure this path can be accessed by all nodes
        acc_perf_data_file_path = 'test_pt_stage_3_fp32_met.json'
        save_acc_perf_to_local(acc_lst, perf_lst, acc_perf_data_file_path)

        def _fake_eval(model):
            acc, perf = next_acc_and_perf(acc_perf_data_file_path)
            logger.info(f"Current evaluate result: acc {acc}, perf: {perf}.")
            time.sleep(perf)
            return acc

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(quant_level=1)
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

        num_processes = 3
        logger.info(f"*** Test: distributed tuning testing test_pt_stage_1_met start. NP: {num_processes} (load acc and perf from local).")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        num_baseline = num_processes # TODO, replace num_baseline with 1 when evaluating baseline only once.
        acc_lst =  [2.0] * num_baseline + [1.0] * 37 + [2.0, 1.0, 1.0]
        perf_lst = [2.0] * num_baseline + [1.0] * 37 + [1.0, 1.0, 1.0]

        # make sure this path can be accessed by all nodes
        acc_perf_data_file_path = 'test_pt_stage_stage_4_fp32_met.json'
        save_acc_perf_to_local(acc_lst, perf_lst, acc_perf_data_file_path)

        def _fake_eval(model):
            acc, perf = next_acc_and_perf(acc_perf_data_file_path)
            logger.info(f"Current evaluate result: acc {acc}, perf: {perf}.")
            time.sleep(perf)
            return acc

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(quant_level=1)
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
        num_processes = 3
        logger.info(f"*** Test: distributed tuning testing test_pt_stage_1_met start. NP: {num_processes} (load acc and perf from local).")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        num_baseline = num_processes # TODO, replace num_baseline with 1 when evaluating baseline only once.
        acc_lst =  [2.0] * num_baseline + [1.0] * 60
        perf_lst = [2.0] * num_baseline + [1.0] * 60

        # make sure this path can be accessed by all nodes
        acc_perf_data_file_path = 'test_pt_stage_not_met.json'
        save_acc_perf_to_local(acc_lst, perf_lst, acc_perf_data_file_path)

        def _fake_eval(model):
            acc, perf = next_acc_and_perf(acc_perf_data_file_path)
            logger.info(f"Current evaluate result: acc {acc}, perf: {perf}.")
            time.sleep(perf)
            return acc

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(quant_level=1)
        # fit
        q_model = fit(model=resnet18,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        if self.rank == 0:
            self.assertIsNone(q_model)  # None of the tuning configs met the requirements!

    def test_pt_num_of_nodes_more_than_len_of_tune_cfg_lst_met(self):
        logger.info("*** Test: distributed tuning testing test_pt_num_of_nodes_more_than_len_of_tune_cfg_lst_met start.")

        num_processes = 18
        logger.info(f"*** Test: distributed tuning testing test_pt_stage_1_met start. NP: {num_processes} (load acc and perf from local).")

        # model
        resnet18 = torchvision.models.resnet18()

        # fake evaluation function
        num_baseline = num_processes # TODO, replace num_baseline with 1 when evaluating baseline only once.
        acc_lst =  [2.0] * num_baseline + [1.0] * 37 + [2.0, 1.0, 1.0] * 6
        perf_lst = [2.0] * num_baseline + [1.0] * 37 + [1.0, 1.0, 1.0] * 6

        # make sure this path can be accessed by all nodes
        acc_perf_data_file_path = 'test_pt_num_of_nodes_more_than_len_of_tune_cfg_lst_met.json'
        save_acc_perf_to_local(acc_lst, perf_lst, acc_perf_data_file_path)

        def _fake_eval(model):
            acc, perf = next_acc_and_perf(acc_perf_data_file_path)
            logger.info(f"Current evaluate result: acc {acc}, perf: {perf}.")
            time.sleep(perf)
            return acc

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 3, 1)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig(quant_level=1)
        # fit
        q_model = fit(model=resnet18,
                      conf=conf,
                      calib_dataloader= dataloader,
                      eval_dataloader=dataloader,
                      eval_func=_fake_eval)
        if self.rank == 0:
            self.assertIsNotNone(q_model)

    def test_pt_met_wait_before_no_met(self):
        pass

    def test_pt_met_wait_before_met(self):
        pass

if __name__ == "__main__":
    unittest.main()
    """

    with open("fake_ut.py", "w", encoding="utf-8") as f:
        f.write(fake_ut)


class TestDistributedTuning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_fake_ut()

    @classmethod
    def tearDownClass(cls):
        os.remove("fake_ut.py")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def setUp(self):
        logger.info(f"CPU: {cpuinfo.get_cpu_info()['brand_raw']}")
        logger.info(f"Test: {sys.modules[__name__].__file__}-{self.__class__.__name__}-{self._testMethodName}")

    def tearDown(self):
        logger.info(f"{self._testMethodName} done.\n")

    @unittest.skipIf(CONDITION, "missing the mpi4py package")
    def test_distributed_tuning(self):
        distributed_cmds = [
            "mpirun -np 3 python fake_ut.py TestDistributedTuning.test_mpi4py_installation",
            "mpirun -np 3 python fake_ut.py TestDistributedTuning.test_pt_stage_1_met",
            "mpirun -np 3 python fake_ut.py TestDistributedTuning.test_pt_stage_3_fp32_met",
            "mpirun -np 3 python fake_ut.py TestDistributedTuning.test_pt_stage_4_fp32_met",
            "mpirun -np 3 python fake_ut.py TestDistributedTuning.test_pt_stage_not_met",
            "mpirun -np 18 python fake_ut.py TestDistributedTuning.test_pt_num_of_nodes_more_than_len_of_tune_cfg_lst_met",
        ]

        for i, distributed_cmd in enumerate(distributed_cmds):
            p = subprocess.Popen(
                distributed_cmd, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
            )  # nosec
            try:
                out, error = p.communicate()
                logger.info(f"Test command: {distributed_cmd}")
                logger.info(out.decode("utf-8"))
                logger.info(error.decode("utf-8"))
                matches = re.findall(r"FAILED", error.decode("utf-8"))
                self.assertEqual(matches, [])

                matches = re.findall(r"OK", error.decode("utf-8"))
                if i == len(distributed_cmds) - 1:
                    self.assertTrue(len(matches) == 18)
                elif i == 0:
                    rank_match = re.findall("rank (\d+) of", error.decode("utf-8"))
                    size_match = re.findall("of (\d+) processes", error.decode("utf-8"))
                    self.assertEqual(sorted(rank_match), ["0", "1", "2"])
                    self.assertEqual(size_match, ["3"] * 3)
                else:
                    self.assertTrue(len(matches) == 3)

            except KeyboardInterrupt:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                assert 0


if __name__ == "__main__":
    unittest.main()
