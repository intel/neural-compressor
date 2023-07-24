import copy
import unittest
import numpy as np
import shutil
import torch
import sys
import math
import transformers

sys.path.append('./')

from neural_compressor.data import Datasets, DATALOADERS
from neural_compressor.data.dataloaders.pytorch_dataloader import PyTorchDataLoader
from neural_compressor.adaptor.torch_utils.smooth_quant import TorchSmoothQuant
import logging
logger = logging.getLogger("neural_compressor")

try:
    import intel_extension_for_pytorch as ipex
    TEST_IPEX = True
except:
    TEST_IPEX = False


class DemoModel(torch.nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.fc1 = torch.nn.Linear(3, 4)
        self.fc2 = torch.nn.Linear(4, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

class DemoCalibDataloader:
    def __init__(self):
        self.batch_size = 1
    def __iter__(self):
        yield torch.randn([1, 3])


class LLMCalibDataloader:
    def __init__(self):
        self.batch_size = 1
    def __iter__(self):
        yield torch.ones([1, 3], dtype=torch.long)
        
class TestTuneSqAlpha(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.ns_workspace = "./nc_workspace"
    
    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.ns_workspace, ignore_errors=True)
        
    def test_sq_tune_alpha(self):
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        tuning_criterion = TuningCriterion(max_trials=5)

        for folding in [True, False]:
            for fp32_model, dataloader in [
                (DemoModel(), DemoCalibDataloader()), 
                # (
                #     transformers.AutoModelForCausalLM.from_pretrained(
                #         'facebook/opt-125m', torchscript=True,),
                #     LLMCalibDataloader()
                # )
            ]:
                conf = PostTrainingQuantConfig(
                    backend='ipex',
                    quant_level=1,
                    tuning_criterion=tuning_criterion,
                    calibration_sampling_size=8,
                    recipes={"smooth_quant": True,
                            "smooth_quant_args": {'folding': folding,
                                                "alpha": np.arange(0.1, 0.4, 0.05).tolist()}
                            }
                )
                eval_result_lst = [1, 0.9, 0.8, 0.7, 1.1]
                def fake_eval(model):
                    acc = eval_result_lst.pop(0)
                    return acc
                    
                q_model = quantization.fit(
                    fp32_model,
                    conf,
                    calib_dataloader=dataloader,
                    eval_func=fake_eval
                )
                q_model.save(self.ns_workspace + "saved_result")

    def test_sq_tune_alpha_common(self, eval_func, alpha=np.arange(0.1, 0.2, 0.05).tolist()):
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        tuning_criterion = TuningCriterion(max_trials=5)

        fp32_model = DemoModel()
        conf = PostTrainingQuantConfig(
            quant_level=1,
            tuning_criterion=tuning_criterion,
            calibration_sampling_size=8,
            recipes={"smooth_quant": True, 
                     "smooth_quant_args": {'folding': False,
                                           "alpha": alpha,
                                           }
                     }
        )
        q_model = quantization.fit(
            fp32_model,
            conf,
            calib_dataloader=DemoCalibDataloader(),
            eval_func=eval_func,
        )
        q_model.save(self.ns_workspace + "saved_result")

    def test_tune_sq_alpha(self):
        from functools import partial
        def fake_eval(model, eval_result_lst):
            acc = eval_result_lst.pop(0)
            return acc

        for eval_result_lst, note in [
                ([1, 0.8, 1.1, 0.7, 1.1], "Expect tuning ends at 2nd trial with alpha is 0.15"),
                ([1, 0.8, 0.9, 0.7, 1.1], "Expect tuning ends at 4th trial with alpha is 0.15"),
                ([1, 0.9, 0.8, 0.7, 1.1], "Expect tuning ends at 4th trial with alpha is 0.10")
                ]:
            logger.info(f"test_sq_tune_alpha_common with eval_result_lst: {eval_result_lst}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst = eval_result_lst )
            self.test_sq_tune_alpha_common(partial_fake_eval)

        for eval_result_lst, alpha, note in [
                ([1, 0.8, 1.1, 0.7, 1.1], 0.5 ,"Expect tuning ends at 2nd trial with alpha is 0.5 and not tune sq's alpha."),
                ([1, 0.8, 0.9, 0.7, 1.1], [0.5], "Expect tuning ends at 4th trial with alpha is  0.5 and not tune sq's alpha."),
                ([1, 0.9, 0.8, 0.7, 1.1], [0.5, 0.7, 0.9] ,"Expect tuning ends at 4th trial with alpha is 0.5")
                ]:
            logger.info(f"test_sq_tune_alpha_common with eval_result_lst: {eval_result_lst}, alpha: {alpha}")
            logger.info(note)
            partial_fake_eval = partial(fake_eval, eval_result_lst=eval_result_lst)
            self.test_sq_tune_alpha_common(partial_fake_eval, alpha=alpha)


if __name__ == '__main__':
    unittest.main()
