"""Tests for HAWQ v2 strategy."""

import copy
import shutil
import unittest

from neural_compressor.utils import logger


# loss function for hawq-v2
def hawq_v2_loss(output, target):
    import torch

    return torch.nn.CrossEntropyLoss()(output, target)


class TestHAWQV2TuningStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        import torchvision

        self.model = torchvision.models.resnet18()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("saved", ignore_errors=True)
        shutil.rmtree("nc_workspace", ignore_errors=True)

    def test_hawq_v2_pipeline(self):
        logger.info("*** Test: HAWQ v2 with pytorch model.")
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # model
        model = copy.deepcopy(self.model)

        # fake evaluation function
        self.test_hawq_v2_pipeline_fake_acc = 0

        def _fake_eval(model):
            self.test_hawq_v2_pipeline_fake_acc -= 1
            return self.test_hawq_v2_pipeline_fake_acc

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((1, 3, 224, 224)))
        dataloader = DATALOADERS["pytorch"](dataset)

        # tuning and accuracy criterion
        strategy_kwargs = {"hawq_v2_loss": hawq_v2_loss}
        tuning_criterion = TuningCriterion(strategy="hawq_v2", strategy_kwargs=strategy_kwargs, max_trials=5)
        conf = PostTrainingQuantConfig(approach="static", quant_level=1, tuning_criterion=tuning_criterion)

        # fit
        q_model = fit(
            model=model, conf=conf, calib_dataloader=dataloader, eval_dataloader=dataloader, eval_func=_fake_eval
        )
        self.assertIsNone(q_model)


if __name__ == "__main__":
    unittest.main()
