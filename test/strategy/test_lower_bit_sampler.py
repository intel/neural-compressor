"""Tests for new data type."""
import os
import shutil
import unittest


def build_model():
    import torch

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1)
            self.linear = torch.nn.Linear(224 * 224, 5)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(1, -1)
            x = self.linear(x)
            return x

    return M()


def add_cap(filename):
    import yaml

    int4_cap = {
        "static": {
            "Conv2d": {
                "weight": {
                    "dtype": ["int4"],
                    "scheme": ["sym"],
                    "granularity": ["per_channel"],
                    "algorithm": ["minmax"],
                },
                "activation": {
                    "dtype": ["uint4"],
                    "scheme": ["sym"],
                    "granularity": ["per_tensor"],
                    "algorithm": ["kl", "minmax"],
                },
            },
        }
    }

    with open(filename) as f:
        con = yaml.safe_load(f)
    con[0]["int4"] = int4_cap
    with open(filename, "w") as out:
        yaml.dump(con, out)


class TestLowerBitQuant(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        import importlib
        import shutil

        nc_path = os.path.dirname(importlib.util.find_spec("neural_compressor").origin)
        self.src = os.path.join(nc_path, "adaptor/pytorch_cpu.yaml")
        self.dst = os.path.join(nc_path, "adaptor/pytorch_cpu_backup.yaml")
        shutil.copyfile(self.src, self.dst)
        add_cap(self.src)

    @classmethod
    def tearDownClass(self):
        shutil.copyfile(self.dst, self.src)
        os.remove(self.dst)
        shutil.rmtree("saved", ignore_errors=True)

    def test_add_int4(self):
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 224, 224)))
        dataloader = DATALOADERS["pytorch"](dataset)
        model = build_model()

        acc_lst = [1, 1.1, 0.9, 1.1, 1.0]

        def fake_eval(model):
            res = acc_lst.pop(0)
            return res

        # tuning and accuracy criterion
        tuning_criterion = TuningCriterion(strategy="basic", timeout=10000, max_trials=2)
        conf = PostTrainingQuantConfig(quant_level=1, tuning_criterion=tuning_criterion)
        q_model = fit(model=model, conf=conf, calib_dataloader=dataloader, eval_func=fake_eval)
        self.assertIsNotNone(q_model)


if __name__ == "__main__":
    unittest.main()
