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


class TestAddNewDataType(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("saved", ignore_errors=True)

    def test_add_int4(self):
        import importlib
        import shutil

        nc_path = os.path.dirname(importlib.util.find_spec("neural_compressor").origin)
        src = os.path.join(nc_path, "adaptor/pytorch_cpu.yaml")
        dst = os.path.join(nc_path, "adaptor/pytorch_cpu_backup.yaml")
        shutil.copyfile(src, dst)
        add_cap(src)
        from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
        from neural_compressor.data import DATALOADERS, Datasets
        from neural_compressor.quantization import fit

        # dataset and dataloader
        dataset = Datasets("pytorch")["dummy"](((100, 3, 224, 224)))
        dataloader = DATALOADERS["pytorch"](dataset)
        model = build_model()

        def fake_eval(model):
            return 1

        # tuning and accuracy criterion
        conf = PostTrainingQuantConfig()
        q_model = fit(model=model, conf=conf, calib_dataloader=dataloader, eval_func=fake_eval)
        shutil.copyfile(dst, src)
        os.remove(dst)
        self.assertIsNotNone(q_model)
        self.assertEqual(q_model._model.conv.zero_point, 7)


if __name__ == "__main__":
    unittest.main()
