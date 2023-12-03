import shutil
import sys
import unittest
from copy import deepcopy

sys.path.insert(0, "./")
import torch
from torch.utils.data import DataLoader, Dataset

from neural_compressor import PostTrainingQuantConfig, quantization
from neural_compressor.adaptor.torch_utils.layer_wise_quant import load_empty_model
from neural_compressor.utils.pytorch import load


class TestLayerWise(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name_or_path = "facebook/opt-125m"
        self.fp32_model = load_empty_model(self.model_name_or_path, torchscript=True)

        class TestDataset(Dataset):
            def __init__(self, size=5, shape=128):
                self.len = size
                self.input_ids = torch.randint(low=0, high=30522, size=(size, shape), dtype=torch.int64)

            def __getitem__(self, index):
                return self.input_ids[index]

            def __len__(self):
                return self.len

        eval_dataset = TestDataset()
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./saved_model", ignore_errors=True)

    def test_rtn_lwq(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            recipes={
                "layer_wise_quant": True,
                # "layer_wise_quant_args": {
                #     "model_path": "facebook/opt-125m",
                # },
                "rtn_args": {"enable_full_range": True},
            },
        )

        q_model = quantization.fit(
            deepcopy(self.fp32_model),
            conf,
            calib_dataloader=self.eval_dataloader,
            eval_func=lambda x: 0.1,
        )
        ouput_dir = "./saved_model"
        q_model.save(ouput_dir)
        load_model = load(ouput_dir, deepcopy(self.fp32_model), weight_only=True)
        self.assertNotEqual(load_model.lm_head.weight.device.type, "meta")

    def test_gptq_lwq(self):
        conf = PostTrainingQuantConfig(
            approach="weight_only",
            op_type_dict={
                ".*": {  # re.match
                    "weight": {
                        "bits": 4,  # 1-8 bits
                        "group_size": 32,
                        "scheme": "sym",
                        "algorithm": "GPTQ",
                    },
                },
            },
            recipes={
                "gptq_args": {"actorder": True, "mse": True, "perchannel": False},
                "layer_wise_quant": True,
            },
        )
        q_model = quantization.fit(deepcopy(self.fp32_model), conf, calib_dataloader=self.eval_dataloader)
        ouput_dir = "./saved_model"
        q_model.save(ouput_dir)
        load_model = load(ouput_dir, deepcopy(self.fp32_model), weight_only=True, layer_wise=True)
        self.assertNotEqual(load_model.lm_head.weight.device.type, "meta")


if __name__ == "__main__":
    unittest.main()
