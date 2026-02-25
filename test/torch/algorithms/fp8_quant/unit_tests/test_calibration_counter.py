from pathlib import Path
import torch

import habana_frameworks.torch.core as htcore

from neural_compressor.torch.quantization import FP8Config, prepare

class MyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 200, bias=False)
        self.fc2 = torch.nn.Linear(10, 200, bias=True)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(inp)
        return x2

calibration_counter = 50

config_dict = {
    "mode": "MEASURE",
    "observer": "maxabs",
    "dump_stats_path": "",
    "calibration_sample_interval": str(calibration_counter)
}

import time

def test_calibration_counter(inc_output_handler):
    config_dict["dump_stats_path"] = str(inc_output_handler)
    dump_stats_path_file_path = Path(config_dict["dump_stats_path"] + "_hooks_maxabs.npz")

    model = MyModel()
    config = FP8Config.from_dict(config_dict)
    data_set_size = calibration_counter * 2
    prepared_model = prepare(model, config)
    data_set = [torch.randn(10) for _ in range(data_set_size)]

    for i in range(1, data_set_size):
        prepared_model(data_set[i])
        if i < calibration_counter:
            # check no file was created
            assert not Path.exists(dump_stats_path_file_path)
        elif i == calibration_counter:
            time.sleep(1) # wait for file to be created
            # check file was created
            assert Path.exists(dump_stats_path_file_path)

