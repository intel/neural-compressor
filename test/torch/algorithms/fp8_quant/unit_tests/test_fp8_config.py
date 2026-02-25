import habana_frameworks.torch.core as htcore
import pytest
import torch

htcore.hpu_set_env()

from neural_compressor.torch.quantization import FP8Config, prepare


# test purpose is to validate that FP8Config parsing from dict succeeds when fake quant default value is given
def test_fakequant_default_config_from_dict():

    config_dict_no_fake_quant = {
        "mode": "AUTO",
        "observer": "maxabs",
        "scale_method": "maxabs_hw",
        "allowlist": {"types": [], "names": []},
        "blocklist": {"types": [], "names": []},
        "dump_stats_path": "./inc_output/measure",
    }  # this config file is to test the default behaviour

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.my_linear = torch.nn.Linear(in_features=32, out_features=32, bias=False, device="hpu")

        def forward(self, input):
            return self.my_linear(input)

    model = MyModel()
    htcore.hpu_initialize()
    config = FP8Config.from_dict(config_dict_no_fake_quant)
    try:
        prepare(model, config)
    except Exception as e:
        pytest.fail("error during config parsing - {}".format(e))
