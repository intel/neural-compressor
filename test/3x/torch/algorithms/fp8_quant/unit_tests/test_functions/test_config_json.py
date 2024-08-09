"""Use this module as an example of how to write new unit tests for layers."""

import habana_quantization_toolkit as hqt
import torch
from habana_quantization_toolkit._quant_common.helper_modules import Matmul
from habana_quantization_toolkit._quant_common.quant_config import QuantMode


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = Matmul()


def test_config_json():
    model = Model()

    for mode in [QuantMode.MEASURE, QuantMode.QUANTIZE]:
        name = {
            QuantMode.MEASURE: "measure",
            QuantMode.QUANTIZE: "quant",
        }[mode]
        config_path = f"llama_{name}"
        hqt.prep_model(model, config_path=config_path)
        hqt.finish_measurements(model)
