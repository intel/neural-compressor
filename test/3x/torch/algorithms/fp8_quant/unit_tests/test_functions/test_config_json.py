"""Use this module as an example of how to write new unit tests for layers."""
import os
import torch
import neural_compressor.torch.algorithms.fp8_quant as fp8_quant
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import QuantMode
from neural_compressor.torch.algorithms.fp8_quant._quant_common.helper_modules import Matmul


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
        config_path = os.path.join(os.environ.get("NEURAL_COMPRESSOR_FORK_ROOT"),
                                   f"neural_compressor/torch/algorithms/fp8_quant/custom_config/llama_{name}.json")
        fp8_quant.prep_model(model, config_path=config_path)
        fp8_quant.finish_measurements(model)
