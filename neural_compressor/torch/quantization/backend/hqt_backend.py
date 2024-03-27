import os
from pathlib import Path
from typing import Union
from neural_compressor.torch.quantization.backend import BaseBackend, backend_register
from neural_compressor.torch.algorithms.habana_fp8 import get_mod_list, update_stats_path_in_config, update_mode

@backend_register(name="hqt")
class HQTBackend(BaseBackend):
    def __init__(self, quant_config):
        super().__init__(quant_config)

    def prepare(self, model):
        # set environment
        os.environ['QUANT_CONFIG'] = self.quant_config

        _prepare(model)
        return model

    def convert(self, model, calib_result):
        # set environment
        os.environ['QUANT_CONFIG'] = self.quant_config

        _convert(model, calib_result)
        return model

def _convert(model, calib_result):
    from habana_quantization_toolkit._hook_method import config, quantize_hooks, scale_method_mapping, scaling_params

    # update mode to QUANTIZE
    update_mode(quant_step=True)

    # update calibration result path
    update_stats_path_in_config(old_stats_path=config.cfg["dump_stats_path"], new_stats_path=calib_result)

    mod_list = get_mod_list(model)
    scaling_method_name = scale_method_mapping[(config.cfg['scale_method'], config.cfg['observer'])]
    scaling_params[scaling_method_name].update(config.cfg['scale_params'])
    config.cfg['scale_params'] = scaling_params[scaling_method_name]

    return quantize_hooks(model, mod_list)

def _prepare(model):
    from habana_quantization_toolkit._hook_method import prepare_model_for_measure

    # update mode to MEASURE
    update_mode(calib_step=True)

    mod_list = get_mod_list(model)

    return prepare_model_for_measure(model, mod_list)
