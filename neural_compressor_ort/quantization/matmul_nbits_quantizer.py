# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union

import onnx
import packaging
from onnxruntime.quantization import matmul_4bits_quantizer

from neural_compressor_ort import nc_version
from neural_compressor_ort.quantization import algorithm_entry as algos
from neural_compressor_ort.quantization import calibrate, config


class RTNWeightOnlyQuantConfig(matmul_4bits_quantizer.RTNWeightOnlyQuantConfig):

    def __init__(self, ratios=None, layer_wise_quant=False):
        super().__init__(ratios=ratios)
        self.layer_wise_quant = layer_wise_quant


class GPTQWeightOnlyQuantConfig(matmul_4bits_quantizer.GPTQWeightOnlyQuantConfig):

    def __init__(
        self,
        calibration_data_reader: calibrate.CalibrationDataReader,
        percdamp=0.01,
        blocksize=128,
        actorder=False,
        mse=False,
        perchannel=True,
        layer_wise_quant=False,
    ):
        super().__init__(
            calibration_data_reader=calibration_data_reader,
            percdamp=percdamp,
            blocksize=blocksize,
            actorder=actorder,
            mse=mse,
            perchannel=perchannel,
        )
        self.layer_wise_quant = layer_wise_quant


class AWQWeightOnlyQuantConfig(matmul_4bits_quantizer.WeightOnlyQuantConfig):

    def __init__(
        self,
        calibration_data_reader: calibrate.CalibrationDataReader,
        enable_auto_scale=True,
        enable_mse_search=True,
    ):
        super().__init__(algorithm="AWQ")
        self.calibration_data_reader = calibration_data_reader
        self.enable_auto_scale = enable_auto_scale
        self.enable_mse_search = enable_mse_search


algorithm_config_mapping = {
    "RTN": RTNWeightOnlyQuantConfig,
    "AWQ": AWQWeightOnlyQuantConfig,
    "GPTQ": GPTQWeightOnlyQuantConfig,
}


class MatMulNBitsQuantizer:

    def __init__(
        self,
        model: Union[onnx.ModelProto, str],
        block_size: int = 128,
        is_symmetric: bool = False,
        accuracy_level: int = 0,
        nodes_to_exclude: List[str] = None,
        algo_config: matmul_4bits_quantizer.WeightOnlyQuantConfig = None,
        n_bits: int = 4,
        providers: List[str] = ["CPUExecutionProvider"],
    ):
        if nodes_to_exclude is None:
            nodes_to_exclude = []
        self.model_path = model if isinstance(model, str) else None
        self.model = model
        self.model = onnx_model.ONNXModel(onnx.load(model)) if isinstance(model, str) else onnx_model.ONNXModel(model)
        self.block_size = block_size
        self.is_symmetric = is_symmetric
        self.accuracy_level = accuracy_level
        self.nodes_to_exclude = list(set(nodes_to_exclude))
        self.algo_config = algo_config or RTNWeightOnlyQuantConfig()
        self.n_bits = n_bits
        self.providers = providers
        self.algorithm = self.algo_config.algorithm
        assert self.algorithm in [
            "RTN",
            "AWQ",
            "GPTQ",
        ], "Only RTN, GPTQ and AWQ algorithms are supported, but get {} algorithm".format(self.algorithm)

    def _generate_nc_config(self):
        config_class = config.config_registry.get_cls_configs()[self.algorithm.lower()]

        quant_kwargs = {
            "weight_bits": self.n_bits,
            "weight_group_size": self.block_size,
            "weight_sym": self.is_symmetric,
            "accuracy_level": self.accuracy_level,
            "providers": self.providers,
        }
        if self.algorithm == "RTN":
            quant_kwargs.update(
                {
                    "layer_wise_quant": self.algo_config.layer_wise_quant,
                }
            )
        elif self.algorithm == "GPTQ":
            quant_kwargs.update(
                {
                    "percdamp": self.algo_config.percdamp,
                    "blocksize": self.algo_config.blocksize,
                    "actorder": self.algo_config.actorder,
                    "mse": self.algo_config.mse,
                    "perchannel": self.algo_config.perchannel,
                    "layer_wise_quant": self.algo_config.layer_wise_quant,
                }
            )
        elif self.algorithm == "AWQ":
            quant_kwargs.update(
                {
                    "enable_auto_scale": self.algo_config.enable_auto_scale,
                    "enable_mse_search": self.algo_config.enable_mse_search,
                }
            )
        nc_config = config_class(**quant_kwargs)

        if len(self.nodes_to_exclude) > 0:
            not_quant_kwargs = {"weight_dtype": "fp32", "white_list": self.nodes_to_exclude}
            nc_config += config_class(**not_quant_kwargs)

        return nc_config

    def int4_quant_algo(self):
        config = self._generate_nc_config()

        utility.logger.info(f"start to quantize model with {self.algorithm} algorithm...")
        model = self.model_path or self.model
        if self.algorithm == "RTN":
            self.model = algos.rtn_quantize_entry(model, config)
        elif self.algorithm == "GPTQ":
            self.model = algos.gptq_quantize_entry(model, config, self.algo_config.calibration_data_reader)
        elif self.algorithm == "AWQ":
            self.model = algos.awq_quantize_entry(model, config, self.algo_config.calibration_data_reader)
        utility.logger.info(f"complete quantization of model with {self.algorithm} algorithm.")

    def process(self):
        assert packaging.version.parse(nc_version.__version__) >= packaging.version.parse(
            "2.6"
        ), "Require neural-compressor >= 2.6 to support weight only quantization!"

        self.int4_quant_algo()
