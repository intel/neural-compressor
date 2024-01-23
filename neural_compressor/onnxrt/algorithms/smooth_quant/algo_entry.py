# Copyright (c) 2023 Intel Corporation
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

import tempfile
from pathlib import Path
from typing import Union

import onnx
from onnxruntime.quantization import StaticQuantConfig, quantize

from neural_compressor.common import Logger
from neural_compressor.common.utils import SMOOTH_QUANT

from neural_compressor.onnxrt.algorithms.smooth_quant.smooth_quant import ORTSmoothQuant
from neural_compressor.onnxrt.quantization import CalibrationDataReader
from neural_compressor.onnxrt.quantization.config import SmoohQuantQuantConfig
from neural_compressor.onnxrt.utils.utility import register_algo

logger = Logger().get_logger()


@register_algo(name=SMOOTH_QUANT)
def smooth_quant_entry(
    model: Union[Path, str],
    quant_config: SmoohQuantQuantConfig,
    calibration_data_reader: CalibrationDataReader,
    *args,
    **kwargs
) -> onnx.ModelProto:
    """The main entry to apply rtn quantization."""
    assert calibration_data_reader is not None, "Please provide calibration_data_reader"

    # smooth operation
    calibration_data_reader.rewind()
    sq = ORTSmoothQuant(
        model,
        calibration_data_reader,
        reduce_range=False,
        providers=quant_config.providers,
    )
    smooth_quant_model = sq.transform(**quant_config.to_dict())
    with tempfile.TemporaryDirectory(prefix="ort.quant.") as tmp_dir:
        # ORT quant API requires str input
        onnx.save_model(
            smooth_quant_model,
            Path(tmp_dir).joinpath("smooth.onnx").as_posix(),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="smooth.onnx_data",
            size_threshold=1024,
            convert_attribute=False,
        )

        # quant operation
        calibration_data_reader.rewind()

        # exclude Mul operations which are inserted during smooth operation
        excluded_nodes = [i.name for i in smooth_quant_model.graph.node if i.name.endswith("_smooth_mul")]
        config = StaticQuantConfig(
            calibration_data_reader=calibration_data_reader,
            nodes_to_exclude=excluded_nodes,
        )

        quantize(
            Path(tmp_dir).joinpath("smooth.onnx").as_posix(),
            Path(tmp_dir).joinpath("quant_model.onnx").as_posix(),
            config,
        )
        model = onnx.load(Path(tmp_dir).joinpath("quant_model.onnx").as_posix())

    return model
