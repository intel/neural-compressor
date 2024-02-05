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

import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import onnx

from neural_compressor.common.base_config import BaseConfig, get_all_config_set_from_config_registry
from neural_compressor.common.base_tuning import TuningConfig, evaluator, init_tuning
from neural_compressor.common.utils import logger
from neural_compressor.onnxrt.quantization.calibrate import CalibrationDataReader
from neural_compressor.onnxrt.quantization.config import FRAMEWORK_NAME
from neural_compressor.onnxrt.quantization.quantize import _quantize

__all__ = [
    "autotune",
    "get_all_config_set",
]


def get_all_config_set() -> Union[BaseConfig, List[BaseConfig]]:
    return get_all_config_set_from_config_registry(fwk_name=FRAMEWORK_NAME)


def autotune(
    model_input: Union[Path, str],
    tune_config: TuningConfig,
    eval_fns: Union[Dict, List[Dict], Callable] = None,
    calibration_data_reader: CalibrationDataReader = None,
) -> Union[None, onnx.ModelProto]:
    """The main entry of auto-tune.

    Args:
        model_input (Union[Path, str]): onnx model path.
        tune_config (TuningConfig): tuning config.
            TuningConfig is created with algorithm configs, parameters supported tuning are in their params_list.
            Support:
            Expand parameters to a list of parameters like TuningConfig(config_set=[RTNConfig(weight_bits=[4, 8])])
            Pass a list of configs like TuningConfig(config_set=[RTNConfig(), GPTQConfig()])
        eval_fns (Union[Dict, List[Dict], Callable]): evaluate functions.
            During evaluation, autotune will only pass model path as input into eatch function.
            Support:
            single eval function,
            Dict like {"eval_fn": eval_acc} or {"eval_fn": eval_acc, "weight": 1.0, "name": "accuracy"},
            List of Dict, like [
                {"eval_fn": eval_acc, "weight": 0.5},
                {"eval_fn": eval_perf, "weight": 0.5, "name": "accuracy"},
                ]
        calibration_data_reader (CalibrationDataReader): dataloader for calibration.
    """
    best_quant_model = None
    evaluator.set_eval_fn_registry(eval_fns)
    evaluator.self_check()
    config_loader, tuning_logger, tuning_monitor = init_tuning(tuning_config=tune_config)
    baseline: float = evaluator.evaluate(model_input)
    tuning_monitor.set_baseline(baseline)
    tuning_logger.tuning_start()
    for trial_index, quant_config in enumerate(config_loader):
        if calibration_data_reader is not None:
            calibration_data_reader.rewind()
        tuning_logger.trial_start(trial_index=trial_index)
        tuning_logger.quantization_start()
        logger.debug("quant config: {}".format(quant_config))
        q_model = _quantize(model_input, quant_config=quant_config, calibration_data_reader=calibration_data_reader)
        tuning_logger.quantization_end()
        tuning_logger.evaluation_start()
        with tempfile.TemporaryDirectory(prefix="ort.quant.") as tmp_dir:
            # evaluate API requires str input
            onnx.save_model(
                q_model,
                Path(tmp_dir).joinpath(Path(model_input).name).as_posix(),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=Path(model_input).with_suffix(Path(model_input).suffix + "_data").name,
                size_threshold=1024,
                convert_attribute=False,
            )
            # copy config.json to tmp dir for evaluation, LLMs evaluation may need it
            if isinstance(model_input, str) and os.path.exists(
                Path(model_input).parent.joinpath("config.json").as_posix()
            ):
                import shutil

                shutil.copyfile(
                    Path(model_input).parent.joinpath("config.json").as_posix(),
                    Path(tmp_dir).joinpath("config.json").as_posix(),
                )
            eval_result: float = evaluator.evaluate(Path(tmp_dir).joinpath(Path(model_input).name).as_posix())
        tuning_logger.evaluation_end()
        logger.info("Evaluation result: %.4f", eval_result)
        tuning_monitor.add_trial_result(trial_index, eval_result, quant_config)
        tuning_logger.trial_end(trial_index)
        if tuning_monitor.need_stop():
            best_quant_config: BaseConfig = tuning_monitor.get_best_quant_config()
            best_quant_model = _quantize(
                model_input, quant_config=best_quant_config, calibration_data_reader=calibration_data_reader
            )
            break
    tuning_logger.tuning_end()
    return best_quant_model
