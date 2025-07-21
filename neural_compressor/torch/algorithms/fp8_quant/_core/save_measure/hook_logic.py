# Copyright (c) 2025 Intel Corporation
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

import asyncio
import os
from threading import Thread

from neural_compressor.torch.algorithms.fp8_quant.utils.logger import logger
from neural_compressor.torch.algorithms.fp8_quant._quant_common.quant_config import get_hqt_config
from .save_files import (
    create_files_names, measure_control_to_state_dict, save_measurements_files, save_measurements,
    get_mod_extra_config_dict, gmod_list )


def dump_direct_call(model):
    save_measurements(model)

def dump_threading(model):
    t = Thread(target=save_measurements, args=(model,), daemon=True)
    t.start()

def dump_async_call(model):
    asyncio.run(dump_async_call_inner(model))

async def dump_async_call_inner(model):
    mcd = get_mod_extra_config_dict(model)
    await save_measurements_async_wrapper(model, mcd)

async def save_measurements_async_wrapper(model, mcd):
    config = get_hqt_config(model).cfg
    fname_base, fname_np, fname_list, measure_type = create_files_names(config)
    sd, sdl = measure_control_to_state_dict(mcd)
    save_measurements_files(model, sd, sdl, gmod_list, fname_np, fname_list, fname_base, measure_type)


def dump_shelv(model):
    pass

_measurement_dump_method = os.getenv("MEASUREMENT_DUMP_METHOD", "1")
_measurement_dump_method_dict = {
    "1": dump_direct_call,
    # below methods shouldn't be currently used as they are not fully completed
    "2": dump_threading,
    "3": dump_async_call,
    "5": dump_shelv
}


def _increment_calibration_samples_counter(model, *args):  # post hook function
    model.calibration_samples_counter += 1
    if model.calibration_samples_counter % model.calibration_sample_interval == 0:
        logger.debug("Reached sampling interval limit: %d, total samples: %d, dumping measurements.",
                     model.calibration_sample_interval, model.calibration_samples_counter)
        _measurement_dump_method_dict[_measurement_dump_method](model)
        logger.debug("finished dumping measurements.")

def add_calibration_samples_counter(model_to_calibrate, calibration_sample_interval):
    """
    Adds a forward post-hook to the model that counts the number of calibration samples processed.
    When the maximum number of samples is reached, it saves the measurements.
    """
    model_to_calibrate.calibration_samples_counter = 0
    model_to_calibrate.calibration_sample_interval = calibration_sample_interval
    model_to_calibrate.register_forward_hook(_increment_calibration_samples_counter)
    logger.info("Calibration samples interval added to the model - %d.", calibration_sample_interval)
