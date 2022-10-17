# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from . import domain
from ... import globals

class Calib_Dataloader(object):
    def __init__(self):
        pass
    def register_transformation(self):
        domain_ = domain.determine_domain(globals.list_code_path[0])
        if domain_ == 'transformers_trainer':
            globals.list_calib_dataloader_name.append('trainer.get_eval_dataloader()')
        elif domain_ == 'transformers_no_trainer':
            pass
        elif domain_ == 'torchvision':
            globals.list_calib_dataloader_name.append('val_loader')
        elif domain_ == 'onnx':
            codes = open(globals.list_code_path[0], 'r').read().split('\n')
            for line in codes:
                line  = line.strip()
                if 'loader' in line and '=' in line:
                    end = 0
                    for i in range(len(line)):
                        if line[i] == '=':
                            end = i
                    if line[end-1] == ' ':
                        globals.list_calib_dataloader_name.append(line[:end-1])
                    else:
                        globals.list_calib_dataloader_name.append(line[:end])
        else: # random model
            pass