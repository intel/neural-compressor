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


class EagerModeQuantizer:
    def prepare(self):
        pass

    def calibrate(self):
        pass

    def convert(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


from typing import Dict

from hqq_config import *


class HQQutizer(EagerModeQuantizer):
    def prepare(self, qconfig_mapping: Dict[str, HQQModuleConfig]):
        # Replace `Linear` with `HQQLinear`
        pass

    def convert(self):
        pass
