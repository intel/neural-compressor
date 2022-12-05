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

from ...utils.line_operation import get_line_left_hand_side

class TensorFlowKerasINC(object):
    def __init__(self, file) -> None:
        self.file = file
        self.result = []

    def transform(self):
        # import pdb
        # pdb.set_trace()
        lines = self.file.split('\n')
        for line in lines:
            if self.is_modify(line):
                self.result.append(line)
                self.result.append("from neural_compressor.conf.config import QuantConf")
                self.result.append("from neural_compressor.experimental import Quantization")
                self.result.append("from neural_compressor.experimental import common")
                self.result.append("quant_config = QuantConf()")
                self.result.append("quant_config.usr_cfg.model.framework = 'tensorflow'")
                self.result.append("quantizer = Quantization(quant_config)")
                self.result.append("quantizer.model = common.Model(" + model_name + ")")
                self.result.append(model_name + " = quantizer.fit()")
            else:
                self.result.append(line)
        for index, line in enumerate(self.result):
            if index != len(self.result)-1:
                self.result[index] += '\n'
        return ''.join(self.result)

    def is_modify(self, s):
        if 'model = tf.' in s:
            return True
        else:
            return False
