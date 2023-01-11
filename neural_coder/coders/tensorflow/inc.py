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

from ...utils.line_operation import get_line_left_hand_side, get_line_indent_level

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
                model_name = "model"
                indent_level = get_line_indent_level(line)
                self.result.append(line)
                self.result.append(" " * indent_level + "from neural_compressor.conf.config import QuantConf")
                self.result.append(" " * indent_level + "from neural_compressor.experimental import Quantization")
                self.result.append(" " * indent_level + "from neural_compressor.experimental import common")
                self.result.append(" " * indent_level + "quant_config = QuantConf()")
                self.result.append(" " * indent_level + "quant_config.usr_cfg.model.framework = 'tensorflow'")
                self.result.append(" " * indent_level + "quantizer = Quantization(quant_config)")
                self.result.append(" " * indent_level + "quantizer.model = common.Model(" + model_name + ")")
                self.result.append(" " * indent_level + model_name + " = quantizer.fit()")
            else:
                self.result.append(line)
        for index, line in enumerate(self.result):
            if index != len(self.result)-1:
                self.result[index] += '\n'
        return ''.join(self.result)

    def is_modify(self, s):
        if 'model = tf.' in s or 'model = load_model(' in s:
            if 'self.model' not in s:
                return True
        else:
            return False
