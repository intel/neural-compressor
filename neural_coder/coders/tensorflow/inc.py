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

from ...utils.line_operation import get_line_indent_level, get_line_left_hand_side


class TensorFlowKerasINC(object):
    def __init__(self, file) -> None:
        self.file = file
        self.result = []

    def transform(self):
        # import pdb
        # pdb.set_trace()
        lines = self.file.split("\n")
        for line in lines:
            if self.is_modify(line):
                model_name = "model"
                indent_level = get_line_indent_level(line)
                self.result.append(line)
                self.result.append(" " * indent_level + "from neural_compressor.quantization import fit")
                self.result.append(" " * indent_level + "from neural_compressor.config import PostTrainingQuantConfig")
                self.result.append(" " * indent_level + "from neural_compressor import common")
                self.result.append(" " * indent_level + "config = PostTrainingQuantConfig(quant_level=1)")
                self.result.append(" " * indent_level + model_name + " = fit(" + model_name + ", conf=config)")
                self.result.append(" " * indent_level + model_name + '.save("./quantized_model")')
            else:
                self.result.append(line)
        for index, line in enumerate(self.result):
            if index != len(self.result) - 1:
                self.result[index] += "\n"
        return "".join(self.result)

    def is_modify(self, s):
        if "model = tf." in s or "model = load_model(" in s:
            if "self.model" not in s:
                return True
        else:
            return False
