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


from ...utils.line_operation import get_line_indent_level

class TrainerToNLPTrainer(object):
    def __init__(self, file) -> None:
        self.file = file
        self.result = []

    def transform(self):
        lines = self.file.split('\n')

        for line in lines:
            if self.is_modify(line):
                new_line = self.modify(line)
                self.result.append(new_line)
            else:
                self.result.append(line)
        for index, line in enumerate(self.result):
            if index != len(self.result)-1:
                self.result[index] += '\n'
        return ''.join(self.result)

    def is_modify(self, s):
        if 'trainer = Trainer(' in s:
            return True
        else:
            return False

    def modify(self, s):
        old = 'Trainer'
        s = s.replace(old, 'NLPTrainer')
        return s
