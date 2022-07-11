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


class Lightning(object):
    def __init__(self, file) -> None:
        self.file = file
        self.result = []

    def transform(self):
        lines = self.file.split('\n')
        for line in lines:
            if self.not_add_accelerator(line) or self.not_add_precision(line):
                new_line = self.add(line)
                if self.not_modify(new_line):
                    new_line = self.modify(new_line)
                self.result.append(new_line)
            elif self.not_modify(line):
                new_line = self.modify(line)
                self.result.append(new_line)
            if not self.not_add_accelerator(line) and not self.not_add_precision(line) and not self.not_modify(line):
                if line == '' and self.result[-1] == '':
                    continue
                self.result.append(line)

        for index, line in enumerate(self.result):
            if index != len(self.result)-1:
                self.result[index] += '\n'
        return ''.join(self.result)

    def not_add_precision(self, s):
        if 'Trainer' in s:
            if 'precision' not in s:
                return True
            else:
                return False
        return False

    def not_add_accelerator(self, s):
        if 'Trainer' in s:
            if 'accelerator' not in s:
                return True
            else:
                return False
        return False

    def add(self, s):
        if 'Trainer' in s:
            if 'precision' not in s:
                s_index = s.find(')')
                s = s[:s_index] + ', precision=\"bf16\"' + s[s_index:]
            if 'accelerator' not in s:
                s_index = s.find(')')
                s = s[:s_index] + ', accelerator=\"cpu\"' + s[s_index:]
        return s

    def not_modify(self, s):
        if 'bf16' in s and 'cpu' in s:
            return False
        return True

    def modify(self, s):
        if '16' in s:
            old = '16'
            s = s.replace(old, '\"bf16\"')
        if '32' in s:
            old = '32'
            s = s.replace(old, '\"bf16\"')
        if '\"gpu\"' in s:
            old = '\"gpu\"'
            s = s.replace(old, '\"cpu\"')
        if '\"tpu\"' in s:
            old = '\"tpu\"'
            s = s.replace(old, '\"cpu\"')
        return s
