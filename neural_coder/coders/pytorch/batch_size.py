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


from ... import globals


class BatchSizeCoder(object):
    def __init__(self, file) -> None:
        self.file = file
        self.result = []

    def transform(self):
        lines = self.file.split('\n')
        for line in lines:
            if self.not_modify(line):
                new_line = self.modify(line)
                self.result.append(new_line)
            else:
                if line == '' and self.result[-1] == '':
                    continue
                self.result.append(line)
        for index, line in enumerate(self.result):
            if index != len(self.result)-1:
                self.result[index] += '\n'
        return ''.join(self.result)

    def not_modify(self, s):
        if 'batch_size' in s and '=' in s:
            return True
        return False

    def modify(self, s):
        idx = s.find('batch_size')
        s_right = s[idx:]
        if ' = ' in s_right:
            index = s.find(' = ')
            s_left = s[:index]
            if 'batch_size' in s_left:
                if ',' in s_left:
                    index1 = s_left.find(',')
                    index2 = s_left.find('batch_size')
                    if index1 > index2:
                        slice1 = s_left[:index1]
                    else:
                        s_left1 = s_left[:index2]
                        s_right = s_left[index2:]
                        index3 = s_left1.rfind(',')
                        if ',' in s_right:
                            index4 = s_right.find(',') + len(s_left1)
                            slice1 = s_left[index3+2:index4]
                        else:
                            slice1 = s_left[index3+2:index]
                    s1 = slice1 + ' = ' + globals.target_batch_size
                    s = s[:] + '\n' + s1
                else:
                    s_right = s[index+3:]
                    s_right = s_right.replace(
                        s_right, globals.target_batch_size)
                    s = s_left + ' = ' + s_right
        elif 'batch_size=' in s:
            idx = s.find('batch_size=')
            s_right = s[idx:]
            idx2 = s_right.find('batch_size')
            if ',' in s_right:
                index2 = s_right.find(',')
                old = s_right[idx2:index2]
                s = s.replace(old, "batch_size=" + globals.target_batch_size)
            elif ')' in s_right:
                index2 = s_right.find(')')
                old = s_right[idx2:index2]
                s = s.replace(old, "batch_size=" + globals.target_batch_size)
            else:
                old = s_right[idx2:]
                s = s.replace(old, "batch_size=" + globals.target_batch_size)
        return s
