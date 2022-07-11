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


class CudaToCpu(object):
    def __init__(self, file) -> None:
        self.file = file
        self.result = []

    def transform(self):
        # import pdb
        # pdb.set_trace()
        lines = self.file.split('\n')
        for line in lines:
            if self.is_delete(line):
                pass
            elif self.is_modify(line):
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

    def is_delete(self, s):
        if 'cuda.' in s and '=' not in s:
            return True
        else:
            return False

    def is_modify(self, s):
        if '\'cuda\'' in s or '\'cuda:0\'' in s or 'cuda()' in s:
            return True
        else:
            return False

    def modify(self, s):
        if '\'cuda\'' in s or '\'cuda:0\'' in s:
            old = '\'cuda\'' if '\'cuda\'' in s else '\'cuda:0\''
            s = s.replace(old, '\'cpu\'')
        elif 'cuda()' in s:
            old = 'cuda'
            s = s.replace(old, 'cpu')
        return s
