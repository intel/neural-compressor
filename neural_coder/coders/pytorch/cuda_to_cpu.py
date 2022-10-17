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

class CudaToCpu(object):
    def __init__(self, file) -> None:
        self.file = file
        self.result = []

    def transform(self):
        lines = self.file.split('\n')
        # determine if jump the whole file (in cases where: args.device, args.cuda etc)
        to_jump = False
        for line in lines:
            if self.is_jump_file(line):
                to_jump = True
                break

        if to_jump: # this file do not need transformation
            for line in lines:
                self.result.append(line)
        else: # this file might need transformation
            for line in lines:
                if self.is_delete(line):
                    indent_level = get_line_indent_level(line)
                    new_line = " " * indent_level + "pass"
                    self.result.append(new_line)
                elif self.is_modify(line):
                    new_line = self.change_to_cpu(line)
                    self.result.append(new_line)
                else:
                    self.result.append(line)
        for index, line in enumerate(self.result):
            if index != len(self.result)-1:
                self.result[index] += '\n'
        return ''.join(self.result)

    def is_jump_file(self, s):
        if "args.device" in s \
            or "args.cpu" in s \
            or "args.gpu" in s \
            or "args.cuda" in s \
            or "torch.cuda.is_available()" in s:
            return True
        else:
            return False

    def is_delete(self, s):
        if 'cuda.' in s and '=' not in s and "if" not in s:
            return True
        else:
            return False

    def is_modify(self, s):
        if '\'cuda\'' in s \
            or '"cuda"' in s \
            or '\'cuda:0\'' in s \
            or '"cuda:0"' in s \
            or 'cuda()' in s:
            return True
        else:
            return False

    def change_to_cpu(self, s):
        if '\'cuda\'' in s or '\'cuda:0\'' in s:
            old = '\'cuda\'' if '\'cuda\'' in s else '\'cuda:0\''
            s = s.replace(old, '\'cpu\'')
        elif '"cuda"' in s or '"cuda:0"' in s:
            old = '"cuda"' if '"cuda"' in s else '"cuda:0"'
            s = s.replace(old, '"cpu"')
        elif 'cuda()' in s:
            old = 'cuda'
            s = s.replace(old, 'cpu')
        return s
