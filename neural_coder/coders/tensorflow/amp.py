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


from ...utils.line_operation import get_line_lhs


class TensorFlowKerasAMP(object):
    def __init__(self, file) -> None:
        self.file = file
        self.result = []
        self.keras_edited_flag = False

    def transform(self):
        # import pdb
        # pdb.set_trace()
        lines = self.file.split('\n')
        for line in lines:
            if self.is_modify(line):
                if '.ConfigProto()' in line:  # TF AMP
                    config_name = get_line_lhs(line)
                    new_line_1 = "from tensorflow.core.protobuf import rewriter_config_pb2"
                    new_line_2 = config_name + \
                        ".graph_options.rewrite_options.auto_mixed_precision_mkl = \
                            rewriter_config_pb2.RewriterConfig.ON"
                    self.result.append(line)
                    self.result.append(new_line_1)
                    self.result.append(new_line_2)
                elif 'keras' in line and 'import' in line:  # Keras AMP
                    if not self.keras_edited_flag:
                        new_line_1 = "from tensorflow.keras.mixed_precision import experimental as mixed_precision"
                        new_line_2 = "policy = mixed_precision.Policy('mixed_bfloat16')"
                        new_line_3 = "mixed_precision.set_policy(policy)"
                        self.result.append(line)
                        self.result.append(new_line_1)
                        self.result.append(new_line_2)
                        self.result.append(new_line_3)
                        self.keras_edited_flag = True
                    else:
                        self.result.append(line)
            else:

                self.result.append(line)
        for index, line in enumerate(self.result):
            if index != len(self.result)-1:
                self.result[index] += '\n'
        return ''.join(self.result)

    def is_modify(self, s):
        if '.ConfigProto()' in s or ('keras' in s and 'import' in s):
            return True
        else:
            return False
