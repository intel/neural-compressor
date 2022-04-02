#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
#


from .direct_q8 import Direct8BitOp, QDQDirect8BitOp


class QResize(Direct8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Resize")

        # if version is less than 11, go to normal quantize.
        if self.quantizer.opset_version < 11:
            super(Direct8BitOp, self).quantize() # pylint: disable=bad-super-call
            return

        # Direct 8bits op
        return super().quantize()


class QDQResize(QDQDirect8BitOp):
    def __init__(self, onnx_quantizer, onnx_node):
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "Resize")

        # if version is less than 11, just keep this node
        if self.quantizer.opset_version < 11:
            return

        # Direct 8bits op
        return super().quantize()
