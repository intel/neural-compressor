# Copyright (c) 2024 Intel Corporation
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
"""The constants utils for Tensorflow."""

SPR_BASE_VERSIONS = (
    "2.11.0202242",
    "2.11.0202250",
    "2.11.0202317",
    "2.11.0202323",
    "2.14.0202335",
    "2.14.dev202335",
    "2.15.0202341",
)

TENSORFLOW_DEFAULT_CONFIG = {
    "device": "cpu",
    "backend": "default",
    "approach": "post_training_static_quant",
    "random_seed": 1978,
    "format": "default",
    "use_bf16": True,
}

DEFAULT_SQ_ALPHA_ARGS = {
    "alpha_min": 0.0,
    "alpha_max": 1.0,
    "alpha_step": 0.1,
    "shared_criterion": "mean",
    "do_blockwise": False,
}

UNIFY_OP_TYPE_MAPPING = {
    "Conv2D": "conv2d",
    "Conv3D": "conv3d",
    "DepthwiseConv2dNative": "conv2d",
    "FusedBatchNormV3": "batchnorm",
    "FusedBatchNorm": "batchnorm",
    "_MklFusedInstanceNorm": "instancenorm",
    "MaxPool": "pooling",
    "MaxPool3D": "pooling",
    "AvgPool": "pooling",
    "ConcatV2": "concat",
    "MatMul": "matmul",
    "BatchMatMul": "matmul",
    "BatchMatMulV2": "matmul",
    "Pad": "pad",
    "Conv2DBackpropInput": "deconv2d",
    "Conv3DBackpropInputV2": "deconv3d",
}
