# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import glob
import os.path

import torch

try:
    from torch.utils.cpp_extension import load as load_ext
    from torch.utils.cpp_extension import CUDA_HOME
except ImportError:
    raise ImportError("The cpp layer extensions requires PyTorch 0.4 or higher")


def _load_C_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir = os.path.dirname(this_dir)
    this_dir = os.path.join(this_dir, "csrc")

    main_file = glob.glob(os.path.join(this_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(this_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(this_dir, "cuda", "*.cu"))

    source = main_file + source_cpu

    extra_cflags = []
    if torch.cuda.is_available() and CUDA_HOME is not None:
        source.extend(source_cuda)
        extra_cflags = ["-DWITH_CUDA"]
    source = [os.path.join(this_dir, s) for s in source]
    extra_include_paths = [this_dir]
    return load_ext(
        "torchvision",
        source,
        extra_cflags=extra_cflags,
        extra_include_paths=extra_include_paths,
    )


_C = _load_C_extensions()
