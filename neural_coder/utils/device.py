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

import os
import subprocess
import torch

from .. import globals


def detect_device():
    if torch.cuda.is_available():
        globals.device = "cuda"
        # torch.cuda.device_count()
        # torch.cuda.get_device_name(0)
        # torch.cuda.get_device_properties(0)
    elif check_has('clinfo | grep "Intel(R) Graphics"'):
        globals.device = "intel_gpu"
    else:
        if check_has('lscpu | grep "amx"'):
            globals.device = "cpu_with_amx"
        else:
            globals.device = "cpu_without_amx"


def check_has(s):
    cmd = s
    try:
        sp = subprocess.Popen(
            cmd,
            env=os.environ,
            shell=True,  # nosec
            stdout=subprocess.PIPE
        )  # nosec
        sp.wait()
        sp, _ = sp.communicate()
        has = bool(len(sp.decode()) > 0)  # 0: no, >0: yes
    except:
        has = False
        print('Checking failed.')
    return has


def detect_code_device_compatibility(code_path):
    # handle github py url
    if "github.com" in code_path and ".py" in code_path:
        import requests
        code_path = code_path.replace("github.com", "raw.githubusercontent.com").replace("/blob","")
        r = requests.get(code_path)
        save_py_path = "./neural_coder_workspace/model_analyze_device.py"
        f = open(save_py_path, "wb")
        f.write(r.content)
        code_path = save_py_path

    lines = open(code_path, 'r').read().split('\n')
    for line in lines:
        if "torch.cuda.is_available()" in line:
            globals.list_code_device_compatibility.append("cuda")
            globals.list_code_device_compatibility.append("cpu")
        if "--device" in line:
            if "cpu" in line:
                globals.list_code_device_compatibility.append("cpu")
            if "cuda" in line:
                globals.list_code_device_compatibility.append("cuda")
            if "gpu" in line:
                globals.list_code_device_compatibility.append("gpu")
            if "cpu" not in line and "gpu" not in line and "cuda" not in line:
                globals.list_code_device_compatibility = ["cpu", "cuda", "gpu"]
        if "args.cpu" in line:
            globals.list_code_device_compatibility.append("cpu")
        if "args.cuda" in line:
            globals.list_code_device_compatibility.append("cuda")
        if "args.gpu" in line:
            globals.list_code_device_compatibility.append("gpu")

    globals.list_code_device_compatibility = \
        list(set(globals.list_code_device_compatibility))
