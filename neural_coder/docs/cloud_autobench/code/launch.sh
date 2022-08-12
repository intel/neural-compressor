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

set -x

export PATH=/home/anaconda3/bin:$PATH
source activate test
echo "[INFO] Start running auto benchmark..."
python -c "from neural_coder import superreport; superreport(code='resnet50.py')"
# Note: you need to uncomment superreport in neural_coder/interface.py and neural_coder/__init__.py to use this API.
