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

import re
def determine_domain(path) -> str:
    codes = open(path, 'r').read()
    if ('import torchvision.models' in codes or 'from torchvision.models' in codes) and 'val_loader' in codes:
        return 'torchvision'
    elif re.search(r'from (.*)transformers import', codes) and re.search(r'(.*)Model(.*)', codes):
        if 'Trainer' in codes or 'trainer' in codes:
            return 'transformers_trainer'
        else:
            return 'transformers_no_trainer'
    elif 'onnx.load(' in codes:
        return 'onnx'
    elif 'keras.Sequential' in codes:
        return 'keras_script'
    elif 'from tensorflow import' in codes or 'import tensorflow' in codes:
        return 'tensorflow_keras_model'
    else:
        return 'random model'
