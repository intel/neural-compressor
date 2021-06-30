# -*- coding: utf-8 -*-
# Copyright (c) 2021 Intel Corporation
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
"""Model type getter."""

from lpot.model.model import get_model_type as lpot_get_model_type
from lpot.ux.utils.expiring_dict import ExpiringDict

model_type_cache = ExpiringDict(ttl=600)


def get_model_type(model_path: str) -> str:
    """Get model_type using local cache."""
    try:
        return model_type_cache[model_path]
    except KeyError:
        try:
            model_type = lpot_get_model_type(model_path)
        except Exception:
            model_type = "not a model"
        model_type_cache[model_path] = model_type
        return model_type
