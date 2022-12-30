#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
# ==============================================================================
"""Built-in dataloaders, datasets, transforms, filters for multiple framework backends."""


import neural_compressor.data.datasets
import neural_compressor.data.transforms
from .datasets import Datasets, Dataset, IterableDataset, dataset_registry
from .dataloaders import DATALOADERS, DataLoader
from .transforms import TRANSFORMS, BaseTransform, transform_registry, Postprocess

from .filters import FILTERS, Filter, filter_registry

__all__ = [
    "DataLoader",
    "DATALOADERS",
    "Datasets",
    "Dataset",
    "IterableDataset",
    "dataset_registry",
    "TRANSFORMS",
    "BaseTransform",
    "transform_registry",
    "Postprocess",
    "FILTERS",
    "Filter",
    "filter_registry",]
