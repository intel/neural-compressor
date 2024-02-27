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

from auto_round.calib_dataset import CALIB_DATASETS  # pylint: disable=E0401


def get_dataloader(
    tokenizer, seqlen=2048, seed=42, train_bs=8, dataset_split="train", dataset_name="NeelNanda/pile-10k"
):
    get_dataloader = CALIB_DATASETS.get(dataset_name, CALIB_DATASETS["NeelNanda/pile-10k"])
    dataloader = get_dataloader(
        tokenizer, seqlen=seqlen, seed=seed, bs=train_bs, split=dataset_split, dataset_name=dataset_name
    )
    return dataloader
