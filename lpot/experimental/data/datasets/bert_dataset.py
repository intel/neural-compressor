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

from .dataset import dataset_registry, Dataset


@dataset_registry(dataset_type="bert", framework="pytorch", dataset_format='')
class PytorchBertDataset(Dataset):
    """Dataset used for model Bert.
       This Dataset is to construct from the Bert TensorDataset and not a full implementation
       from yaml cofig. The original repo link is: https://github.com/huggingface/transformers.
       When you want use this Dataset, you should add it before you initialize your DataLoader.
       (TODO) add end to end support for easy config by yaml by adding the method of
       load examples and process method.

    Args: dataset (list): list of data.
          task (str): the task of the model, support "classifier", "squad".
          model_type (str, default='bert'): model type, support 'distilbert', 'bert',
                                            'xlnet', 'xlm'.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according 
                                                 to specific conditions.
    """

    def __init__(self, dataset, task, model_type='bert', transform=None, filter=None):
        self.dataset = dataset
        assert task in ("classifier", "squad"), "Bert task support only classifier squad"
        self.task = task
        self.transform = transform
        self.model_type = model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.task == 'classifier':
            inputs = {
                'input_ids': sample[0],
                'attention_mask': sample[1],
                'labels': sample[3]}

            if self.model_type != 'distilbert':
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                if self.model_type in ['bert', 'xlnet']:
                    inputs['token_type_ids'] = sample[2] 
            sample = (inputs, inputs['labels'])

        elif self.task == 'squad':
            inputs = {
                'input_ids': sample[0],
                'attention_mask': sample[1], }
            if self.model_type != 'distilbert':
                # XLM, DistilBERT and RoBERTa don't use segment_ids
                inputs['token_type_ids'] = sample[2] if self.model_type in [
                    'bert', 'xlnet'] else None
            if self.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': sample[4], 'p_mask': sample[5]})
            example_indices = sample[3]
            sample = (inputs, example_indices)
        return sample

@dataset_registry(dataset_type="bert", framework="tensorflow", dataset_format='')
class TensorflowBertDataset(Dataset):
    """Configuration for Tensorflow Bert Dataset.

    This dataset supports tfrecord data, please refer to Guide to create tfrecord file first.

    Args: root (str): path of dataset.
          label_file (str): path of label file.
          task (str, default='squad'): task type of model.
          model_type (str, default='bert'): model type, support 'bert'.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according 
                                                 to specific conditions
    """
    def __init__(self, root, label_file, task='squad', 
            model_type='bert', transform=None, filter=None):
        import json
        with open(label_file) as lf:
            label_json = json.load(lf)
            assert label_json['version'] == '1.1', 'only support squad 1.1'
            self.label = label_json['data']
        self.root = root
        self.transform = transform
        self.filter = filter

    def __getitem__(self, index):
        return self.root, self.label

    def __len__(self):
        return 1

