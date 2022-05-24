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

import os
import logging
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from neural_compressor.utils.utility import LazyImport
from .dataset import dataset_registry, Dataset
torch = LazyImport('torch')
transformers = LazyImport('transformers')

logger = logging.getLogger()

@dataset_registry(dataset_type="bert", framework="pytorch", dataset_format='')
class PytorchBertDataset(Dataset):
    """Dataset used for model Bert.
       This Dataset is to construct from the Bert TensorDataset and not a full implementation
       from yaml config. The original repo link is: https://github.com/huggingface/transformers.
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

@dataset_registry(dataset_type="GLUE", framework="onnxrt_qlinearops, \
                    onnxrt_integerops", dataset_format='')
class ONNXRTBertDataset(Dataset):
    """Dataset used for model Bert.

    Args: data_dir (str): The input data dir.
          model_name_or_path (str): Path to pre-trained student model or shortcut name,
                                    selected in the list:
          max_seq_length (int, default=128): The maximum length after tokenization.
                                Sequences longer than this will be truncated,
                                sequences shorter will be padded.
          do_lower_case (bool, default=True): Whether to lowercase the input when tokenizing.
          task (str, default=mrpc): The name of the task to fine-tune.
                                    Choices include mrpc, qqp, qnli, rte,
                                    sts-b, cola, mnli, wnli.
          model_type (str, default='bert'): model type, support 'distilbert', 'bert',
                                            'mobilebert', 'roberta'.
          dynamic_length (bool, default=False): Whether to use fixed sequence length.
          evaluate (bool, default=True): Whether do evaluation or training.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according
                                                 to specific conditions.
    """
    def __init__(self, data_dir, model_name_or_path, max_seq_length=128,\
                do_lower_case=True, task='mrpc', model_type='bert', dynamic_length=False,\
                evaluate=True, transform=None, filter=None):
        task = task.lower()
        model_type = model_type.lower()
        assert task in ['mrpc', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', \
            'mnli', 'wnli'], 'Unsupported task type'
        assert model_type in ['distilbert', 'bert', 'mobilebert', 'roberta'], 'Unsupported \
            model type'
        self.dynamic_length = dynamic_length
        self.model_type = model_type
        self.max_seq_length = max_seq_length
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,
            do_lower_case=do_lower_case)
        self.dataset = load_and_cache_examples(data_dir, model_name_or_path, \
            max_seq_length, task, model_type, tokenizer, evaluate)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def load_and_cache_examples(data_dir, model_name_or_path, max_seq_length, task, \
    model_type, tokenizer, evaluate):
    from torch.utils.data import TensorDataset

    processor = transformers.glue_processors[task]()
    output_mode = transformers.glue_output_modes[task]
    # Load data features from cache or dataset file
    if not os.path.exists("./dataset_cached"):
        os.makedirs("./dataset_cached")
    cached_features_file = os.path.join("./dataset_cached", 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, model_name_or_path.split('/'))).pop(),
        str(max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Load features from cached file {}.".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        logger.info("Create features from dataset file at {}.".format(data_dir))
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(data_dir) if evaluate else \
            processor.get_train_examples(data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                task=task,
                                                label_list=label_list,
                                                max_length=max_seq_length,
                                                output_mode=output_mode,
        )
        logger.info("Save features into cached file {}.".format(cached_features_file))
        torch.save(features, cached_features_file)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, \
        all_seq_lengths, all_labels)
    return dataset

def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=128,
    task=None,
    label_list=None,
    output_mode="classification",
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    processor = transformers.glue_processors[task]()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Use label list {} for task {}.".format(label_list, task))
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        seq_length = len(input_ids)
        padding_length = max_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + \
            ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, \
            "Error with input_ids length {} vs {}".format(
            len(input_ids), max_length)
        assert len(attention_mask) == max_length, \
            "Error with attention_mask length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, \
            "Error with token_type_ids length {} vs {}".format(
            len(token_type_ids), max_length
        )
        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        feats = InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label=label,
            seq_length=seq_length,
        )
        features.append(feats)
    return features

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED,
            ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
        seq_length: (Optional) The length of input sequence before padding.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    seq_length: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

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
