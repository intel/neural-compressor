# Copyright 2019 Dong-Hyun Lee, Kakao Brain.

import csv
import itertools
import pprint
from typing import NamedTuple
from enum import Enum

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import tokenization


class Config(NamedTuple):
    """ Config for classification dataset """
    task: str = "agnews"
    vocab_file: str = "../uncased_L-12_H-768_A-12/vocab.txt"
    data_file: dict = {"train": "../agnews/train.csv",
                       "eval": "../agnews/test.csv"}
    max_len: int = 128
    comments: list = [] # for comments in json file


def get_class(task):
    """ Mapping from task string to Dataset Class """
    table = {"mrpc": MRPC, "agnews": AGNews}
    return table[task]



class Pipeline:
    """ Preprocess Pipeline Class : callable """
    def __call__(self, x):
        raise NotImplementedError


class PreprocessedTextDataset(Dataset):
    """ Preprocessed Text Dataset Class """
    def __init__(self, text_file, pipelines=[], n_data=None):
        super().__init__()
        data = []
        # an instance is a list of fields
        for instance in itertools.islice(self.get_instances(text_file), n_data):
            # a bunch of pre-processing for instance
            for pipeline in pipelines:
                assert isinstance(pipeline, Pipeline)
                instance = pipeline(instance)
            data.append(instance)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_instances(self, text_file):
        """ get array of instances from text_file """
        raise NotImplementedError

    def get_tensors(self):
        """ get torch tensors from list of integers """
        return (torch.tensor(x, dtype=torch.long) for x in zip(*self.data))


### Dataset Classes ###

class MRPC(PreprocessedTextDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, *args, **kwags):
        super().__init__(*args, **kwags)

    def get_instances(self, text_file):
        with open(text_file, "r") as f:
            # each line is a list of fields
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for line in itertools.islice(lines, 1, None): # skip header
                # label, text_a, text_b
                yield line[0], line[3], line[4]


class MNLI(PreprocessedTextDataset):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names
    def __init__(self, *args, **kwags):
        super().__init__(*args, **kwags)

    def get_instances(self, text_file):
        with open(text_file, "r") as f:
            # each line is a list of fields
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for line in itertools.islice(lines, 1, None): # skip header
                # label, text_a, text_b
                yield line[-1], line[8], line[9]


class AGNews(PreprocessedTextDataset):
    """ Dataset class for AGNews """
    labels = ("1", "2", "3", "4") # label names
    def __init__(self, *args, **kwags):
        super().__init__(*args, **kwags)

    def get_instances(self, text_file):
        with open(text_file, "r") as f:
            # each line is a list of fields
            for line in csv.reader(f, delimiter=',', quotechar='"'):
                # label, text_a, text_b(N/A)
                yield line[0], line[1]+' '+line[2], None


### Pipeline Classes for preprocessing ###

class RemoveSymbols(Pipeline):
    """ Remove unnecessary symbols """
    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols

    def __call__(self, instance):
        label, text_a, text_b = instance

        for c in self.symbols:
            text_a = text_a.replace(c, ' ')
            text_b = text_b.replace(c, ' ') if text_b else None

        return (label, text_a, text_b)


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) \
                   if text_b else []

        return (label, tokens_a, tokens_b)


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=128):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)


class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=128):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, label_id)


if __name__ == '__main__':
    # Test Case (Optional)

    cfg = Config(task="agnews",
                 vocab_file="../uncased_L-12_H-768_A-12/vocab.txt",
                 data_file={"train": "../agnews/train.csv",
                            "eval": "../agnews/test.csv"},
                 max_len=16)

    #import json
    #cfg = Config(**json.load(open('config/mrpc_data.json')))
    #print(cfg.task)

    TaskDataset = get_class(cfg.task)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=cfg.vocab_file,
        do_lower_case=True)
    pipelines = [RemoveSymbols('\\'),
                 Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
                 AddSpecialTokensWithTruncation(cfg.max_len),
                 TokenIndexing(tokenizer.convert_tokens_to_ids,
                               TaskDataset.labels, cfg.max_len)]

    print(f"\n* Take a look at the dataset according to pipeline (max_len : {cfg.max_len}):\n")
    for i in range(len(pipelines)+1):
        print("Preprocessing Pipeline : ", end="")
        for proc in pipelines[:i]:
            print(type(proc).__name__, end=", ")
        dataset = TaskDataset(cfg.data_file["train"], pipelines[:i], n_data=15)
        print('\n', dataset[0], '\n', dataset[1], '\n')

    print("\nTensors from DataLoader : \n")
    dataset = TensorDataset(*dataset.get_tensors())
    for i, data in enumerate(DataLoader(dataset, batch_size=5, shuffle=True)):
        print(f"<batch {i}>")
        pprint.pprint(data)
        print()


