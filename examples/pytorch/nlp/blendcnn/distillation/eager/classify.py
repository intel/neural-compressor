# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Fine-tuning on A Classification Task with pretrained Transformer """

import json
from typing import NamedTuple
import fire

import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import checkpoint
import tokenization
import optim
import trainer
import data
import models

from utils import set_seeds, get_device


class Config(NamedTuple):
    """ Config for classification """
    mode: str = "train"
    seed: int = 12345
    cfg_data: str = "config/agnews_data.json"
    cfg_model: str = "config/bert_base.json"
    cfg_optim: str = "config/finetune/agnews/optim.json"
    model_file: str = ""
    pretrain_file: str = "../uncased_L-12_H-768_A-12/bert_model.ckpt"
    save_dir: str = "../exp/bert/finetune/agnews"
    comments: str = [] # for comments in json file


def main(config='config/finetune/agnews/train.json'):

    cfg = Config(**json.load(open(config, "r")))

    cfg_data = data.Config(**json.load(open(cfg.cfg_data, "r")))
    cfg_model = models.Config(**json.load(open(cfg.cfg_model, "r")))
    cfg_optim = trainer.Config(**json.load(open(cfg.cfg_optim, "r")))

    set_seeds(cfg.seed)

    TaskDataset = data.get_class(cfg_data.task) # task dataset class according to the task
    tokenizer = tokenization.FullTokenizer(vocab_file=cfg_data.vocab_file, do_lower_case=True)
    dataset = TaskDataset(cfg_data.data_file[cfg.mode], pipelines=[
        data.RemoveSymbols('\\'),
        data.Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
        data.AddSpecialTokensWithTruncation(cfg_data.max_len),
        data.TokenIndexing(tokenizer.convert_tokens_to_ids,
                           TaskDataset.labels,
                           cfg_data.max_len)
    ], n_data=None)
    dataset = TensorDataset(*dataset.get_tensors()) # To Tensors
    data_iter = DataLoader(dataset, batch_size=cfg_optim.batch_size, shuffle=True)

    model = models.BlendCNN(cfg_model, len(TaskDataset.labels))
    checkpoint.load_embedding(model.embed, cfg.pretrain_file)

    optimizer = optim.optim4GPU(cfg_optim, model)

    train_loop = trainer.TrainLoop(
        cfg_optim, model, data_iter, optimizer, cfg.save_dir, get_device()
    )

    def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        loss = nn.CrossEntropyLoss()(logits, label_id)
        return loss

    def evaluate(model, batch):
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)
        result = (label_pred == label_id).float() #.cpu().numpy()
        accuracy = result.mean()
        return accuracy, result

    if cfg.mode == "train":
        train_loop.train(get_loss, cfg.model_file, None) # not use pretrain_file
        print("Training has been done properly.")

    elif cfg.mode == "eval":
        results = train_loop.eval(evaluate, cfg.model_file)
        total_accuracy = torch.cat(results).mean().item()
        print(f"Accuracy: {total_accuracy}")


if __name__ == '__main__':
    fire.Fire(main)
