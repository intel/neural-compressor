# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
#
# Copyright (c) 2020 Intel Corporation
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

""" Fine-tuning on A Classification Task with pretrained Transformer """

import json
from typing import NamedTuple
import fire

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    ### Prepare Dataset and Preprocessing ###

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
    tensors = TensorDataset(*dataset.get_tensors()) # To Tensors
    data_iter = DataLoader(tensors, batch_size=cfg_optim.batch_size, shuffle=False)

    ### Fetch Teacher's output and put it into the dataset ###

    def fetch_logits(model):
        def get_logits(model, batch):
            input_ids, segment_ids, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, input_mask)
            return 0.0, logits

        train_loop = trainer.TrainLoop(cfg_optim, model, data_iter, None, None, get_device())
        results = torch.cat(train_loop.eval(get_logits, cfg.model_file))
        return results


    if cfg.mode == "train":
        print("Fetching teacher's output...")
        teacher = models.Classifier4Transformer(cfg_model, len(TaskDataset.labels))
        teacher.load_state_dict(torch.load(cfg.model_file)) # use trained model
        with torch.no_grad():
            teacher_logits = fetch_logits(teacher)

        tensors = TensorDataset(teacher_logits, *dataset.get_tensors()) # To Tensors
        data_iter = DataLoader(tensors, batch_size=cfg_optim.batch_size, shuffle=False)

    ### Models ###

    model = models.BlendCNN(cfg_model, len(TaskDataset.labels))
    checkpoint.load_embedding(model.embed, cfg.pretrain_file)

    optimizer = optim.optim4GPU(cfg_optim, model)

    train_loop = trainer.TrainLoop(
        cfg_optim, model, data_iter, optimizer, cfg.save_dir, get_device()
    )

    def get_loss(model, batch, global_step): # make sure loss is a scalar tensor
        teacher_logits, input_ids, segment_ids, input_mask, label_id = batch
        T = 1.0
        logits = model(input_ids, segment_ids, input_mask)
        loss = 0.1*nn.CrossEntropyLoss()(logits, label_id)
        loss += 0.9*nn.KLDivLoss()(
            F.log_softmax(logits/T, dim=1),
            F.softmax(teacher_logits/T, dim=1)
        )
        #loss = 0.9*nn.MSELoss()(logits, teacher_logits)
        return loss

    def evaluate(model, batch):
        input_ids, segment_ids, input_mask, label_id = batch
        logits = model(input_ids, segment_ids, input_mask)
        _, label_pred = logits.max(1)
        result = (label_pred == label_id).float() #.cpu().numpy()
        accuracy = result.mean()
        return accuracy, result

    if cfg.mode == "train":
        train_loop.train(get_loss, None, None) # not use pretrain file
        print("Training has been done properly.")

    elif cfg.mode == "eval":
        results = train_loop.eval(evaluate, cfg.model_file)
        total_accuracy = torch.cat(results).mean().item()
        print(f"Accuracy: {total_accuracy}")


if __name__ == '__main__':
    fire.Fire(main)
