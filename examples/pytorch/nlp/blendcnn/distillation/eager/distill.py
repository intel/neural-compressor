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
import os

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
import argparse
import time
from tqdm import tqdm

import torch.autograd.profiler as profiler
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


def main(config='config/distill/mrpc/train.json', args=None):
    cfg = Config(**json.load(open(config, "r")))

    cfg_data = data.Config(**json.load(open(cfg.cfg_data, "r")))
    cfg_model = models.Config(**json.load(open(cfg.cfg_model, "r")))
    cfg_optim = trainer.Config(**json.load(open(cfg.cfg_optim, "r")))

    set_seeds(cfg.seed)

    ### Prepare Dataset and Preprocessing ###
    TaskDataset = data.get_class(cfg_data.task) # task dataset class according to the task

    def get_data_iterator(mode='train'):
        tokenizer = tokenization.FullTokenizer(vocab_file=cfg_data.vocab_file, do_lower_case=True)
        dataset = TaskDataset(cfg_data.data_file[mode], pipelines=[
            data.RemoveSymbols('\\'),
            data.Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
            data.AddSpecialTokensWithTruncation(cfg_data.max_len),
            data.TokenIndexing(tokenizer.convert_tokens_to_ids,
                            TaskDataset.labels,
                            cfg_data.max_len)
        ], n_data=None)
        tensors = TensorDataset(*dataset.get_tensors()) # To Tensors
        data_iter = DataLoader(tensors, batch_size=cfg_optim.batch_size, shuffle=False)
        return data_iter, dataset
    
    data_iter, dataset = train_dataloader, _ = get_data_iterator(mode='train')
    eval_dataloader, _ = get_data_iterator(mode='eval')

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
        teacher = models.Classifier4Transformer(cfg_model, len(TaskDataset.labels))
        teacher.load_state_dict(torch.load(cfg.model_file)) # use trained model
        print("Fetching teacher's output...")
        with torch.no_grad():
            teacher_logits = fetch_logits(teacher)

        tensors = TensorDataset(teacher_logits, *dataset.get_tensors()) # To Tensors
        train_dataloader = data_iter = DataLoader(tensors, batch_size=cfg_optim.batch_size, shuffle=False)
        
        from neural_compressor.experimental.common.criterion import PyTorchKnowledgeDistillationLoss
        criterion = PyTorchKnowledgeDistillationLoss(temperature=args.temperature,
                                                     loss_types=args.loss_types,
                                                     loss_weights=args.loss_weights)

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
        # train_loop.train(get_loss, None, None) # not use pretrain file
        # print("Training has been done properly.")
        
        from neural_compressor.experimental import Distillation, common
        distiller = Distillation(args.distillation_yaml)

        def train_func(model):
            epochs = 30
            iters = 120
            distiller.on_train_begin()
            for nepoch in range(epochs):
                model.train()
                cnt = 0
                loss_sum = 0.
                iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
                for batch in iter_bar:
                    teacher_logits, input_ids, segment_ids, input_mask, target = batch
                    cnt += 1
                    output = model(input_ids, segment_ids, input_mask)
                    loss = criterion(output, target)
                    loss = distiller.on_after_compute_loss(
                        {
                            'input_ids': input_ids,
                            'segment_ids': segment_ids,
                            'input_mask': input_mask
                        },
                        output,
                        loss,
                        teacher_logits
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if cnt >= iters:
                        break
                print('Average Loss: {}'.format(loss_sum / cnt))
                distiller.on_epoch_end()

        def eval_func(model):
            results = [] # prediction results
            total_samples = 0
            total_time = 0
            index = 0
            model.eval()
            iter_bar = tqdm(eval_dataloader, desc='Iter (accuracy=X.XXX)')
            for batch in iter_bar:
                index += 1
                tic = time.time()
                if os.environ.get('BLENDCNN_PROFILING') is not None:
                    with profiler.profile(record_shapes=True) as prof:
                        with torch.no_grad():
                            accuracy, result = evaluate(model, batch)
                else:
                    with torch.no_grad(): # evaluation without gradient calculation
                        accuracy, result = evaluate(model, batch) 
                results.append(result)
                iter_bar.set_description('Iter (accuracy=%.3f)'%accuracy.item())
                if index > args.warmup:
                    total_samples += batch[0].size()[0]
                    total_time += time.time() - tic
            total_accuracy = torch.cat(results).mean().item()
            throughput = total_samples / total_time
            print('Latency: %.3f ms' % (1 / throughput * 1000))
            print('Throughput: %.3f samples/sec' % (throughput))
            print('Accuracy: %f ' % (total_accuracy))

            if os.environ.get('BLENDCNN_PROFILING') is not None:
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            return total_accuracy   

        distiller.student_model = common.Model(model)
        distiller.teacher_model = common.Model(teacher)
        distiller.criterion = criterion
        distiller.train_func = train_func
        distiller.eval_func = eval_func
        model = distiller.fit()
        model.save(args.output_model)
        return

    elif cfg.mode == "eval":
        results = train_loop.eval(evaluate, cfg.model_file)
        total_accuracy = torch.cat(results).mean().item()
        print(f"Accuracy: {total_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default='config/distill/mrpc/train.json', 
                        type=str, metavar='PATH', help='path to model config file')
    parser.add_argument("--output_model", default='./models/blendcnn', 
                        type=str, metavar='PATH', help='path to put tuned model')
    parser.add_argument("--warmup", type=int, default=10,
                        help="warmup for performance")
    parser.add_argument("--distillation_yaml", default='./distillation.yaml', type=str, metavar='PATH',
                        help='path to Intel® Neural Compressor config file')

    parser.add_argument("--temperature", default=1, type=float,
                        help='temperature parameter of distillation')
    parser.add_argument("--loss_types", default=['CE', 'KL'], type=str, nargs='+',
                        help='loss types of distillation, should be a list of length 2, '
                        'first for student targets loss, second for teacher student loss.')
    parser.add_argument("--loss_weights", default=[0.5, 0.5], type=float, nargs='+',
                        help='loss weights of distillation, should be a list of length 2, '
                        'and sum to 1.0, first for student targets loss weight, '
                        'second for teacher student loss weight.')
    args = parser.parse_args()
    main(args.model_config, args)
