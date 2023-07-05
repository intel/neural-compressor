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
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import checkpoint
import tokenization
import optim
import trainer
import data
import models
from utils import set_seeds, get_device
import os

class Config(NamedTuple):
    """ Config for classification """
    mode: str = "train"
    seed: int = 12345
    cfg_data: str = "config/agnews_data.json"
    cfg_model: str = "config/bert_base.json"
    cfg_optim: str = "config/finetune/agnews/optim.json"
    model_file: str = ""
    pretrain_file: str = "models/uncased_L-12_H-768_A-12/bert_model.ckpt"
    save_dir: str = "./saved_results"
    comments: str = [] # for comments in json file


def main(config='config/blendcnn/mrpc/eval.json', args=None):
    cfg = Config(**json.load(open(config, "r")))

    cfg_data = data.Config(**json.load(open(cfg.cfg_data, "r")))
    cfg_model = models.Config(**json.load(open(cfg.cfg_model, "r")))
    cfg_optim = trainer.Config(**json.load(open(cfg.cfg_optim, "r")))

    set_seeds(cfg.seed)

    TaskDataset = data.get_class(cfg_data.task) # task dataset class according to the task
    tokenizer = tokenization.FullTokenizer(vocab_file=cfg_data.vocab_file, do_lower_case=True)
    dataset = TaskDataset(args.dataset_location, pipelines=[
        data.RemoveSymbols('\\'),
        data.Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
        data.AddSpecialTokensWithTruncation(cfg_data.max_len),
        data.TokenIndexing(tokenizer.convert_tokens_to_ids,
                           TaskDataset.labels,
                           cfg_data.max_len)
    ], n_data=None)
    dataset = TensorDataset(*dataset.get_tensors()) # To Tensors
    data_iter = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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


    class Bert_DataLoader(object):
        def __init__(self, loader=None, model_type=None, device='cpu', batch_size=1):
            self.loader = loader
            self.model_type = model_type
            self.device = device
            self.batch_size = batch_size

        def __iter__(self):
            for batch in self.loader:
                batch = tuple(t.to(self.device) for t in batch)
                outputs = {'output_all': (batch[0], batch[1], batch[2]),
                           'labels': batch[3]}

                yield outputs['output_all'], outputs['labels']


    eval_dataloader = Bert_DataLoader(loader=data_iter, batch_size=args.batch_size)

    def eval_func(model):
        results = [] # prediction results
        total_samples = 0
        total_time = 0
        index = 0
        for batch, label in eval_dataloader:
            index += 1
            tic = time.time()
            with torch.no_grad(): # evaluation without gradient calculation
                accuracy, result = evaluate(model, (*batch, label)) 
            results.append(result)
            if index > args.warmup:
                total_samples += batch[0].size()[0]
                total_time += time.time() - tic
        total_accuracy = torch.cat(results).mean().item()
        throughput = total_samples / total_time
        print('Latency: %.3f ms' % (1 / throughput * 1000))
        print('Throughput: %.3f samples/sec' % (throughput))
        print('Accuracy: %.3f ' % (total_accuracy))
        return total_accuracy

    if cfg.mode == "train":
        train_loop.train(get_loss, cfg.model_file, None) # not use pretrain_file
        print("Training has been done properly.")

    elif cfg.mode == "eval":

        if args.tune:
            from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig
            from neural_compressor import quantization
            accuracy_criterion = AccuracyCriterion(higher_is_better=True, tolerable_loss=0.01)
            conf = PostTrainingQuantConfig(accuracy_criterion=accuracy_criterion)
            q_model = quantization.fit(model,
                                        conf,
                                        calib_dataloader=eval_dataloader,
                                        eval_func=eval_func)
            q_model.save(cfg.save_dir)
            exit(0)

        # Benchmark or accuracy
        if args.performance or args.accuracy:
            if args.int8:
                from neural_compressor.utils.pytorch import load
                new_model = load(
                        os.path.abspath(os.path.expanduser(args.output_dir)), model)
            else:
                new_model = model

            if args.performance:
                from neural_compressor.config import BenchmarkConfig
                from neural_compressor import benchmark
                b_conf = BenchmarkConfig(warmup=5, iteration=100, cores_per_instance=4, num_of_instance=1)
                benchmark.fit(new_model, b_conf, b_dataloader=eval_dataloader)
            else:
                eval_func(new_model)

            exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default='config/blendcnn/mrpc/eval.json', 
                        type=str, metavar='PATH', help='path to model config file')
    parser.add_argument("--dataset_location", default='./MRPC/dev.tsv', 
                        type=str, metavar='PATH', help='path to dataset')
    parser.add_argument("--input_model", default='exp/bert/blendcnn/mrpc/model_final.pt', 
                        type=str, metavar='PATH', help='path of model')
    parser.add_argument("--output_model", default='', 
                        type=str, metavar='PATH', help='path to put tuned model')
    parser.add_argument("--tune", action='store_true',
                        help="run Intel® Neural Compressor to tune int8 acc.")
    parser.add_argument("--warmup", type=int, default=10,
                        help="warmup for performance")
    parser.add_argument("--iter", default=0, type=int,
                        help='For accuracy measurement only.')
    parser.add_argument("--batch_size", default=32, type=int,
                        help='dataset batch size.')
    parser.add_argument('--performance', dest='performance', action='store_true',
                        help='run benchmark')
    parser.add_argument('-r', "--accuracy", dest='accuracy', action='store_true',
                        help='For accuracy measurement only.')
    parser.add_argument("--tuned_checkpoint", default='./saved_results', type=str, metavar='PATH',
                        help='path to checkpoint tuned by  (default: ./saved_results)')
    parser.add_argument('--int8', dest='int8', action='store_true',
                        help='run Intel® Neural Compressor model benchmark')
    args = parser.parse_args()
    main(args.model_config, args)
