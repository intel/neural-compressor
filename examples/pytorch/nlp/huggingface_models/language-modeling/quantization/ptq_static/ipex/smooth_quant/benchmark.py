import os.path
import time

import transformers
import torch
from tqdm import tqdm
import sys
import argparse
import numpy as np

sys.path.insert(0, './')

parser = argparse.ArgumentParser()
parser.add_argument('--int8', action='store_true', help="eval fp32 model or int8 model")
parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-560m')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--warmup', type=int, default=5)
args = parser.parse_args()

from torch.nn.functional import pad

class Evaluator:
    def __init__(self, dataset, tokenizer, device, batch_size=args.batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.dataloader = INCDataloader(dataset, tokenizer, batch_size, device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        index = 1
        latency_list = []
        for input_ids, label, label_indices in tqdm(self.dataloader):
            start = time.time()
            outputs = model(input_ids)
            end = time.time()
            latency_list.append(end - start)
            last_token_logits = outputs[0][:, label_indices, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            index += 1
        acc = hit / total
        latency = np.array(latency_list[args.warmup:]).mean() / self.batch_size
        print("Accuracy: {:.3f}".format(acc))
        print("Batch size = {}".format(self.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
        return acc


class INCDataloader:
    def __init__(self, dataset, tokenizer, batch_size=1, device='cpu', for_calib=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.for_calib = for_calib
        import math
        self.length = math.ceil(len(dataset) / self.batch_size)
        self.pad_len = 196

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    def pad_input(self, input):
        input_id = input['input_ids'].unsqueeze(0)
        label = input_id[:, -1].to(self.device)
        pad_len = self.pad_len - input_id.shape[1]
        label_index = -2 - pad_len
        input_id = pad(input_id, (0, pad_len), value=1)

        return (input_id, label, label_index)

    def __iter__(self):
        input_ids = None
        labels = None
        label_indices = None
        for idx, batch in enumerate(self.dataset):
            input_id, label, label_index = self.pad_input(batch)

            if input_ids is None:
                input_ids = input_id
                labels = label
                label_indices = [label_index]
            else:
                input_ids = torch.cat((input_ids, input_id), 0)
                labels = torch.cat((labels, label), 0)
                label_indices.append(label_index)

            if (idx + 1) % self.batch_size == 0:
                if self.for_calib:
                    if input_ids.shape[1] > 512:
                        input_ids = input_ids[:, 512]
                    yield input_ids
                else:
                    yield (input_ids, labels, label_indices)
                input_ids = None
                labels = None
                label_indices = None
        if (idx + 1) % self.batch_size != 0:
            if self.for_calib:
                if input_ids.shape[1] > 512:
                    input_ids = input_ids[:, 512]
                yield input_ids
            else:
                yield (input_ids, labels, label_indices)

    def __len__(self):
        return self.length


from datasets import load_dataset

model_name = args.model_name_or_path
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
eval_dataset = load_dataset('lambada', split='validation')

evaluator = Evaluator(eval_dataset, tokenizer, 'cpu')

from neural_compressor.benchmark import fit
from neural_compressor.config import BenchmarkConfig

if args.int8:
    print("benchmarking int8 model")
    from neural_compressor.utils.pytorch import load

    int8_folder = model_name.split('/')[-1]
    if not os.path.exists(int8_folder):
        print(f"could not find int8 folder {int8_folder} ")
        exit()
    model = load(int8_folder)
else:
    print("benchmarking fp32 model")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                              torchscript=True  ##FIXME
                                                              )
    model.eval()
conf = BenchmarkConfig(backend='ipex')
fit(model, conf, b_func=evaluator.evaluate)
