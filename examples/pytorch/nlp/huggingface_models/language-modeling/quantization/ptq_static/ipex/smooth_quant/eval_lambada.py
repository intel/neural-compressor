import os.path

import transformers
import torch
from tqdm import tqdm
import sys
import argparse

sys.path.insert(0, './')

parser = argparse.ArgumentParser()
parser.add_argument('--int8', action='store_true', default=False, help="eval fp32 model or int8 model")
parser.add_argument('--sq', action='store_true', default=False, help="whether to use smooth quant")
# parser.add_argument('--calib_num', type=int, default=100, help="calibration num for sq")
parser.add_argument('--model_name_or_path', type=str, default='bigscience/bloom-560m')
parser.add_argument('--alpha', default=0.5, help="Set alpha=auto to use alpha tuning.")
parser.add_argument('--log_frequency', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--kl', action='store_true', default=False, help="whether to use kl divergence for calibration")
parser.add_argument('--fallback_add', action='store_true', default=False, help="Whether to add fp32 fallback option" )
args = parser.parse_args()

from torch.nn.functional import pad

class Evaluator:
    def __init__(self, dataset, tokenizer, device, batch_size=args.batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataloader = INCDataloader(dataset, tokenizer, batch_size, device)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        index = 1
        for input_ids, label, label_indices in tqdm(self.dataloader):
            outputs = model(input_ids)
            last_token_logits = outputs[0][torch.arange(len(label_indices)), label_indices, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if index % args.log_frequency == 0:
                print(hit / total, flush=True)
            index += 1
        acc = hit / total
        print(acc, flush=True)
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

model = transformers.AutoModelForCausalLM.from_pretrained(model_name,
                                                          torchscript=True  ##FIXME
                                                          )
model.eval()

if args.int8:
    calib_dataset = load_dataset('lambada', split='train')
    calib_dataset = calib_dataset.shuffle(seed=42)
    calib_dataloader = INCDataloader(calib_dataset, tokenizer, device='cpu', batch_size=1, for_calib=True)


    def eval_func(model):
        acc = evaluator.evaluate(model)
        return acc


    from neural_compressor import PostTrainingQuantConfig
    from neural_compressor import quantization

    recipes = {}
    if args.sq:
        recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': args.alpha}}
    op_type_dict = {}
    if args.kl:
        op_type_dict = {'linear': {'activation': {'algorithm': ['kl']}}}
    if args.fallback_add:
        op_type_dict["add"] = {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}

    conf = PostTrainingQuantConfig(quant_level=1, backend='ipex', excluded_precisions=["bf16"],##use basic tuning
                                   recipes=recipes,
                                   op_type_dict=op_type_dict)

    q_model = quantization.fit(model,
                               conf,
                               calib_dataloader=calib_dataloader,
                               eval_func=eval_func)
    save_model_name = model_name.split("/")[-1]
    q_model.save(f"{save_model_name}")

else:
    acc = evaluator.evaluate(model)
    print(f'Original model accuracy: {acc}')
