import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset
import sys
from transformers import set_seed
from torch.nn.functional import pad

sys.path.append('./')
from neural_compressor.utils.pytorch import load
set_seed(42)


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            ##example = self.tokenizer(examples['text'], padding='max_length', max_length=512)
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        index = 1
        for batch in tqdm(self.dataset):
            input_ids = batch['input_ids'].unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            outputs = model(input_ids)

            last_token_logits = outputs[0][:, -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if index % 300 == 0:
                print(hit / total)
            index += 1

        acc = hit / total
        return acc


class CalibDataloader():
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = 1

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    def __iter__(self):
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, :512]
            yield input_ids



model_name ="/data2/models/opt-66b"
model_name = "facebook/opt-125m"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# load the dataset
print("Load the datasets.")
dataset = load_dataset('lambada', split='validation')
dataset = dataset.shuffle(seed=42)
calib_dataloader = CalibDataloader(dataset, tokenizer, 'cpu')
print("Dataset loaded.")

evaluator = Evaluator(dataset, tokenizer, 'cpu')
def eval_func(model):
    acc = evaluator.evaluate(model)
    return acc

print("Obtain the int8 model")
tuned_checkpoint = "/data2/models/opt-66b-int8-sq-whc/"
tuned_checkpoint='opt-1.3b-int8-sq'
import os
q_model = load(os.path.expanduser(tuned_checkpoint))
# # model = q_model
# print("int8 model loaded, ready to evaluate.")
acc = eval_func(q_model)
# acc = eval_func(model)
print(f"Evaluation process ended, the acc is {acc}")
