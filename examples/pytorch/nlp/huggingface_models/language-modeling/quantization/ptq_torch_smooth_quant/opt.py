import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset
import sys
from transformers import set_seed
from torch.nn.functional import pad

sys.path.append('./')

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
                input_ids = input_ids[:, 512]
            yield input_ids


##model_name = "/data2/models/opt-125m/"
model_name = "facebook/opt-125m"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length=512)
dataset = load_dataset('lambada', split='validation')
dataset = dataset.shuffle(seed=42)
calib_dataloader = CalibDataloader(dataset, tokenizer, 'cpu')

dataset_eval = load_dataset('lambada', split='validation')
dataset_eval = dataset_eval.shuffle(seed=42)
evaluator = Evaluator(dataset_eval, tokenizer, 'cpu')

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
)


# tokenize the dataset
def tokenize_function(examples):
    global tokenizer
    example = tokenizer(examples['text'])
    return example


my_dataset = dataset.map(tokenize_function, batched=True)
my_dataset.set_format(type='torch', columns=['input_ids'])


# sq = SmoothQuant(model, my_dataset)
# model = sq.transform()
def eval_func(model):
    acc = evaluator.evaluate(model)
    return acc


from neural_compressor import PostTrainingQuantConfig
from neural_compressor import quantization

conf = PostTrainingQuantConfig(backend='ipex', excluded_precisions=["bf16"])

q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=calib_dataloader,
                           eval_func=eval_func
                           )

q_model.save('opt-1.25b-int8-sq')
