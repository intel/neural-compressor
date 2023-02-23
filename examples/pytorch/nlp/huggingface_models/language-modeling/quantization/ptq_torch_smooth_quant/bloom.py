
import torch
import transformers
from tqdm import tqdm
from datasets import load_dataset
import sys
from transformers import set_seed
sys.path.append('./')

set_seed(42)

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        index = 0
        for batch in tqdm(self.dataset):
            # if index == 100:
            #     break
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            # attention_mask = torch.ones_like(input_ids)
            outputs = model(input_ids)
            last_token_logits = outputs[0][:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            index += 1
            if index % 300 == 0:
                print(hit / total)
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
            yield input_ids


model_name = "bigscience/bloom-560m"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
train_dataset = load_dataset('lambada', split='validation')
train_dataset = train_dataset.shuffle(seed=42)
calib_dataloader = CalibDataloader(train_dataset, tokenizer, 'cpu')

val_dataset = load_dataset('lambada', split='validation')
val_dataset = val_dataset.shuffle(seed=42)
evaluator = Evaluator(val_dataset, tokenizer, 'cpu')

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
)


# tokenize the dataset
def tokenize_function(examples):
    global tokenizer
    example = tokenizer(examples['text'])
    return example


def eval_func(model):
    acc = evaluator.evaluate(model)
    return acc


from neural_compressor import PostTrainingQuantConfig
from neural_compressor import quantization

conf = PostTrainingQuantConfig(backend='ipex',recipes={"smooth_quant":True})
q_model = quantization.fit(model,
                           conf,
                           calib_dataloader=calib_dataloader,
                           eval_func=eval_func)
