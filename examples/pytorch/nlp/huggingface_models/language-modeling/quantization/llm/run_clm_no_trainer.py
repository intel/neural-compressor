import argparse
import os
import sys
sys.path.append('./')
import time
import json
import re
import torch
from datasets import load_dataset
import datasets
from torch.nn.functional import pad
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quantize", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="By default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
# dynamic only now
parser.add_argument("--approach", type=str, default='dynamic', 
                    help="Select from ['dynamic', 'static', 'weight-only']")
parser.add_argument("--weight_dtype", type=str, default="int8", 
                    choices=["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3", "fp6_e3m2", 
                                                "fp6_e2m3", "fp4", "float16", "bfloat12"],
                    help="weight data type")
parser.add_argument("--act_dtype", type=str, default="int8", 
                    choices=["int8", "int4", "int2", "fp8_e5m2", "fp8_e4m3", "fp6_e3m2", 
                                                "fp6_e2m3", "fp4", "float16", "bfloat12"],
                    help="input activation data type")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--woq", action="store_true")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--performance", action="store_true")
parser.add_argument("--iters", default=100, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=512, type=int,
                    help="calibration iters.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai",
    "hellaswag","winogrande","piqa","wikitext"],
    type=str, help="tasks list for accuracy validation")
parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")

args = parser.parse_args()
calib_size = 1

class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196, is_calib=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max
        self.is_calib = is_calib

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        example = self.tokenizer(examples["text"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):

        input_ids_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            if self.is_calib:
                input_ids = input_ids[:self.pad_max] if len(input_ids) > self.pad_max else input_ids
            else:
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)

        return (torch.vstack(input_ids_padded), torch.tensor(last_ind))

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        for i, (input_ids, last_ind) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1

            start = time.time()
            outputs = model(input_ids)
            latency += time.time() - start

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            print(hit / total)
            print("Processed minibatch:", i)
            if i==3:
                break

        acc = hit / total
        print("Accuracy: ", acc)
        print("Latency: ", latency)
        return acc


def get_user_model():
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.peft_model_id is not None:
        from peft import PeftModel
        user_model = PeftModel.from_pretrained(user_model, args.peft_model_id)

    user_model.eval()
    return user_model, tokenizer

if args.quantize:
    # dataset
    user_model, tokenizer = get_user_model()
    calib_dataset = load_dataset(args.dataset, split="train")
    # calib_dataset = datasets.load_from_disk('/your/local/dataset/pile-10k/') # use this if trouble with connecting to HF
    calib_dataset = calib_dataset.shuffle(seed=42)
    calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size, pad_max=args.pad_max_length, is_calib=True)
    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=calib_size,
        shuffle=False,
        collate_fn=calib_evaluator.collate_batch,
    )

    def calib_func(prepared_model):
        for i, calib_input in enumerate(calib_dataloader):
            if i > args.calib_iters:
                break
            prepared_model(calib_input[0])

    eval_dataset = load_dataset('lambada', split='validation')
    evaluator = Evaluator(eval_dataset, tokenizer)
    def eval_func(model):
        acc = evaluator.evaluate(model)
        return acc

    from neural_compressor.torch import MXQuantConfig, quantize
    quant_config = MXQuantConfig(weight_dtype=args.weight_dtype, act_dtype=args.act_dtype, weight_only=args.woq)
    user_model = quantize(model=user_model, quant_config=quant_config)
    # eval_func(user_model)


if args.accuracy:
    user_model.eval()
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    results = evaluate(
        model="hf-causal",
        model_args='pretrained='+args.model+',tokenizer='+args.model+',dtype=float32',
        user_model=user_model,
        batch_size=args.batch_size,
        tasks=args.tasks,
    )
    dumped = json.dumps(results, indent=2)
    if args.save_accuracy_path:
        with open(args.save_accuracy_path, "w") as f:
            f.write(dumped)
    for task_name in args.tasks:
        if task_name == "wikitext":
            acc = results["results"][task_name]["word_perplexity"]
        else:
            acc = results["results"][task_name]["acc"]
    print("Accuracy: %.5f" % acc)
    print('Batch size = %d' % args.batch_size)

if args.performance:
    user_model.eval()
    from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate
    import time
    samples = args.iters * args.batch_size
    start = time.time()
    results = evaluate(
        model="hf-causal",
        model_args='pretrained='+args.model+',tokenizer='+args.model+',dtype=float32',
        user_model=user_model,
        batch_size=args.batch_size,
        tasks=args.tasks,
        limit=samples,
    )
    end = time.time()
    for task_name in args.tasks:
        if task_name == "wikitext":
            acc = results["results"][task_name]["word_perplexity"]
        else:
            acc = results["results"][task_name]["acc"]
    print("Accuracy: %.5f" % acc)
    print('Throughput: %.3f samples/sec' % (samples / (end - start)))
    print('Latency: %.3f ms' % ((end - start)*1000 / samples))
    print('Batch size = %d' % args.batch_size)
