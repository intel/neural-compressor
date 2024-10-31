import argparse
import os
import sys

sys.path.append('./')
import time
import re
import torch
from datasets import load_dataset
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

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
parser.add_argument(
    '--seed',
    type=int, default=42, help='Seed for sampling the calibration data.'
)
parser.add_argument("--approach", type=str, default='static',
                    help="Select from ['dynamic', 'static', 'weight-only']")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--ipex", action="store_true", help="Use intel extension for pytorch.")
parser.add_argument("--load", action="store_true", help="Load quantized model.")
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
parser.add_argument("--tasks", default="lambada_openai,hellaswag,winogrande,piqa,wikitext",
                    type=str, help="tasks for accuracy validation")
parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")

args = parser.parse_args()
if args.ipex:
    import intel_extension_for_pytorch as ipex
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
        return self.tokenizer(examples["text"])

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
            if (i + 1) % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print("Accuracy: ", acc)
        print("Latency: ", latency)
        return acc


def get_user_model():
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torchscript=True,  # torchscript will force `return_dict=False` to avoid jit errors
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.peft_model_id is not None:
        from peft import PeftModel
        user_model = PeftModel.from_pretrained(user_model, args.peft_model_id)

    # to channels last
    user_model = user_model.to(memory_format=torch.channels_last)
    user_model.eval()
    return user_model, tokenizer


if args.quantize:
    # dataset
    user_model, tokenizer = get_user_model()
    calib_dataset = load_dataset(args.dataset, split="train")
    # calib_dataset = datasets.load_from_disk('/your/local/dataset/pile-10k/') # use this if trouble with connecting to HF
    calib_dataset = calib_dataset.shuffle(seed=args.seed)
    calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size, pad_max=args.pad_max_length, is_calib=True)
    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=calib_size,
        shuffle=False,
        collate_fn=calib_evaluator.collate_batch,
    )

    
    from neural_compressor.torch.quantization import StaticQuantConfig
    excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
    quant_config = StaticQuantConfig(excluded_precisions=excluded_precisions)
    if re.search("gpt", user_model.config.model_type):
        quant_config.set_local("add", StaticQuantConfig(w_dtype="fp32", act_dtype="fp32"))

    from neural_compressor.torch.algorithms.smooth_quant import move_input_to_device
    from tqdm import tqdm
    def run_fn(model):
        calib_iter = 0
        for batch in tqdm(calib_dataloader, total=args.calib_iters):
            batch = move_input_to_device(batch, device=None)
            if isinstance(batch, tuple) or isinstance(batch, list):
                model(batch[0])
            elif isinstance(batch, dict):
                model(**batch)
            else:
                model(batch)
            
            calib_iter += 1
            if calib_iter >= args.calib_iters:
                break
        return

    from utils import get_example_inputs
    example_inputs = get_example_inputs(user_model, calib_dataloader)

    from neural_compressor.torch.quantization import prepare, convert
    user_model = prepare(model=user_model, quant_config=quant_config, example_inputs=example_inputs)
    run_fn(user_model)
    user_model = convert(user_model)
    user_model.save(args.output_dir)

if args.load:
    if args.int8 or args.int8_bf16_mixed:
        print("load int8 model")
        from neural_compressor.torch.quantization import load
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        config = AutoConfig.from_pretrained(args.model)
        user_model = load(os.path.abspath(os.path.expanduser(args.output_dir)))
        setattr(user_model, "config", config)
    else:
        user_model, tokenizer = get_user_model()


if args.accuracy:
    user_model.eval()
    from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
    eval_args = LMEvalParser(
        model="hf",
        user_model=user_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        device="cpu",
    )
    results = evaluate(eval_args)
    for task_name in args.tasks.split(","):
        if task_name == "wikitext":
            acc = results["results"][task_name]["word_perplexity,none"]
        else:
            acc = results["results"][task_name]["acc,none"]
    print("Accuracy: %.5f" % acc)
    print('Batch size = %d' % args.batch_size)

if args.performance:
    user_model.eval()
    from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
    import time

    samples = args.iters * args.batch_size
    eval_args = LMEvalParser(
        model="hf",
        user_model=user_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        limit=samples,
        device="cpu",
    )
    start = time.time()
    results = evaluate(eval_args)
    end = time.time()
    for task_name in args.tasks.split(","):
        if task_name == "wikitext":
            acc = results["results"][task_name]["word_perplexity,none"]
        else:
            acc = results["results"][task_name]["acc,none"]
    print("Accuracy: %.5f" % acc)
    print('Throughput: %.3f samples/sec' % (samples / (end - start)))
    print('Latency: %.3f ms' % ((end - start) * 1000 / samples))
    print('Batch size = %d' % args.batch_size)
