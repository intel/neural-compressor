import sys
sys.path.append("./")
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import pad

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from datasets import load_dataset

#from neural_compressor.adaptor.torch_utils.weight_only import gptq_quantize
from neural_compressor.adaptor.torch_utils.weight_only import gptq_quantize
from neural_compressor import quantization, PostTrainingQuantConfig
from evaluation import evaluate as lm_evaluate

@torch.no_grad()
def eval_ppl_with_gptq(model, test_dataloader, dev):
    print('Evaluating ...', flush=True)
    # model.eval()
    model.to(dev)

    testenc = test_dataloader.input_ids
    nsamples = test_dataloader.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    test_dataloader = test_dataloader.to(dev)
    nlls = []
    for i in range(nsamples):
        batch = test_dataloader[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_dataloader[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()

class INCDataloader(object):
    def __init__(self, gptq_dataloader):
        self.batch_size = 1
        self.gptq_dataloader = gptq_dataloader
        self.length = len(gptq_dataloader)
        self.batch_size = 1

    def __iter__(self):
        pass

# INC original dataloader example
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
        if args.weight_only_algo in ['AWQ', 'TEQ']:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            example = self.tokenizer(examples["text"], padding="max_length", max_length=self.pad_max)
        else:
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
                # input_ids = input_ids[:self.pad_max] if len(input_ids) > self.pad_max else input_ids
                input_ids = input_ids
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
            if i % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print("Accuracy: ", acc)
        lantecy = latency / len(self.dataset)
        print("Latency: ", latency)
        return acc

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name_or_path', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        '--dataset', type=str, default="NeelNanda/pile-10k",
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument("--weight_only_algo", default="RTN", 
        choices=['RTN', 'AWQ', 'TEQ', 'GPTQ'],
        help="Weight-only parameter."
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening. Please refer to GPTQ paper Part IV for more information'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--group_size', type=int, default=-1,
        help='Groupsize to use for quantization, in every group weights shares same quantization parameters; default uses full row.'
    )
    parser.add_argument(
        '--block_size', type=int, default=128,
        help='Block size. sub weight matrix size to run GPTQ.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )
    parser.add_argument(
        '--pad_max_length', type=int, default=2048,
        help='sequence length for GPTQ calibration'
    )
    parser.add_argument(
        '--calib_size', type=int, default=1,
        help='batch size for calibration'
    )
    parser.add_argument('--act-order', action='store_true', 
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument('--use_max_length', action='store_true', 
        help='Only select data whose length equals or more than model.seqlen, please refer to GPTQ original implementation'
    )
    parser.add_argument('--gpu', action='store_true', help='Whether to use gpu')

    args = parser.parse_args()

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, trust_remote_code=True)
    model.seqlen = args.pad_max_length
    model.eval()

    # dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    calib_dataset = load_dataset(args.dataset, split="train") # default
    # calib_dataset = datasets.load_from_disk('/your/local/pile-10k/') # use this if trouble with connecting to HF
    calib_dataset = calib_dataset.shuffle(seed=args.seed)
    calib_evaluator = Evaluator(calib_dataset, tokenizer, args.calib_size, is_calib=True)
    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=args.calib_size,
        shuffle=False,
        collate_fn=calib_evaluator.collate_batch,
    )

    if args.gpu and torch.cuda.is_available():
        DEV = torch.device('cuda:0')
    else:
        DEV = torch.device('cpu')
    
    model = model.to(DEV)

    if args.sym:
        sym_opt = "sym"
    else:
        sym_opt = "asym"

    # method 1: use general INC API
    # conf = PostTrainingQuantConfig(
    #     approach='weight_only',
    #     op_type_dict={
    #         '.*':{ 	# re.match
    #             "weight": {
    #                 'bits': args.wbits, # 1-8 bits 
    #                 'group_size': args.group_size,  # -1 (per-channel)
    #                 'scheme': sym_opt, 
    #                 'algorithm': 'GPTQ', 
    #             },
    #         },
    #     },
    #     op_name_dict={
    #         '.*lm_head':{ 	# re.match
    #             "weight": {
    #                 'dtype': 'fp32'
    #             },
    #         },
    #     },
    #     recipes={
    #         'gptq_args':{
    #             'percdamp': 0.01, 
    #             'act_order':args.act_order, 
    #             'block_size': args.block_size, 
    #             'nsampeles': args.nsamples,
    #             'use_max_length': args.use_max_length,
    #             'pad_max_length': args.pad_max_length
    #         },
    #     },
    # )
    # q_model = quantization.fit(model, conf, calib_dataloader=calib_dataloader,)

    # method 2: directly use INC built-in function, for some models like falcon, please use this function
    conf = {
        ".*":{
            'wbits': args.wbits, # 1-8 bits 
            'group_size': args.group_size,  # -1 (per-channel)
            'sym': (sym_opt == "sym"),
            'act_order': args.act_order,
        }
    } 
    q_model, gptq_config = gptq_quantize(
        model, 
        weight_config=conf, 
        dataloader=calib_dataloader, 
        nsamples = args.nsamples, 
        use_max_length = args.use_max_length,
        pad_max_length = args.pad_max_length
    )

    results = lm_evaluate(
        model="hf-causal",
        model_args='pretrained='+args.model_name_or_path+',tokenizer='+args.model_name_or_path+',dtype=float32',
        user_model=q_model.to(DEV), tasks=["lambada_openai"],
        device=DEV.type,
        batch_size=4
    )
