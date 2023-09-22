import sys
sys.path.append("./")
import math
import time
import numpy as np
import torch
import torch.nn as nn
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

# from cnn_daily_loader_wenhua import CNNDAILYMAIL
from cnn_dm_dataset import CNNDAILYMAIL
from torch.utils.data import DataLoader

from tqdm import tqdm

import evaluate
import nltk
nltk.download("punkt", quiet=False)
metric = evaluate.load("rouge")
import rouge_score

from neural_compressor import quantization, PostTrainingQuantConfig

def get_gptj(model):

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import GPTJForCausalLM, AutoModelForCausalLM
    # model = GPTJForCausalLM.from_pretrained(model) # load the model with fp32 precision
    # model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model = GPTJForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    return model

def postprocess_text(preds, targets):
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets

def benchmark(model, benchmark_dataset, tokenizer, sources, targets, check=False):
    #input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    #torch.cuda.synchronize()
    # for idx in range(len(targets)):
    #     if idx >= 5: break
    #     print(targets[idx])

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.transformer.h):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    generate_kwargs = {
        "early_stopping": True,
        "max_new_tokens": 128,
        "min_new_tokens": 30,
        "num_beams": 4,
        }

    #preds = []
    batch_targets = []
    predictions = []
    ground_truths = []

    with torch.no_grad(), torch.inference_mode():
        times = []
        #for i, (input_ids, labels) in enumerate(benchmark_dataset):# in range(input_ids.numel()):
        for i in tqdm(range(len(sources))): #tqdm(range(len(sources))):
            input_ids, input_lens = benchmark_dataset[i]
            input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)

            print(input_ids)

            #input_lens = input_ids.shape[-1]
            attention_mask = torch.ones((1, input_ids.numel()), device=DEV)

            tick = time.time()
            out = model.generate(input_ids, **generate_kwargs, pad_token_id=tokenizer.pad_token_id, )
            
            sync()
            times.append(time.time() - tick)

            out_tokens = out.cpu().numpy() #[:,input_len:]
            #print("Iter {}".format(i))
            #print("Input len: {}".format(input_lens));
            #print("Output len: {}".format(out_tokens.shape[-1] - input_ids.shape[-1]))
            print("Inference time: {}".format(round(times[-1],3)))

            pred = out_tokens[:, input_lens:]
            pred_batch = tokenizer.batch_decode(pred, skip_special_tokens=True)
            targ_batch = targets[i:i+1]
            preds, targs = postprocess_text(pred_batch, targ_batch)
            print(f"====={targs}=====\n")
            predictions.extend(preds)
            ground_truths.extend(targs)
            
            #cache['past'] = list(out.past_key_values)
            del out
        #sync()
        print('Median:', np.median(times))

        print("Predictions: {}".format(len(predictions)))
        print("References: {}".format(len(ground_truths)))
        result = metric.compute(predictions=predictions, references=ground_truths, use_stemmer=True, use_aggregator=False)
        result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
        prediction_lens = [len(pred) for pred in predictions]
        result["gen_len"] = np.sum(prediction_lens)
        result["gen_num"] = len(predictions)
        print(result)

def gptj_multigpu(model, gpus, gpu_dist):
    #model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    model.transformer.wte = model.transformer.wte.to(gpus[0])
    #if hasattr(model.model, 'norm') and model.model.norm:
    #    model.model.norm = model.model.norm.to(gpus[0])

    if hasattr(model.transformer, 'ln_f') and model.transformer.ln_f:
        model.transformer.ln_f = model.transformer.ln_f.to(gpus[0])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[0])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):

        def __init__(self, module, invalidate_cache):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
            self.invalidate_cache=invalidate_cache

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)

            if cache['mask'] is None or cache['mask'].device != self.dev or self.invalidate_cache:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']

            if cache['position_ids'] is None or cache['position_ids'].device != self.dev or self.invalidate_cache:
                cache['position_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['position_ids'] = cache['position_ids']
            
            tmp = self.module(*inp, **kwargs)
            return tmp

    #layers = model.model.layers
    layers = model.transformer.h
    from math import ceil
    if not gpu_dist:
        pergpu = ceil(len(layers) / len(gpus))
        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(0 if i == 0 or i == len(layers) -1 else gpus[(i-1) // pergpu]), i==0)
    else:
        assert gpu_dist[0] >= 2, "At least two layers must be on GPU 0."
        assigned_gpus = [0] * (gpu_dist[0]-1)
        for i in range(1, len(gpu_dist)):
            assigned_gpus = assigned_gpus + [i] * gpu_dist[i]

        remaining_assignments = len(layers)-len(assigned_gpus) - 1
        if remaining_assignments > 0:
            assigned_gpus = assigned_gpus + [-1] * remaining_assignments

        assigned_gpus = assigned_gpus + [0]

        for i in range(len(layers)):
            layers[i] = MoveModule(layers[i].to(gpus[assigned_gpus[i]]), i==0)

    model.gpus = gpus

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name_or_path', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'pile'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
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
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--calib-data-path', type=str, help="Path to calibration json file")
    parser.add_argument('--val-data-path', type=str, help="Path to validation json file")
    parser.add_argument('--calib-iters', type=int, default=128, help="Number of samples for calibration")
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--use_fp16', action='store_true', help='Whether to convert model to fp16 before using GPTQ.')
    parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU.')
    parser.add_argument(
        '--block_size', type=int, default=128,
        help='Block size. sub weight matrix size to run GPTQ.'
    )
    parser.add_argument(
        '--pad_max_length', type=int, default=2048,
        help='sequence length for GPTQ calibration'
    )
    parser.add_argument('--use_max_length', action='store_true', 
        help='Only select data whose length equals or more than model.seqlen, please refer to GPTQ original implementation'
    )

    # load the gptj model
    args = parser.parse_args()
    # method 1: directly import AutoModelForCausalLM
    model = get_gptj(args.model_name_or_path)
    model.eval()

    if args.use_gpu and torch.cuda.is_available():
        DEV = torch.device('cuda:0')
    else:
        DEV = torch.device('cpu')

    if args.use_fp16:
        model.half()
    model = model.to(DEV)

    # load the dataset
    calib_dataset = CNNDAILYMAIL(args.model_name_or_path, args.calib_data_path, is_calib=True, num_samples=args.calib_iters)
    dataloader=DataLoader(calib_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=calib_dataset.collate_batch
    )

    # # do the quantization
    print('Starting ...')
    if args.sym:
        sym_opt = 'sym'
    else:
        sym_opt = 'asym'

    conf = PostTrainingQuantConfig(
        approach='weight_only',
        op_type_dict={
            '.*':{ 	# re.match
                "weight": {
                    'bits': args.wbits, # 1-8 bits 
                    'group_size': args.group_size,  # -1 (per-channel)
                    'scheme': sym_opt, 
                    'algorithm': 'GPTQ', 
                },
            },
        },
        op_name_dict={
            '.*lm_head':{ 	# re.match
                "weight": {
                    'dtype': 'fp32'
                },
            },
        },
        recipes={
            'gptq_args':{
                'percdamp': 0.01, 
                'act_order':args.act_order,
                'block_size': args.block_size, 
                'nsampeles': args.nsamples,
                'use_max_length': args.use_max_length,
                'pad_max_length': args.pad_max_length
            },
        },
    )

    q_model = quantization.fit(model, conf, calib_dataloader=dataloader,)

    q_model.save("./gptj-gptq-gs128-calib128-calibration-fp16/")
    # q_model.float()
    # q_model.save("./gptj-gptq-gs128-calib128-calibration-fp32/")
    # benchmarking first 100 examples
    # if args.benchmark:
    if True:
        # use half to accerlerate inference
        model.half()
        model = model.to(DEV)

        val_dataset = CNNDAILYMAIL(
            args.model_name_or_path,
            args.val_data_path,
            #is_calib = True,
            num_samples=None
        )

        tokenizer = val_dataset.tokenizer
        sources = val_dataset.sources
        targets = val_dataset.targets
        benchmark_set = DataLoader(val_dataset,
            batch_size=1,
            shuffle=False,
            # collate_fn=val_dataset.collate_batch
            )

        benchmark(model, val_dataset, tokenizer, sources, targets, check=args.check)
    print("Done")
