import sys
sys.path.append("./")
import math
import time
import numpy as np
import torch
import torch.nn as nn
import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

from cnn_dm_dataset import CNNDAILYMAIL
from torch.utils.data import DataLoader

from tqdm import tqdm

import evaluate
import nltk
nltk.download("punkt", quiet=False)
metric = evaluate.load("rouge")
import rouge_score

def get_gptj(model):

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import GPTJForCausalLM, AutoModelForCausalLM
    model = GPTJForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    #model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = 2048
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

    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    # method 1: directly import AutoModelForCausalLM
    model = get_gptj(args.model_name_or_path)
    model.eval()

    # import pdb;pdb.set_trace()
    calib_dataset = CNNDAILYMAIL(args.model_name_or_path, args.calib_data_path, is_calib=True, num_samples=args.calib_iters)
    dataloader=DataLoader(calib_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=calib_dataset.collate_batch
    )

    DEV = torch.device('cuda:0')

    # do the quantization
    print('Starting ...')
    weight_config = {
        'wbits': args.wbits, 
        'group_size': args.group_size, 
        'sym': args.sym,
        'percdamp': args.percdamp,
        'actorder': args.act_order
    }
    print(weight_config)
    from neural_compressor.adaptor.torch_utils.weight_only import gptq_quantize
    quantizers = gptq_quantize(model, weight_config=weight_config, dataloader=dataloader, device = DEV)

    import pdb;pdb.set_trace()

    # benchmarking first 100 examples
    # if args.benchmark:
    if True:
        # use half to accerlerate inference
        model.half()
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            gptj_multigpu(model, gpus, gpu_dist)
        else:
            model = model.to(DEV)

        val_dataset = CNNDAILYMAIL(args.model_name_or_path,
                args.val_data_path,is_calib=False,
                num_samples=None)

        tokenizer = val_dataset.tokenizer
        sources = val_dataset.sources
        targets = val_dataset.targets
        benchmark_set = DataLoader(val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=val_dataset.collate_batch
            )

        benchmark(model, val_dataset, tokenizer, sources, targets, check=args.check)
    print("Done")