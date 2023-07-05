import sys
sys.path.append("./")
import math
import time

import torch
import torch.nn as nn
import transformers

from neural_compressor.adaptor.torch_utils import gptq
from transformers import AutoModelForCausalLM
from evaluation import evaluate as lm_evaluate

def get_bloom(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'pile'],
        help='Where to extract calibration data from.'
    )
    #----------generalized into one code and control gptq behavior with this argument.---------
    parser.add_argument(
        '--arch', type=str, choices=['bloom', 'opt', 'llama'],
        help='Supported different llm architectures within one general code set'
    )
    #------------------------------------------------------------------------------------------
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
        '--groupsize', type=int, default=-1,
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


    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model, low_cpu_mem_usage=True)
    model.seqlen = 2048

    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    DEV = torch.device('cuda:0')

    print('Starting ...')

    # import pdb;pdb.set_trace()
    bloom_quantizer = gptq.GPTQuantizer(model, dataloader, DEV, args)
    bloom_quantizer.execute_quantization() # do quantization

    results = lm_evaluate(
        model="hf-causal",
        model_args=f'pretrained="{args.model}",tokenizer="{args.model}",dtype=float32',
        user_model=model.to(DEV), tasks=["lambada_openai"],
        device=DEV.type,
        batch_size=4
    )
    # post quantization 