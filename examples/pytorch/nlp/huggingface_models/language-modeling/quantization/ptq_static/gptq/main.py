import math
import time

import torch
import torch.nn as nn
import transformers

from neural_compressor.adaptor.torch_utils.gptq import *

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

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
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

    model = get_bloom(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    DEV = "cuda:0"
    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        bloom_sequential(model, dataloader, DEV, args)
        print(time.time() - tick)

    print('Starting ...')

    # import pdb;pdb.set_trace()
    bloom_quantizer = GPTQuantizer(model, dataloader, DEV, args)
    bloom_quantizer() # do quantization

    # datasets = ['wikitext2', 'ptb', 'c4'] 
    # if args.new_eval:
    #   datasets = ['wikitext2', 'ptb-new', 'c4-new']
    # for dataset in datasets: 
    #     dataloader, testloader = get_loaders(
    #         dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
    #     )
    #     print(dataset)
    #     bloom_eval(model, testloader, DEV)
