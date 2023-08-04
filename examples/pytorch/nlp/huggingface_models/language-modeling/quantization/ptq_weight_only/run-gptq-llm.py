import sys
sys.path.append("./")
import math
import time

import torch
import torch.nn as nn
import transformers

#from neural_compressor.adaptor.torch_utils.weight_only import gptq_quantize
from neural_compressor.adaptor.torch_utils.weight_only import gptq_quantize
from neural_compressor import quantization, PostTrainingQuantConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
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

if __name__ == '__main__':
    import argparse
    from datautils import *

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
    parser.add_argument('--gpu', action='store_true', help='Whether to use gpu')

    args = parser.parse_args()

    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    # import pdb;pdb.set_trace()
    # method 1: directly import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, trust_remote_code=True)
    model.seqlen = 2048
    # method 2:
    # model = get_llama(args.model)

    model.eval()
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model_name_or_path, seqlen=model.seqlen
    )
    dataloader = INCDataloader(dataloader)

    if args.gpu and torch.cuda.is_available():
        DEV = torch.device('cuda:0')
    else:
        DEV = torch.device('cpu')
    
    model = model.to(DEV)

    print('Starting ...')
    # import pdb;pdb.set_trace()
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
    #         'gptq_args':{'percdamp': 0.01, 'actorder':args.act_order},
    #     },
    # )
    # q_model = quantization.fit(model, conf, calib_dataloader=dataloader,)

    # method 2: directly use build-in function, for some models like falcon, please use this function
    conf = {
        ".*":{
            'wbits': args.wbits, # 1-8 bits 
            'group_size': args.group_size,  # -1 (per-channel)
            'sym': (sym_opt == "sym"),
            'actorder': args.act_order,
        }
    } 
    q_model, gptq_config = gptq_quantize(model, weight_config=conf, dataloader=dataloader)

    results = lm_evaluate(
        model="hf-causal",
        model_args=f'pretrained="{args.model_name_or_path}",tokenizer="{args.model_name_or_path}",dtype=float32',
        user_model=q_model.to(DEV), tasks=["lambada_openai"],
        device=DEV.type,
        batch_size=4
    )

    # datasets = ['wikitext2']

    # for dataset in datasets:
    #     dataloader, testloader = get_loaders(
    #         dataset, seed=0, model=args.model_name_or_path, seqlen=model.seqlen
    #     )
    #     print(dataset, flush=True)
    #     ppl = eval_ppl_with_gptq(model, testloader, device)
    #     results.update({dataset: ppl})
