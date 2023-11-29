import argparse
import copy

import transformers.modeling_utils

parser = argparse.ArgumentParser()
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.functional import F

from torch.autograd import Function

from datasets import load_from_disk
from torch.utils.data import DataLoader

import os
from transformers import set_seed
from functools import partial
from torch.amp import autocast
from neural_compressor.adaptor.torch_utils.optround import OPTRoundQuantizer
from eval import eval_model
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':

    parser.add_argument(
        "--model_name", default="facebook/opt-125m"  ##LaMini-GPT conv1d
    )

    parser.add_argument("--num_bits", default=4, type=int,
                        help="number of  bits")

    parser.add_argument("--group_size", default=128, type=int,
                        help="group size")

    parser.add_argument("--train_bs", default=8, type=int,
                        help="train batch size")

    parser.add_argument("--eval_bs", default=32, type=int,
                        help="eval batch size")

    parser.add_argument("--device", default=0, type=str,
                        help="device gpu int number, or 'cpu' ")

    parser.add_argument("--iters", default=400, type=int,
                        help=" iters")

    parser.add_argument("--dynamic_max_gap", default=0, type=int,
                        help="stop tuning if no best solution found within max_gap steps")

    parser.add_argument("--not_use_mse", action='store_true',
                        help=" whether use mse to get best grad")

    parser.add_argument("--use_quant_input", action='store_true',
                        help="whether to use the output of quantized block to tune the next block")

    parser.add_argument("--sampler", default="rand", type=str,
                        help="sampling type, rand or fix")

    parser.add_argument("--clip_val", default=0.5, type=float,
                        help="clip value")

    parser.add_argument("--lr", default=0.0025, type=float,
                        help="step size")

    parser.add_argument("--min_max_lr", default=0.0025, type=float,
                        help="step size")

    parser.add_argument("--lr_decay_type", default="linear", type=str,
                        help="lr decay type")

    parser.add_argument("--momentum", default=-1, type=float,
                        help="momentum")

    parser.add_argument("--seed", default=42, type=int,
                        help="seed")

    parser.add_argument("--eval_fp16_baseline", action='store_true',
                        help="whether to eval FP16 baseline")

    parser.add_argument("--amp", action='store_true',
                        help="amp")

    parser.add_argument("--not_with_attention", action='store_true',
                        help="tuning with attention_mask input")

    parser.add_argument("--seqlen", default=512, type=int,
                        help="sequence length")

    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="gradient accumulate steps")

    parser.add_argument("--n_blocks", default=1, type=int, help="num of blocks to tune together")

    parser.add_argument("--n_samples", default=512, type=int,
                        help="number of samples")

    parser.add_argument("--lr_wr", default=0.0, type=float,
                        help="lr warmup ratio")

    parser.add_argument("--low_gpu_mem_usage", action='store_true',
                        help="low_gpu_mem_usage")

    parser.add_argument("--trust_remote_code", default=True,
                        help="Transformers parameter: use the external repo")

    parser.add_argument("--enable_minmax_tuning", action='store_true',
                        help="enable_tuning_minmax")

    # parser.add_argument("--tasks", default=["lambada_openai", "hellaswag", "winogrande", "piqa"],
    #                     help="lm-eval tasks")
    parser.add_argument("--scheme", default="asym",
                        help="sym or asym")
    parser.add_argument("--tasks", default=["lambada_openai"],
                        help="lm-eval tasks")
    #
    # parser.add_argument("--tasks", default=["lambada_openai", "hellaswag", "coqa", "winogrande", "piqa", "truthfulqa_mc",\
    #                 "openbookqa", "boolq", "rte", "arc_easy", "arc_challenge", "hendrycksTest-*", "wikitext2", "ptb-new", "c4-new"],
    #                 help="lm-eval tasks") # "truthfulqa_gen"

    parser.add_argument("--output_dir", default="./tmp_optround", type=str,
                        help="Where to store the final model.")

    args = parser.parse_args()
    set_seed(args.seed)
    # args.model_name = "/models/LaMini-GPT-124M"
    model_name = args.model_name

    if model_name[-1] == "/":
        model_name = model_name[:-1]
    print(model_name, flush=True)

    tasks = args.tasks

    if args.device == "cpu":
        device_str = "cpu"
    else:
        device_str = f"cuda:{int(args.device)}"
    cuda_device = torch.device(device_str)
    is_glm = bool(re.search("chatglm", model_name.lower()))
    if is_glm:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=args.trust_remote_code
            ##low_cpu_mem_usage has impact to acc, changed the random seed?
        )
    model = model.eval()
    # align wigh GPTQ to eval ppl
    if "opt" in model_name:
        seqlen = model.config.max_position_embeddings
        model.seqlen = model.config.max_position_embeddings
    else:
        seqlen = 2048
        model.seqlen = seqlen
    seqlen = args.seqlen

    if "llama" in model_name:
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.eval_fp16_baseline:
        eval_model(model, args.model_name, tokenizer, args.tasks, device=device_str)
        exit()

    optq = OPTRoundQuantizer(model, tokenizer, args.num_bits, args.group_size, args.scheme, bs=args.train_bs,
                             seqlen=seqlen, n_blocks=args.n_blocks)
    optq.quantize()
    eval_model(optq.model, args.model_name, tokenizer, args.tasks, device=device_str)
