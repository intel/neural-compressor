import argparse
import copy

parser = argparse.ArgumentParser()
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.functional import F

from torch.autograd import Function

from datasets import load_from_disk
from torch.utils.data import DataLoader

import os
from transformers import set_seed
from functools import partial
from torch.amp import autocast
from eval import eval_model
from collections import UserDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["HF_HOME"] = "/models/huggingface"
from signroundv3 import q_dq_weight

if __name__ == '__main__':

    parser.add_argument(
        "--model_name", default="/models/opt-125m"
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

    parser.add_argument("--sym", action='store_true',
                        help=" sym quantization")

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

    parser.add_argument("--lr", default=0.05, type=float,
                        help="step size")

    parser.add_argument("--min_max_lr", default=0.05, type=float,
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

    parser.add_argument("--enable_minmax_tuning", action='store_true',
                        help="enable_tuning_minmax")

    # parser.add_argument("--tasks", default=["lambada_openai", "hellaswag", "winogrande", "piqa"],
    #                     help="lm-eval tasks")

    # parser.add_argument("--tasks", default=["lambada_openai"],
    #                     help="lm-eval tasks")
    #
    # parser.add_argument("--tasks",
    #                     default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
    #                              'coqa', 'truthfulqa_mc', 'openbookqa', 'boolq', 'rte', 'arc_easy', 'arc_challenge',
    #                              'hendrycksTest-*', 'wikitext', 'drop', 'gsm8k'],##all
    #                     help="lm-eval tasks")  # "truthfulqa_gen"

    # parser.add_argument("--tasks",
    #                     default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
    #                              'coqa', 'truthfulqa_mc', 'openbookqa', 'boolq', 'rte', 'arc_easy', 'arc_challenge',
    #                              'hendrycksTest-*', 'wikitext', 'drop', 'gsm8k'],##all
    # parser.add_argument("--tasks",
    #                     default=['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
    #                              "hendrycksTest-*", "wikitext", "truthfulqa_mc", "openbookqa", "boolq", "rte",
    #                              "arc_easy", "arc_challenge"],
    #                     help="lm-eval tasks")  # "truthfulqa_gen"

    parser.add_argument("--tasks", default=["lambada_openai"],
                        help="lm-eval tasks")

    parser.add_argument("--output_dir", default="./tmp_optround", type=str,
                        help="Where to store the final model.")

    args = parser.parse_args()
    set_seed(args.seed)

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, trust_remote_code=True,
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    excel_name = f"{model_name}_{args.num_bits}_{args.group_size}"
    if args.eval_fp16_baseline:
        if not args.low_gpu_mem_usage:
            model = model.to(cuda_device)
        excel_name += "_fp16.xlsx"
        eval_model(output_dir=model_name, model=model, tokenizer=tokenizer, tasks=args.tasks, \
                   eval_bs=args.eval_bs, use_accelerate=args.low_gpu_mem_usage, device=cuda_device,
                   eval_orig_float=True, excel_file=excel_name)
        exit()

    if args.iters <= 0:
        print("eval rtn", flush=True)
        excel_name += "_optround.xlsx"
        q_dq_weight(model, num_bits=args.num_bits, group_size=args.group_size)
        model.half()
        if not args.low_gpu_mem_usage:
            model = model.to(cuda_device)
        eval_model(output_dir=args.output_dir, model=model, tokenizer=tokenizer, tasks=args.tasks, \
                   eval_bs=args.eval_bs, use_accelerate=args.low_gpu_mem_usage, device=cuda_device,
                   excel_file=excel_name)
        exit()

    dataset_name = "NeelNanda/pile-10k"
    # if os.path.exists(dataset_name.split('/')[-1]):
    #     calib_dataset = load_from_disk(dataset_name.split('/')[-1])
    # else:
    #     calib_dataset = load_dataset(dataset_name, split="train")
    #     calib_dataset.save_to_disk(dataset_name.split('/')[-1])
    #
    # calib_dataset = calib_dataset.shuffle(seed=args.seed)
    # calib_dataset = calib_dataset.map(tokenize_function, batched=True)
    # calib_dataset.set_format(type='torch', columns=['input_ids'])
    # calib_dataloader = DataLoader(
    #     calib_dataset,
    #     batch_size=args.eval_bs,
    #     shuffle=False,
    #     collate_fn=collate_batch
    # )
    target_m = None
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and 'ModuleList' in type(m).__name__:
            target_m = (n, m)

    block_names = []
    for n, m in target_m[1].named_children():
        block_names.append(target_m[0] + "." + n)
    seqlen = args.seqlen
    if args.amp and args.device != "cpu":
        model = model.half()
    elif args.amp and args.device == "cpu":
        model = model.to(torch.bfloat16)
    if not args.low_gpu_mem_usage:
        model = model.to(cuda_device)

    import time

    start_time = time.time()
    # save_input_actor = SaveInputs(model, calib_dataloader, seqlen, block_names[0])
    # inputs = save_input_actor.get_inputs(n_samples=args.n_samples)
    # del save_input_actor
    # if args.amp and args.device != "cpu":
    #     model = model.to("cpu").to(torch.float)
    #
    # model = model.to("cpu")
    # torch.cuda.empty_cache()
    # q_dq_weight_round(model, inputs, block_names, num_bits=args.num_bits, group_size=args.group_size,
    #                   n_blocks=args.n_blocks, device=cuda_device)
    from signroundv3 import  OPTRoundQuantizer
    scheme = "asym"
    if args.sym:
        scheme = "sym"

    optq = OPTRoundQuantizer(model, tokenizer, args.num_bits, args.group_size, scheme, bs=args.train_bs,
                             seqlen=seqlen, n_blocks=args.n_blocks, iters=args.iters, lr=args.lr,
                             minmax_lr=args.min_max_lr, use_quant_input=args.use_quant_input)  ##TODO args pass
    optq.quantize()
    end_time = time.time()
    print(end_time - start_time, flush=True)

    torch.cuda.empty_cache()
    model.half()
    model.eval()
    output_dir = args.output_dir + "_" + args.model_name.split('/')[-1] + f"_w{args.num_bits}_g{args.group_size}"

    # model.to(cuda_device)
    # eval_model(model, model_name, tokenizer, tasks=args.tasks, eval_bs=args.eval_bs)
    excel_name = f"{output_dir}_result.xlsx"
    output_dir += "/"
    print(excel_name, flush=True)
    eval_model(output_dir=output_dir, model=model, tokenizer=tokenizer, tasks=args.tasks, \
               eval_bs=args.eval_bs, use_accelerate=args.low_gpu_mem_usage, device=cuda_device, excel_file=excel_name,
               limit=None)
