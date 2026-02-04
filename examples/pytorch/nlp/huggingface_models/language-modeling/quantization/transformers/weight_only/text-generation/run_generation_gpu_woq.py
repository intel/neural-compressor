import argparse
import re
import time
import json
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.generation import GenerationConfig
import intel_extension_for_pytorch as ipex
from neural_compressor.transformers import AutoModelForCausalLM, RtnConfig, GPTQConfig
from neural_compressor.transformers.quantization.utils import convert_dtype_str2torch
from neural_compressor.transformers.generation import _greedy_search, _beam_search
from transformers.utils import check_min_version
import contextlib
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="Qwen/Qwen-7B-Chat", const="Qwen/Qwen-7B-Chat"
)
parser.add_argument("--revision", default=None, type=str)
parser.add_argument("--trust_remote_code", action="store_true")
parser.add_argument(
    "--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k"
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--num_beams", default=1, type=int, help="number of beams"
)
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quant_lm_head", action="store_true",  help="whether to quant the lm_head layer in transformers")
parser.add_argument("--for_inference", action="store_true",  help="whether to replace ipex linear for inference")
parser.add_argument("--use_layer_wise",  nargs='?', const=True, default=None, type=lambda x: bool(strtobool(x)), 
                    help="""whether to use layerwise quant. Case-insensitive and
                            true values are 'y', 'yes', 't', 'true', 'on', and '1'; 
                            false values are 'n', 'no', 'f', 'false', 'off', and '0'.""")
# ============Benchmark configs==============
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--benchmark_batch_size", default=1, type=int,
                    help="batch size num.")
parser.add_argument("--do_profiling", action="store_true")
parser.add_argument("--profile_token_latency", action="store_true")
parser.add_argument("--benchmark_iters", default=10, type=int, help="num iter")
parser.add_argument("--num_warmup", default=3, type=int, help="num warmup")
# ============Accuracy configs==============
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--eval_batch_size", default=56, type=int,
                    help="batch size num.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--tasks", default="lambada_openai", type=str, \
                    help="tasks list for accuracy validation")
parser.add_argument("--add_bos_token", action="store_true", help="whether to add bos token for accuracy validation.")
# ============WeightOnlyQuant configs===============
parser.add_argument("--bits", type=int, default=4, choices=[4])
parser.add_argument("--woq", action="store_true")
parser.add_argument("--woq_algo", default="Rtn", choices=['Rtn', 'GPTQ'], 
                    help="Weight-only parameter.")
parser.add_argument("--weight_dtype", type=str, default="int4",
                    choices=[
                        "int4",  # int4 == int4_fullrange
                        "int4_fullrange",
                        ]
                   )
parser.add_argument("--batch_size", default=8, type=int,
                    help="calibration batch size num.")
parser.add_argument("--group_size", type=int, default=128)
parser.add_argument("--scheme", default="sym")
parser.add_argument("--device", default="xpu")
parser.add_argument("--compute_dtype", default="fp16")
parser.add_argument("--load_in_4bit", type=bool, default=False)
parser.add_argument("--load_in_8bit", type=bool, default=False)
# ============GPTQ configs==============
parser.add_argument(
    "--desc_act",
    action="store_true",
    help="Whether to apply the activation order GPTQ heuristic.",
)
parser.add_argument(
    "--damp_percent",
    type=float,
    default=0.01,
    help="Percent of the average Hessian diagonal to use for dampening.",
)
parser.add_argument(
    "--blocksize",
    type=int,
    default=128,
    help="Block size. sub weight matrix size to run GPTQ.",
)
parser.add_argument(
    "--n_samples", type=int, default=512, help="Number of calibration data samples."
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=2048,
    help="Calibration dataset sequence max length, this should align with your model config",
)
parser.add_argument(
    "--static_groups",
    action="store_true",
    help="Use determined group to do quantization",
)
# =======================================
args = parser.parse_args()
torch_dtype = convert_dtype_str2torch(args.compute_dtype)

# transformers version >= 4.32.0 contained the mpt modeling definition.
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py
check_min_version("4.31.0")

# get model config
config = AutoConfig.from_pretrained(
    args.model,
    use_cache=True, # to use kv cache.
    trust_remote_code=args.trust_remote_code,
    revision=args.revision,
)

user_model = None

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
# Ensure pad_token is set for tasks that require it (e.g., truthfulqa)
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

quantization_config = None
if args.woq:
    if args.woq_algo.lower() == "gptq":
        quantization_config = GPTQConfig(
            tokenizer=tokenizer,
            dataset=args.dataset,
            bits=args.bits,
            desc_act=args.desc_act,
            damp_percent=args.damp_percent,
            sym=True if args.scheme == "sym" else False,
            blocksize=args.blocksize,
            n_samples=args.n_samples,
            static_groups=args.static_groups,
            group_size=args.group_size,
            seq_len=args.seq_len,
            compute_dtype=args.compute_dtype,
            scale_dtype=args.compute_dtype,
            weight_dtype=args.weight_dtype,
            batch_size=args.batch_size,
            quant_lm_head=args.quant_lm_head,
            use_layer_wise=args.use_layer_wise,
        )
    elif args.woq_algo.lower() == "rtn":
        quantization_config = RtnConfig(
            compute_dtype=args.compute_dtype,
            weight_dtype=args.weight_dtype,
            group_size=args.group_size,
            scale_dtype=args.compute_dtype,
            quant_lm_head=args.quant_lm_head,
            use_layer_wise=args.use_layer_wise,
        )

# get model
if quantization_config is not None:
    user_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                      device_map=args.device,
                                                      quantization_config=quantization_config,
                                                      trust_remote_code=args.trust_remote_code,
                                                      torch_dtype=torch.float16,
                                                      for_inference=args.for_inference,
                                                      )
elif args.load_in_4bit or args.load_in_8bit:
    user_model = AutoModelForCausalLM.from_pretrained(args.model,
                                                      device_map=args.device,
                                                      load_in_4bit=args.load_in_4bit,
                                                      load_in_8bit=args.load_in_8bit,
                                                      )
if user_model is not None:
    user_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

enable_optimize_transformers = False
opt_gpu_model_type_list = ["llama", "gptj", "mistral", "qwen", "phi3"]

if config.model_type in opt_gpu_model_type_list:
    enable_optimize_transformers = True

if args.benchmark:
    if config.model_type == "qwen":
        prompt = "它完成了，并提交了。你可以在Android和网络上玩美味生存。在网络上玩是有效的，但你必须模拟多次触摸才能移动桌子."
    else:
        prompt = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)

    user_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, device_map=args.device, torch_dtype=torch_dtype) \
            if user_model is None else user_model
    user_model = user_model.to(memory_format=torch.channels_last)
    if quantization_config is None:
        quantization_config = user_model.quantization_config if hasattr(user_model, "quantization_config") else None
    if enable_optimize_transformers:
        print("Optimize with IPEX...")
        user_model = ipex.optimize_transformers(
            user_model.eval(), device=args.device, inplace=True, quantization_config=quantization_config, dtype=torch_dtype)
    else:
        print("Disabled optimization with IPEX...")
    # start
    num_iter = args.benchmark_iters
    num_warmup = args.num_warmup
    prompt = [prompt] * args.benchmark_batch_size
    amp_enabled = True
    amp_dtype = torch_dtype

    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=args.num_beams)
    if args.profile_token_latency:
        ipex.transformers.optimize.convert_function(user_model, "greedy_search", _greedy_search)
        ipex.transformers.optimize.convert_function(user_model, "_greedy_search", _greedy_search)
        if not enable_optimize_transformers:
            ipex.transformers.optimize.convert_function(user_model, "beam_search", _beam_search)
            ipex.transformers.optimize.convert_function(user_model, "_beam_search", _beam_search)
        user_model.config.token_latency = True

    total_time = 0.0
    total_list = []
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        for i in range(num_iter + num_warmup):
            if args.do_profiling:
                context = torch.autograd.profiler_legacy.profile(enabled=args.do_profiling, use_xpu=True, record_shapes=True)
            else:
                context = contextlib.nullcontext()
            with context as prof:
                input_ids = tokenizer(
                    prompt, return_tensors="pt").input_ids.to(args.device)
                tic = time.time()
                output = user_model.generate(
                    input_ids, max_new_tokens=int(args.max_new_tokens), **generate_kwargs
                )
                if args.device == "xpu":
                    torch.xpu.synchronize()
                toc = time.time()
                gen_ids = output[0] if args.profile_token_latency else output
                gen_text = tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True)
            if args.do_profiling and i >= num_warmup and (i == num_warmup or i == num_iter + num_warmup - 1):
                print(f"Save pt for iter {i}")
                torch.save(prof.key_averages().table(
                    sort_by="self_xpu_time_total"), f"./profile_{i}.pt")
                # torch.save(prof.table(sort_by="id", row_limit=-1),
                #            './profile_id.pt')
                # torch.save(prof.key_averages(
                #     group_by_input_shape=True).table(), "./profile_detail.pt")
                prof.export_chrome_trace(f"./trace_{i}.json")
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = [x.shape[0] for x in gen_ids]
            total_new_tokens = [
                o - i if user_model.config.model_type != "t5" else o
                for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            print(gen_text, total_new_tokens, flush=True)
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
            if i >= num_warmup:
                total_time += toc - tic
                if args.profile_token_latency:
                    total_list.append(output[1])

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter - num_warmup)
    print("Inference latency: %.5f sec." % latency)
    throughput = (args.max_new_tokens + input_size) / latency
    print("Average throughput: {} samples/sec".format(throughput))

    if args.profile_token_latency:
        import numpy as np
        from itertools import chain

        first_latency = np.mean([x[0] for x in total_list])
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n)
        print("First token average latency: %.5f sec." % first_latency)
        print("Average 2... latency: %.5f sec." % average_2n_latency)
        print(total_list)


if args.accuracy:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, device_map=args.device, torch_dtype=torch_dtype) \
            if user_model is None else user_model
    if quantization_config is None:
        quantization_config = user_model.quantization_config if hasattr(user_model, "quantization_config") else None
    if enable_optimize_transformers:
        print("Optimize with IPEX...")
        user_model = ipex.optimize_transformers(
            user_model.eval(), device=args.device, inplace=True, quantization_config=quantization_config, dtype=torch_dtype)
    else:
        print("Disabled optimization with IPEX...")
    from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
    args = LMEvalParser(model = "hf", 
                        tokenizer = tokenizer,
                        user_model = user_model,
                        tasks = args.tasks,
                        device = args.device,
                        batch_size = args.eval_batch_size,
                        add_bos_token = args.add_bos_token,)
    results = evaluate(args)
    for task_name in args.tasks.split(","):
        if task_name == "wikitext":
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity,none"]))
        else:
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc,none"]))