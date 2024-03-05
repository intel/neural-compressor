import os
os.environ["EXPERIMENTAL_WEIGHT_SHARING"] = "False"

### USE_GAUDI2_SCALE requires PT_USE_FP8_AMAX for torch.mm/bmm, or got failure
# os.environ["USE_GAUDI2_SCALE"] = "True"
# os.environ["PT_USE_FP8_AMAX"] = "True"

### graphs will dump to .graph_dumps folder
# os.environ["GRAPH_VISUALIZATION"] = "True"
# import shutil
# shutil.rmtree(".graph_dumps", ignore_errors=True)

import argparse
import time
import json
import re
import torch
import habana_frameworks.torch.hpex
import torch.nn.functional as F
import deepspeed
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import habana_frameworks.torch.core as htcore
from accelerate import init_empty_weights
from utils import show_msg, eval_func


torch.set_grad_enabled(False)
htcore.hpu_set_env()
torch.device('hpu')




parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="facebook/opt-125m"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--to_graph", action="store_true")
parser.add_argument("--approach", type=str, default=None,
                    help="Select from ['dynamic', 'static' 'cast']")
parser.add_argument("--precision", type=str, default='fp32',
                    help="Select from ['fp8_e4m3', 'fp8_e5m2', 'bf16', 'fp16', 'fp32'], \
                        ['bf16', 'fp16'] only work with cast approach")
parser.add_argument("--autotune", action="store_true")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--performance", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--skip_fp8_mm", action="store_true")
parser.add_argument("--dump_to_excel", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--load", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=100, type=int,
                    help="calibration iters.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai"], \
                    type=str, choices=["hellaswag", "lambada_openai", "piqa", "winogrande", "copa", 
                                       "rte", "openbookqa", "lambada_standard", "wikitext"],
                    help="tasks list for accuracy validation")
parser.add_argument("--limit", default=None, type=int,
                    help="the sample num of evaluation.")
parser.add_argument("--max_new_tokens", default=100, type=int,
                    help="calibration iters.")
parser.add_argument('--buckets', type=int, nargs='+', \
                    help="Input length buckets to use with static_shapes", default=[256, 512])
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument("--skip_lm_head", action="store_true")
args = parser.parse_args()


world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '-1'))


model_dtype = torch.float32
if re.search("llama", args.model.lower()) or re.search("bloom", args.model.lower()):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if world_size > 1:
        config = AutoConfig.from_pretrained(args.model)
        model_dtype = torch.bfloat16 # RuntimeErrorCastToFp8V2 input must be of float or bfloat16 dtype
        deepspeed.init_distributed(dist_backend="hccl")
        with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
            user_model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)
        import tempfile
        checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
        from optimum.habana.checkpoint_utils import write_checkpoints_json # in optimum-habana
        write_checkpoints_json(
            args.model,
            local_rank,
            checkpoints_json,
            token=None,
        )
    else:
        if args.load:
            config = AutoConfig.from_pretrained(args.model)
            with init_empty_weights():
                user_model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)
        else:
            user_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map='hpu',
                torch_dtype=model_dtype,
            )
elif re.search("chatglm", args.model.lower()):
    if args.load:
        config = AutoConfig.from_pretrained(args.model, torch_dtype=model_dtype)
        with init_empty_weights():
            user_model = AutoModelForCausalLM.from_config(config)
    else:
        from models.modeling_chatglm import ChatGLMForConditionalGeneration
        user_model = ChatGLMForConditionalGeneration.from_pretrained(
            args.model,
            revision=args.revision,
            device_map='hpu',
            torch_dtype=model_dtype,
        )
    # print(user_model.transformer.output_layer.weight.dtype) # always fp16
    user_model.float() # static fp8 need float32 for graph compiler
else:
    if args.load:
        config = AutoConfig.from_pretrained(args.model)
        with init_empty_weights():
            user_model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)
    else:
        user_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            device_map='hpu',
            torch_dtype=model_dtype,
        )


if world_size > 1:
    if re.search("llama", args.model.lower()):
        ds_inference_kwargs = {"dtype": model_dtype}
        ds_inference_kwargs["tensor_parallel"] = {"tp_size": world_size}
        ds_inference_kwargs["enable_cuda_graph"] = False
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        ds_inference_kwargs["injection_policy"] = {LlamaDecoderLayer: ("self_attn.o_proj", "mlp.down_proj")}
        ds_inference_kwargs["checkpoint"] = checkpoints_json.name
        ds_model = deepspeed.init_inference(user_model, **ds_inference_kwargs)
    else:
        ds_model = deepspeed.init_inference(user_model,
                                        mp_size=world_size,
                                        replace_with_kernel_inject=False)
    user_model = ds_model.module


# tokenizer
if re.search("baichuan", args.model.lower()):
    from models.tokenization_baichuan import BaichuanTokenizer
    tokenizer = BaichuanTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code
    )
tokenizer.pad_token = tokenizer.eos_token

user_model.eval()


### dynamic & static quantization ###
if args.approach in ["dynamic", "static"] and not args.load:
    print("device:", next(user_model.parameters()).device)
    from neural_compressor.torch.quantization import (
        quantize, autotune, FP8Config, get_default_fp8_config, TuningConfig, get_default_fp8_config_set
    )
    dtype = args.precision
    if args.approach == "dynamic":
        from neural_compressor.torch.algorithms.habana_fp8 import quantize_dynamic
        user_model = quantize_dynamic(user_model, dtype, inplace=True)
    elif args.approach == "static":
        qconfig = FP8Config(w_dtype=dtype, act_dtype=dtype, approach="static")
        if args.skip_lm_head:
            fp32_config = FP8Config(w_dtype="fp32", act_dtype="fp32")
            qconfig.set_local("lm_head", fp32_config)
        # dataset
        from datasets import load_dataset
        calib_dataset = load_dataset(args.dataset, split="train").select(range(100))
        calib_dataset = calib_dataset.shuffle(seed=42)
        calib_data = []
        for examples in calib_dataset:
            calib_data.append(
                tokenizer(
                    examples["text"], 
                    return_tensors="pt", 
                    max_length=64, 
                    padding="max_length", 
                    truncation=True
                )
            )

        def calib_func(model):
            for i, calib_input in enumerate(calib_data):
                if i >= args.calib_iters:
                    break
                model(
                    input_ids=calib_input["input_ids"].to('hpu'),
                    attention_mask=calib_input["attention_mask"].to('hpu'),
                )

        user_model = quantize(user_model, qconfig, calib_func, inplace=True)
        # saving
        print(user_model)
        if args.save and local_rank in [-1, 0]:
            user_model.save("saved_results")


if args.load:
    from neural_compressor.torch.quantization import load
    user_model = load(user_model, "saved_results")


if args.approach in ["dynamic", "static"] or  args.load:
    # It enables weights constant folding
    from habana_frameworks.torch.core.quantization import _check_params_as_const, _mark_params_as_const
    _mark_params_as_const(user_model)  # can reduce memory allocated and speed up
    _check_params_as_const(user_model)



# If torch.matmul and torch.bmm are not replaced by INC module, 
# Below codes can make torch.matmul and torch.bmm run on fp8 by injection.
if not args.skip_fp8_mm and args.precision in ['fp8_e4m3', 'fp8_e5m2']:
    def replace_torch_mm_bmm():
        from neural_compressor.torch.amp.fp8.functions import fp8_matmul
        torch.matmul = fp8_matmul
        torch.bmm = fp8_matmul

    replace_torch_mm_bmm()


# inference optimization
if args.to_graph:
    import habana_frameworks.torch.hpu.graphs as htgraphs
    user_model = htgraphs.wrap_in_hpu_graph(user_model)


# dump message of HPU after quantization or reloading
show_msg()


### generation, performance and accuracy validation ###
if args.generate:
    input_prompt = "Here is my prompt"
    print("Prompt sentence:", input_prompt)
    generation_config = {
        "min_new_tokens": args.max_new_tokens, "max_new_tokens": args.max_new_tokens,
        # "do_sample": False, "temperature": 0.9, "num_beams": 4,
    }
    input_tokens = tokenizer(input_prompt, return_tensors="pt").to('hpu')
    eval_start = time.perf_counter()
    if args.approach == "cast":
        from neural_compressor.torch.amp import autocast
        if args.precision == "fp8_e4m3":
            dtype = torch.float8_e4m3fn
        elif args.precision == "fp8_e5m2":
            dtype = torch.float8_e5m2
        elif args.precision == "fp16":
            dtype = torch.float16
        elif args.precision == "bf16":
            dtype = torch.bfloat16
        with autocast('hpu', dtype=dtype):
            outputs = user_model.generate(**input_tokens, **generation_config)
    else:
        outputs = user_model.generate(**input_tokens, **generation_config)

    output_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    eval_end = time.perf_counter()
    print("Generated sentence:", output_sentence)
    print("Duration:", eval_end - eval_start)


if args.performance:
    eval_start = time.perf_counter()
    input_prompt = "Intel is a company which"
    input_tokens = torch.ones((1, 128), dtype=torch.long).to('hpu')
    generation_config = {"min_new_tokens": 100, "max_new_tokens": 100}
    outputs = user_model.generate(input_tokens, **generation_config)
    print("Duration of generating 100 tokens :", time.perf_counter() - eval_start)


if args.accuracy:
    eval_func(user_model, tokenizer=tokenizer, args=args)

# dump final message of HPU
show_msg()
