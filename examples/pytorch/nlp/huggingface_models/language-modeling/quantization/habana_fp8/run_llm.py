import argparse
import time
import json
import re
import torch
import transformers
import os
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import habana_frameworks.torch.hpex
from habana_frameworks.torch.hpu import memory_stats
import numpy as np
import lm_eval
import lm_eval.tasks
import lm_eval.evaluator
torch.set_grad_enabled(False)


def itrex_bootstrap_stderr(f, xs, iters):
    from lm_eval.metrics import _bootstrap_internal, sample_stddev
    res = []
    chunk_size = min(1000, iters)
    it = _bootstrap_internal(f, chunk_size)
    for i in range(iters // chunk_size):
        bootstrap = it((i, xs))
        res.extend(bootstrap)
    return sample_stddev(res)

# to avoid out-of-memory caused by Popen for large language models.
lm_eval.metrics.bootstrap_stderr = itrex_bootstrap_stderr


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
parser.add_argument("--precision", type=str, default='fp8_e4m3',
                    help="Select from ['fp8_e4m3', 'fp8_e5m2', 'bf16', 'fp16'], \
                        ['bf16', 'fp16'] only work with cast approach")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--performance", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=100, type=int,
                    help="calibration iters.")
parser.add_argument("--tasks", nargs='+', default=["lambada_openai"], type=str, \
                    choices=["winogrande", "copa", "piqa", "rte", "hellaswag", \
                    "openbookqa", "lambada_openai", "lambada_standard", "wikitext"],
                    help="tasks list for accuracy validation")
parser.add_argument("--limit", default=None, type=int,
                    help="the sample num of evaluation.")
parser.add_argument("--max_new_tokens", default=100, type=int,
                    help="calibration iters.")
parser.add_argument('--buckets', type=int, nargs='+', \
                    help="Input length buckets to use with static_shapes", default=[129])
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
args = parser.parse_args()


world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '-1'))

#if local_rank == 0:
#    os.environ["ENABLE_CONSOLE"] = 'True'
#    os.environ["LOG_LEVEL_ALL"] = '0'

# model
if re.search("llama", args.model.lower()) or re.search("bloom", args.model.lower()):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    torch.device('hpu')
    config = AutoConfig.from_pretrained(args.model)
    if world_size > 1:
        model_dtype = torch.bfloat16
        deepspeed.init_distributed(dist_backend="hccl")
        with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
            user_model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)
        import tempfile
        checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
        from utils import write_checkpoints_json
        write_checkpoints_json(
             args.model,
             local_rank,
             checkpoints_json,
             token=None,
        )
    else:
        user_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map='hpu',
        )
elif re.search("chatglm", args.model.lower()):
    from models.modeling_chatglm import ChatGLMForConditionalGeneration
    user_model = ChatGLMForConditionalGeneration.from_pretrained(
        args.model,
        revision=args.revision,
        device_map='hpu',
    )
else:
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        device_map='hpu',
    )

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

user_model.eval()

if args.approach in ["dynamic", "static"]:
    print("device:", next(user_model.parameters()).device)
    from neural_compressor.torch.quantization.config import FP8QConfig, get_default_fp8_qconfig
    if args.precision == "fp8_e4m3":
        dtype = torch.float8_e4m3fn
        qconfig = get_default_fp8_qconfig()
    else:
        dtype = torch.float8_e5m2
        qconfig = FP8QConfig(weight_dtype=torch.float8_e5m2, act_dtype=torch.float8_e5m2, approach="static")


    from neural_compressor.torch.quantization.fp8 import quantize_dynamic
    from neural_compressor.torch.quantization import quantize
    if args.approach == "dynamic":
        user_model = quantize_dynamic(user_model, dtype, inplace=True)
    elif args.approach == "static":
        # dataset
        from datasets import load_dataset
        calib_dataset = load_dataset(args.dataset, split="train").select(range(100))
        calib_dataset = calib_dataset.shuffle(seed=42)
        calib_data = []
        for examples in calib_dataset:
            calib_data.append(
                tokenizer(examples["text"], return_tensors="pt", max_length=128)
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
    print(user_model, flush=True)

if args.to_graph:
    import habana_frameworks.torch.hpu.graphs as htgraphs
    user_model = htgraphs.wrap_in_hpu_graph(user_model)

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

    class HabanaModelAdapter(lm_eval.base.BaseLM):
        def __init__(self, tokenizer, model, args, options):
            super().__init__()
            self.tokenizer = tokenizer
            self.model = model.eval()
            self._batch_size = args.batch_size
            self.buckets = list(sorted(args.buckets))
            self.options = options
            self._device = "hpu"
            torch.set_grad_enabled(False)

        @property
        def eot_token_id(self):
            return self.model.config.eos_token_id

        @property
        def max_length(self):
            return self.buckets[-1]

        @property
        def max_gen_toks(self):
            raise NotImplementedError()

        @property
        def batch_size(self):
            return self._batch_size

        @property
        def device(self):
            # We need to do padding ourselves, otherwise we'll end up with recompilations
            # Returning 'cpu' to keep tensors on CPU in lm_eval code
            return 'cpu' # 'hpu'

        def tok_encode(self, string):
            if re.search("chatglm3", args.model.lower()) or re.search("llama", args.model.lower()) :
                string = string.lstrip()
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

        def _model_generate(self, context, max_length, eos_token_id):
            raise NotImplementedError()

        def find_bucket(self, length):
            return [b for b in self.buckets if b >= length][0]

        def _model_call(self, inps):
            #print(inps.shape)
            seq_length = inps.shape[-1]
            bucket_length = self.find_bucket(seq_length)
            padding_length = bucket_length - seq_length
            if True:
                import torch.nn.functional as F
                inps = F.pad(inps, (0, padding_length), value=self.model.config.pad_token_id)

            logits = self.model(inps.to(self._device))['logits']
            if True and padding_length > 0:
                logits = logits[:, :-padding_length, :]
            logits = logits.to(torch.float32)
            return logits

    lm_tasks = lm_eval.tasks.get_task_dict(args.tasks)
    options = None
    lm = HabanaModelAdapter(tokenizer, user_model, args, options)

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
            results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit)
    else:
        results = lm_eval.evaluator.evaluate(lm, lm_tasks, limit=args.limit)
    print(lm_eval.evaluator.make_table(results))
    eval_end = time.perf_counter()
    print("Duration:", eval_end - eval_start)
    results['args'] = vars(args)
    results['duration'] = eval_end - eval_start


    dumped = json.dumps(results, indent=2)
    for task_name in args.tasks:
        if task_name == "wikitext":
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]), flush=True)
        else:
            print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]), flush=True)

# show memory usage
mem_stats = memory_stats()
mem_dict = {
    "memory_allocated (GB)": np.round(mem_stats["InUse"] / 1024**3, 2),
    "max_memory_allocated (GB)": np.round(mem_stats["MaxInUse"] / 1024**3, 2),
    "total_memory_available (GB)": np.round(mem_stats["Limit"] / 1024**3, 2),
}
for k, v in mem_dict.items():
    print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))