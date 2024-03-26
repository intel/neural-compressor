import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


world_size = int(os.getenv('WORLD_SIZE', '1'))
local_rank = int(os.getenv('LOCAL_RANK', '-1'))


def init_model(args):
    import deepspeed
    model_dtype = torch.float32
    if re.search("llama", args.model.lower()) or re.search("bloom", args.model.lower()):
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
            user_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map='hpu',
                torch_dtype=model_dtype,
            )
    elif re.search("chatglm", args.model.lower()):
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
        user_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            device_map='hpu',
            torch_dtype=model_dtype,
        )
    # load weight for multi-cards
    if world_size > 1:
        if re.search("llama", args.model.lower()) or re.search("bloom", args.model.lower()):
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
    return user_model


def init_empty_model(model_name):
    from accelerate import init_empty_weights
    model_dtype = torch.float32
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)
    return model


def init_tokenizer(args):
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
    return tokenizer


def show_msg():
    import numpy as np
    import glob
    from habana_frameworks.torch.hpu import memory_stats
    print("Number of HPU graphs:", len(glob.glob(".graph_dumps/*PreGraph*")))
    mem_stats = memory_stats()
    mem_dict = {
        "memory_allocated (GB)": np.round(mem_stats["InUse"] / 1024**3, 2),
        "max_memory_allocated (GB)": np.round(mem_stats["MaxInUse"] / 1024**3, 2),
        "total_memory_available (GB)": np.round(mem_stats["Limit"] / 1024**3, 2),
    }
    for k, v in mem_dict.items():
        print("{:35} = {} GB".format(k[:-5].replace("_", " ").capitalize(), v))


def itrex_bootstrap_stderr(f, xs, iters):
    from lm_eval.metrics import _bootstrap_internal, sample_stddev
    res = []
    chunk_size = min(1000, iters)
    it = _bootstrap_internal(f, chunk_size)
    for i in range(iters // chunk_size):
        bootstrap = it((i, xs))
        res.extend(bootstrap)
    return sample_stddev(res)


def save_to_excel(dict):
    import pandas as pd
    df_new = pd.DataFrame(dict)
    try:
        df_existing = pd.read_excel('output.xlsx')
    except FileNotFoundError:
        df_existing = pd.DataFrame()
    df_combined = pd.concat([df_existing, df_new], axis=0, ignore_index=True)
    df_combined.to_excel('output.xlsx', index=False, engine='openpyxl', header=True)


def eval_func(user_model, tokenizer, args):
    import os
    import re
    import time
    import json
    import torch
    import habana_frameworks.torch.hpex
    import torch.nn.functional as F
    import lm_eval
    import lm_eval.tasks
    import lm_eval.evaluator

    # to avoid out-of-memory caused by Popen for large language models.
    lm_eval.metrics.bootstrap_stderr = itrex_bootstrap_stderr

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
            if (
                re.search("chatglm3", args.model.lower()) or
                re.search("llama", args.model.lower()) or
                re.search("mistral", args.model.lower())
            ):
                string = string.lstrip()
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

        def _model_generate(self, context, max_length, eos_token_id):
            raise NotImplementedError()

        def find_bucket(self, length):
            return [b for b in self.buckets if b >= length][0]

        def _model_call(self, inputs):
            seq_length = inputs.shape[-1]
            padding_length = 0
            bucket_length = self.find_bucket(seq_length)
            padding_length = bucket_length - seq_length
            inputs = F.pad(inputs, (0, padding_length), value=self.model.config.pad_token_id)
            logits = self.model(inputs.to(self._device))["logits"].cpu()

            if padding_length > 0:
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

    # make sure that result is dumped only once during multi-cards evaluation
    local_rank = int(os.getenv('LOCAL_RANK', '-1'))
    if local_rank in [-1, 0]:
        dumped = json.dumps(results, indent=2)
        accu_dict = {}
        case_name = str(args.approach) + "-" + args.precision
        for task_name in args.tasks:
            if task_name == "wikitext":
                print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["word_perplexity"]), flush=True)
                accu_dict[task_name] = [args.model, case_name, results["results"][task_name]["word_perplexity"]]
            else:
                print("Accuracy for %s is: %s" % (task_name, results["results"][task_name]["acc"]), flush=True)
                accu_dict[task_name] = [args.model, case_name, results["results"][task_name]["acc"]]
        accu_dict["duration"] = [args.model, case_name, results["duration"]]
        if args.dump_to_excel:
            save_to_excel(accu_dict)
    return results["results"][task_name]["acc"]
