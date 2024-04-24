import argparse
import os
import sys

sys.path.append('./')
import time
import json
import re
import torch
from datasets import load_dataset
import datasets
from torch.nn.functional import pad
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", nargs="?", default="EleutherAI/gpt-j-6b"
)
parser.add_argument(
    "--trust_remote_code", default=True,
    help="Transformers parameter: use the external repo")
parser.add_argument(
    "--revision", default=None,
    help="Transformers parameter: set the model hub commit number")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k")
parser.add_argument("--output_dir", nargs="?", default="./saved_results")
parser.add_argument("--quantize", action="store_true")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="By default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument(
    '--seed',
    type=int, default=42, help='Seed for sampling the calibration data.'
)
parser.add_argument("--approach", type=str, default='static',
                    help="Select from ['dynamic', 'static', 'weight-only']")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--ipex", action="store_true", help="Use intel extension for pytorch.")
parser.add_argument("--accuracy", action="store_true")
parser.add_argument("--performance", action="store_true")
parser.add_argument("--iters", default=100, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="For accuracy measurement only.")
parser.add_argument("--save_accuracy_path", default=None,
                    help="Save accuracy results path.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=512, type=int,
                    help="calibration iters.")
parser.add_argument("--tasks", default="lambada_openai,hellaswag,winogrande,piqa,wikitext",
                    type=str, help="tasks for accuracy validation, text-generation and code-generation tasks are different.")
parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")
# ============SmoothQuant configs==============
parser.add_argument("--sq", action="store_true")
parser.add_argument("--alpha", default="auto", help="Smooth quant parameter.")
# ============WeightOnly configs===============
parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ', 'GPTQ'],
                    help="Weight-only parameter.")
parser.add_argument("--woq_bits", type=int, default=8)
parser.add_argument("--woq_group_size", type=int, default=-1)
parser.add_argument("--woq_scheme", default="sym")
parser.add_argument("--woq_enable_mse_search", action="store_true")
parser.add_argument("--woq_enable_full_range", action="store_true")
# =============GPTQ configs====================
parser.add_argument("--gptq_actorder", action="store_true",
                    help="Whether to apply the activation order GPTQ heuristic.")
parser.add_argument('--gptq_percdamp', type=float, default=.01,
                    help='Percent of the average Hessian diagonal to use for dampening.')
parser.add_argument('--gptq_block_size', type=int, default=128, help='Block size. sub weight matrix size to run GPTQ.')
parser.add_argument('--gptq_nsamples', type=int, default=128, help='Number of calibration data samples.')
parser.add_argument('--gptq_use_max_length', action="store_true",
                    help='Set all sequence length to be same length of args.gptq_pad_max_length')
parser.add_argument('--gptq_pad_max_length', type=int, default=2048, help='Calibration dataset sequence max length, \
                                                                           this should align with your model config, \
                                                                           and your dataset builder args: args.pad_max_length')
parser.add_argument('--gptq_static_groups', action='store_true', help='Use determined group to do quantization')
# ==============code generation args===========
parser.add_argument("--code_generation", action="store_true")
parser.add_argument("--n_samples", default=200, type=int)
parser.add_argument(
    "--limit", default=None, type=int, help="Limit number of samples to eval"
)
parser.add_argument("--allow_code_execution", action="store_true")
parser.add_argument("--prefix", default="")
parser.add_argument("--generation_only", action="store_true")
parser.add_argument("--postprocess", action="store_false")
parser.add_argument("--save_references", action="store_true")
parser.add_argument("--save_generations", action="store_true")
parser.add_argument("--instruction_tokens", default=None)
parser.add_argument("--save_generations_path", default="generations.json")
parser.add_argument("--load_generations_path", default=None)
parser.add_argument("--metric_output_path", default="evaluation_results.json")
parser.add_argument("--max_length_generation", default=512, type=int)
parser.add_argument("--temperature", default=0.8, type=float)
parser.add_argument("--top_p", default=0.8, type=float)
parser.add_argument("--top_k", default=0, type=int)
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--check_references", action="store_true")
parser.add_argument("--max_memory_per_gpu", type=str, default=None)
parser.add_argument(
    "--modeltype",
    default="causal",
    help="AutoModel to use, it can be causal or seq2seq",
)
parser.add_argument(
    "--limit_start",
    type=int,
    default=0,
    help="Optional offset to start from when limiting the number of samples",
)

args = parser.parse_args()
if args.ipex:
    import intel_extension_for_pytorch as ipex
calib_size = 1


class Evaluator:
    def __init__(self, dataset, tokenizer, batch_size=8, pad_val=1, pad_max=196, is_calib=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.pad_val = pad_val
        self.pad_max = pad_max
        self.is_calib = is_calib

        # tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def tokenize_function(self, examples):
        if args.woq_algo in ['TEQ']:
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            example = self.tokenizer(examples["text"], padding="max_length", max_length=self.pad_max)
        else:
            example = self.tokenizer(examples["text"])
        return example

    @torch.no_grad()
    def collate_batch(self, batch):

        input_ids_padded = []
        last_ind = []

        for text in batch:
            input_ids = text["input_ids"]
            pad_len = self.pad_max - input_ids.shape[0]
            last_ind.append(input_ids.shape[0] - 1)
            if self.is_calib:
                if args.woq_algo != 'GPTQ':
                    input_ids = input_ids[:self.pad_max] if len(input_ids) > self.pad_max else input_ids
            else:
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_val)
            input_ids_padded.append(input_ids)

        return (torch.vstack(input_ids_padded), torch.tensor(last_ind))

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        latency = 0
        test_dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_batch,
        )
        for i, (input_ids, last_ind) in enumerate(test_dataloader):
            label = input_ids[torch.arange(len(last_ind)), last_ind]
            input_ids[torch.arange(len(last_ind)), last_ind] = self.pad_val
            pad_len = self.pad_max - last_ind - 1

            start = time.time()
            outputs = model(input_ids)
            latency += time.time() - start

            last_token_logits = outputs[0][torch.arange(len(last_ind)), -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            if (i + 1) % 50 == 0:
                print(hit / total)
                print("Processed minibatch:", i)

        acc = hit / total
        print("Accuracy: ", acc)
        print("Latency: ", latency)
        return acc


def get_user_model():
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    torchscript = False
    if args.sq or args.ipex or args.woq_algo in ['AWQ', 'TEQ']:
        torchscript = True
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if args.approach == 'weight_only':
        user_model = user_model.float()

    # Set model's seq_len when GPTQ calibration is enabled.
    if args.woq_algo == 'GPTQ':
        user_model.seqlen = args.gptq_pad_max_length

    if args.peft_model_id is not None:
        from peft import PeftModel
        user_model = PeftModel.from_pretrained(user_model, args.peft_model_id)

    # to channels last
    user_model = user_model.to(memory_format=torch.channels_last)
    user_model.eval()
    return user_model, tokenizer


if args.quantize:
    # dataset
    user_model, tokenizer = get_user_model()
    calib_dataset = load_dataset(args.dataset, split="train")
    # calib_dataset = datasets.load_from_disk('/your/local/dataset/pile-10k/') # use this if trouble with connecting to HF
    calib_dataset = calib_dataset.shuffle(seed=args.seed)
    calib_evaluator = Evaluator(calib_dataset, tokenizer, args.batch_size, pad_max=args.pad_max_length, is_calib=True)
    calib_dataloader = DataLoader(
        calib_evaluator.dataset,
        batch_size=calib_size,
        shuffle=False,
        collate_fn=calib_evaluator.collate_batch,
    )


    def calib_func(prepared_model):
        for i, calib_input in enumerate(calib_dataloader):
            if i > args.calib_iters:
                break
            prepared_model(calib_input[0])


    recipes = {}
    eval_func = None
    from neural_compressor import PostTrainingQuantConfig, quantization

    # specify the op_type_dict and op_name_dict
    if args.approach == 'weight_only':
        op_type_dict = {
            '.*': {  # re.match
                "weight": {
                    'bits': args.woq_bits,  # 1-8 bits
                    'group_size': args.woq_group_size,  # -1 (per-channel)
                    'scheme': args.woq_scheme,  # sym/asym
                    'algorithm': args.woq_algo,  # RTN/AWQ/TEQ
                },
            },
        }
        op_name_dict = {
            'lm_head': {"weight": {'dtype': 'fp32'}, },
            'embed_out': {"weight": {'dtype': 'fp32'}, },  # for dolly_v2
        }
        recipes["rtn_args"] = {
            "enable_mse_search": args.woq_enable_mse_search,
            "enable_full_range": args.woq_enable_full_range,
        }
        recipes['gptq_args'] = {
            'percdamp': args.gptq_percdamp,
            'act_order': args.gptq_actorder,
            'block_size': args.gptq_block_size,
            'nsamples': args.gptq_nsamples,
            'use_max_length': args.gptq_use_max_length,
            'pad_max_length': args.gptq_pad_max_length,
            'static_groups': args.gptq_static_groups,
            "enable_mse_search": args.woq_enable_mse_search,
        }
        # GPTQ: use assistive functions to modify calib_dataloader and calib_func
        # TEQ: set calib_func=None, use default training func as calib_func
        if args.woq_algo in ["GPTQ", "TEQ"]:
            calib_func = None

        conf = PostTrainingQuantConfig(
            approach=args.approach,
            op_type_dict=op_type_dict,
            op_name_dict=op_name_dict,
            recipes=recipes,
        )
    else:
        if re.search("gpt", user_model.config.model_type):
            op_type_dict = {
                "add": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}},
            }
        else:
            op_type_dict = {}
        excluded_precisions = [] if args.int8_bf16_mixed else ["bf16"]
        if args.sq:
            # alpha can be a float number of a list of float number.
            args.alpha = args.alpha if args.alpha == "auto" else eval(args.alpha)
            if re.search("falcon", user_model.config.model_type):
                recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': args.alpha, 'folding': False}}
            else:
                recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': args.alpha}}

        conf = PostTrainingQuantConfig(
            backend="ipex" if args.ipex else "default",
            approach=args.approach,
            excluded_precisions=excluded_precisions,
            op_type_dict=op_type_dict,
            recipes=recipes,
        )

        # eval_func should be set when tuning alpha.
        if isinstance(args.alpha, list):
            eval_dataset = load_dataset('lambada', split='validation')
            evaluator = Evaluator(eval_dataset, tokenizer)

            def eval_func(model):
                acc = evaluator.evaluate(model)
                return acc

    q_model = quantization.fit(
        user_model,
        conf,
        calib_dataloader=calib_dataloader,
        calib_func=calib_func,
        eval_func=eval_func,
    )

    q_model.save(args.output_dir)

if args.int8 or args.int8_bf16_mixed:
    print("load int8 model")
    from neural_compressor.utils.pytorch import load

    if args.ipex:
        user_model = load(os.path.abspath(os.path.expanduser(args.output_dir)))
    else:
        user_model, _ = get_user_model()
        kwargs = {'weight_only': True} if args.approach == 'weight_only' else {}
        user_model = load(os.path.abspath(os.path.expanduser(args.output_dir)), user_model, **kwargs)
else:
    user_model, _ = get_user_model()

if args.accuracy:
    user_model.eval()
    if args.code_generation:
        from intel_extension_for_transformers.transformers.llm.evaluation.bigcode_eval import evaluate
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        results = evaluate(
            model=user_model,
            tokenizer=tokenizer,
            tasks=args.tasks,
            batch_size=args.batch_size,
            args=args,
        )
        for task_name in args.tasks:
            if task_name == "truthfulqa_mc":
                acc = results["results"][task_name]["mc1"]
            else:
                acc = results["results"][task_name]["acc"]
    else:
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        eval_args = LMEvalParser(
            model="hf", 
            user_model=user_model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            tasks=args.tasks,
            device="cpu",
        )
        results = evaluate(eval_args)
        for task_name in args.tasks.split(","):
            if task_name == "wikitext":
                acc = results["results"][task_name]["word_perplexity,none"]
            else:
                acc = results["results"][task_name]["acc,none"]

    print("Accuracy: %.5f" % acc)
    print('Batch size = %d' % args.batch_size)

if args.performance:
    import time
    user_model.eval()
    samples = args.iters * args.batch_size

    if args.code_generation:
        from intel_extension_for_transformers.transformers.llm.evaluation.bigcode_eval import evaluate
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        start = time.time()
        results = evaluate(
            model=user_model,
            tokenizer=tokenizer,
            tasks=args.tasks,
            batch_size=args.batch_size,
            args=args,
        )
        end = time.time()
        for task_name in args.tasks:
            if task_name == "truthfulqa_mc":
                acc = results["results"][task_name]["mc1"]
            else:
                acc = results["results"][task_name]["acc"]
    else:
        from intel_extension_for_transformers.transformers.llm.evaluation.lm_eval import evaluate, LMEvalParser
        eval_args = LMEvalParser(
            model="hf", 
            user_model=user_model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            tasks=args.tasks,
            device="cpu",
        )
        start = time.time()
        results = evaluate(eval_args)
        end = time.time()
        for task_name in args.tasks.split(","):
            if task_name == "wikitext":
                acc = results["results"][task_name]["word_perplexity,none"]
            else:
                acc = results["results"][task_name]["acc,none"]
    print("Accuracy: %.5f" % acc)
    print('Throughput: %.3f samples/sec' % (samples / (end - start)))
    print('Latency: %.3f ms' % ((end - start) * 1000 / samples))
    print('Batch size = %d' % args.batch_size)
