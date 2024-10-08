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
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from neural_compressor.torch.utils import is_hpex_available

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
parser.add_argument("--int8", action="store_true")
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
                    type=str, help="tasks for accuracy validation")
parser.add_argument("--peft_model_id", type=str, default=None, help="model_name_or_path of peft model")
# ============WeightOnly configs===============
parser.add_argument("--woq_algo", default="RTN", choices=['RTN', 'AWQ', 'TEQ', 'GPTQ', 'AutoRound'],
                    help="Weight-only parameter.")
parser.add_argument("--woq_bits", type=int, default=8)
parser.add_argument("--woq_dtype", type=str, default="int")
parser.add_argument("--woq_group_size", type=int, default=-1)
parser.add_argument("--woq_group_dim", type=int, default=1)
parser.add_argument("--woq_scheme", default="sym")
parser.add_argument("--woq_use_mse_search", action="store_true")
parser.add_argument("--woq_use_full_range", action="store_true")
# =============GPTQ configs====================
parser.add_argument("--gptq_actorder", action="store_true",
                    help="Whether to apply the activation order GPTQ heuristic.")
parser.add_argument('--gptq_percdamp', type=float, default=.01,
                    help='Percent of the average Hessian diagonal to use for dampening.')
parser.add_argument('--gptq_block_size', type=int, default=128, help='Block size. sub weight matrix size to run GPTQ.')
parser.add_argument('--gptq_static_groups', action="store_true",
                    help="Whether to calculate group wise quantization parameters in advance. "
                        "This option mitigate actorder's extra computational requirements.")
parser.add_argument('--gptq_nsamples', type=int, default=128, help='Number of calibration data samples.')
parser.add_argument('--gptq_use_max_length', action="store_true",
                    help='Set all sequence length to be same length of args.gptq_max_seq_length')
parser.add_argument('--gptq_max_seq_length', type=int, default=2048,
                    help='Calibration dataset sequence max length, '
                        'this should align with your model config, '
                        'and your dataset builder args: args.pad_max_length')
# =============AWQ configs====================
parser.add_argument("--use_auto_scale", action="store_true",
                    help="Enables best scales search based on activation distribution.")
parser.add_argument("--use_auto_clip", action="store_true",
                    help="Enables clip range searchc.")
parser.add_argument("--folding", action="store_true",
                    help="Allow insert mul before linear when the scale cannot be absorbed by last layer for TEQ/AWQ.")
parser.add_argument('--absorb_layer_dict', type=dict, default={},
                    help="The layer dict that scale can be absorbed for TEQ/AWQ.")
# ============AUTOROUND configs==============
parser.add_argument(
    "--lr",
    type=float,
    default=None,
    help="learning rate, if None, it will be set to 1.0/iters automatically",
)
parser.add_argument(
    "--minmax_lr",
    type=float,
    default=None,
    help="minmax learning rate, if None,it will beset to be the same with lr",
)
parser.add_argument("--autoround_iters", default=200, type=int, help="num iters for autoround calibration.")
parser.add_argument("--autoround_nsamples", default=128, type=int, help="num samples for autoround calibration.")
parser.add_argument(
    "--disable_quanted_input",
    action="store_true",
    help="whether to use the output of quantized block to tune the next block",
)

# =============DoubleQuant configs====================
parser.add_argument("--double_quant_type",
                    type=str,
                    default=None,
                    choices=['GGML_TYPE_Q4_K', 'BNB_NF4'],
                    help="DoubleQuant parameter")
parser.add_argument("--double_quant_dtype",
                    type=str,
                    default="fp32",
                    help="Data type for double quant scale.")
parser.add_argument("--double_quant_bits",
                    type=int,
                    default=8,
                    help="Number of bits used to represent double_quant scale.")
parser.add_argument("--double_quant_use_sym",
                    type=bool,
                    default=True,
                    help="Indicates whether double quant scale are symmetric.")
parser.add_argument("--double_quant_group_size",
                    type=int,
                    default=256,
                    help="Size of double quant groups.")
# =======================================

args = parser.parse_args()
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
    torchscript = False
    if  args.woq_algo in ['AWQ', 'TEQ']:
        torchscript = True
    user_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torchscript=torchscript,  # torchscript will force `return_dict=False` to avoid jit errors
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    user_model = user_model.float()
    if args.woq_algo == 'AutoRound':
        user_model.to(torch.float32)

    # Set model's seq_len when GPTQ calibration is enabled.
    if args.woq_algo == 'GPTQ':
        user_model.seqlen = args.gptq_max_seq_length

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

    # 3.x api
    from neural_compressor.torch.quantization import (
        RTNConfig,
        GPTQConfig,
        AWQConfig,
        AutoRoundConfig,
        TEQConfig,
        prepare,
        convert
    )
    from neural_compressor.torch.utils import get_double_quant_config_dict
    weight_sym = True if args.woq_scheme == "sym" else False
    if args.double_quant_type is not None:
        double_quant_config_dict = get_double_quant_config_dict(args.double_quant_type)

    if args.woq_algo == "RTN":
        if args.double_quant_type is not None:
            double_quant_config_dict.update(
                {
                    # TODO: add group_dim into double quant config?
                    "use_full_range": args.woq_use_full_range,
                    "use_mse_search": args.woq_use_mse_search,
                }
            )
            quant_config = RTNConfig.from_dict(double_quant_config_dict)
        else:
            quant_config = RTNConfig(
                dtype=args.woq_dtype,
                bits=args.woq_bits,
                use_sym=weight_sym,
                group_size=args.woq_group_size,
                group_dim=args.woq_group_dim,
                use_full_range=args.woq_use_full_range,
                use_mse_search=args.woq_use_mse_search,
                use_double_quant=False,
                double_quant_bits=args.double_quant_bits,
                double_quant_dtype=args.double_quant_dtype,
                double_quant_use_sym=args.double_quant_use_sym,
                double_quant_group_size=args.double_quant_group_size,
            )
        quant_config.set_local("lm_head", RTNConfig(dtype="fp32"))
        user_model = prepare(model=user_model, quant_config=quant_config)
        user_model = convert(model=user_model)
    elif args.woq_algo == "GPTQ":
        from utils import DataloaderPreprocessor
        dataloaderPreprocessor = DataloaderPreprocessor(
            dataloader_original=calib_dataloader,
            use_max_length=args.gptq_use_max_length,
            max_seq_length=args.gptq_max_seq_length,
        )
        dataloader_for_calibration = dataloaderPreprocessor.get_prepared_dataloader()
        from neural_compressor.torch.algorithms.weight_only.utility import move_input_to_device
        from tqdm import tqdm
        def run_fn_for_gptq(model, dataloader_for_calibration, *args):
            for batch in tqdm(dataloader_for_calibration):
                batch = move_input_to_device(batch, device=None)
                if isinstance(batch, tuple) or isinstance(batch, list):
                    model(batch[0])
                elif isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)
            return
        if args.double_quant_type is not None:
            double_quant_config_dict.update(
                {
                    "use_mse_search": args.woq_use_mse_search,
                    "percdamp": args.gptq_percdamp,
                    "act_order": args.gptq_actorder,
                    "block_size": args.gptq_block_size,
                    "static_groups": args.gptq_static_groups,
                }
            )
            quant_config = GPTQConfig.from_dict(double_quant_config_dict)
        else:
            quant_config = GPTQConfig(
                dtype=args.woq_dtype,
                bits=args.woq_bits,
                use_sym=weight_sym,
                group_size=args.woq_group_size,
                use_mse_search=args.woq_use_mse_search,
                percdamp=args.gptq_percdamp,
                act_order=args.gptq_actorder,
                block_size=args.gptq_block_size,
                static_groups=args.gptq_static_groups,
                use_double_quant=False,
                double_quant_bits=args.double_quant_bits,
                double_quant_dtype=args.double_quant_dtype,
                double_quant_use_sym=args.double_quant_use_sym,
                double_quant_group_size=args.double_quant_group_size,
            )
        quant_config.set_local("lm_head", GPTQConfig(dtype="fp32"))
        user_model = prepare(model=user_model, quant_config=quant_config)
        run_fn_for_gptq(user_model, dataloader_for_calibration)
        user_model = convert(user_model)
    elif args.woq_algo == "AWQ":
        quant_config = AWQConfig(
            dtype=args.woq_dtype,
            bits=args.woq_bits,
            use_sym=weight_sym,
            group_size=args.woq_group_size,
            group_dim=args.woq_group_dim,
            use_auto_scale=args.use_auto_scale,
            use_auto_clip=args.use_auto_clip,
            folding=args.folding,
            absorb_layer_dict=args.absorb_layer_dict,
        )
        example_inputs = torch.ones([1, args.pad_max_length], dtype=torch.long)
        run_fn = calib_func
        user_model = prepare(model=user_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(user_model)
        user_model = convert(user_model)
    elif args.woq_algo == "TEQ":
        quant_config = TEQConfig(
            dtype=args.woq_dtype,
            bits=args.woq_bits,
            use_sym=weight_sym,
            group_size=args.woq_group_size,
            group_dim=args.woq_group_dim,
            folding=args.folding,
        )
        example_inputs = torch.ones([1, args.pad_max_length], dtype=torch.long)
        run_fn = calib_func
        user_model = prepare(model=user_model, quant_config=quant_config, example_inputs=example_inputs)
        run_fn(user_model)
        user_model = convert(user_model)
    elif args.woq_algo == "AutoRound":
        quant_config = AutoRoundConfig(
                dtype=args.woq_dtype,
                bits=args.woq_bits,
                use_sym=weight_sym,
                group_size=args.woq_group_size,
                enable_quanted_input=not args.disable_quanted_input,
                lr=args.lr,
                minmax_lr=args.minmax_lr,
                seqlen=args.pad_max_length,
                nsamples=args.autoround_nsamples,
                iters=args.autoround_iters,
            )
        quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))
        from neural_compressor.torch.algorithms.weight_only.autoround import get_dataloader
        dataloader = get_dataloader(tokenizer=tokenizer,
                                                seqlen=args.pad_max_length,
                                                dataset_name=datasets,
                                                seed=args.seed,
                                                bs=args.batch_size,
                                                nsamples=args.autoround_nsamples)
        @torch.no_grad()
        def run_fn_for_autoround(model, dataloader):
            for data in dataloader:
                if isinstance(data, tuple) or isinstance(data, list):
                    model(*data)
                elif isinstance(data, dict):
                    model(**data)
                else:
                    model(data)
        run_fn = run_fn_for_autoround
        run_args = (dataloader,)
        user_model = prepare(model=user_model, quant_config=quant_config)
        run_fn(user_model, *run_args)
        user_model = convert(user_model)
    
        

    user_model.save(args.output_dir)


# TODO: we need run_benchmark.sh for loading and remove --accuracy in run_quant.sh, currently run_quant.sh will get fp32 result

if args.int8 or args.int8_bf16_mixed:
    print("load int8 model")

    from neural_compressor.torch.quantization import load
    user_model, _ = get_user_model()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    user_model = load(
        os.path.abspath(os.path.expanduser(args.output_dir)),
        user_model,
        device="hpu" if is_hpex_available() else "cpu",
    )
    setattr(user_model, "config", config)
else:
    user_model, tokenizer = get_user_model()


if args.accuracy:
    user_model.eval()
    from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
    eval_args = LMEvalParser(
        model="hf",
        user_model=user_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        device="hpu" if is_hpex_available() else "cpu",
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
    user_model.eval()
    from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
    import time

    samples = args.iters * args.batch_size
    eval_args = LMEvalParser(
        model="hf",
        user_model=user_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        limit=samples,
        device="hpu" if is_hpex_available() else "cpu",
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
