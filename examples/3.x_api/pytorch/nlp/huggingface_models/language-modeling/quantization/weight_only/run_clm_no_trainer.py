import argparse
import os
import sys

sys.path.append('./')
import time
import json
import re
import torch
from datasets import load_dataset
from functools import lru_cache
import datasets
from packaging import version
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from neural_compressor.torch.utils import is_hpex_available


if is_hpex_available():
    import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
    htcore.hpu_set_inference_env()
device = "hpu" if is_hpex_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="?", default="EleutherAI/gpt-j-6b",
                    help="Path to pre-trained model (on the HF Hub or locally).")
parser.add_argument("--trust_remote_code", default=True,
                    help="""Whether to trust the execution of code from datasets/models defined on the Hub.
                    This option should only be set to `True` for repositories you trust and in which you have read the code, 
                    as it will execute code present on the Hub on your local machine.""")
parser.add_argument("--revision", default=None,
                    help="The specific model version to use (can be a branch name, tag name or commit id).")
parser.add_argument("--dataset", nargs="?", default="NeelNanda/pile-10k", const="NeelNanda/pile-10k",
                    help="Calibration dataset name.")
parser.add_argument("--output_dir", nargs="?", default="./saved_results",
                    help="Path to save the output results.")
parser.add_argument("--quantize", action="store_true",
                    help="Enable model quantization.")
parser.add_argument('--seed', type=int, default=42,
                    help='Seed for sampling the calibration data.')
parser.add_argument("--load", action="store_true",
                    help="Load weight-only quantized model from the `output_dir`.")
parser.add_argument("--accuracy", action="store_true",
                    help="Enable accuracy measurement.")
parser.add_argument("--performance", action="store_true",
                    help="Enable benchmarking measurement.")
parser.add_argument("--iters", default=100, type=int,
                    help="Number of inference iterations for benchmarking.")
parser.add_argument("--batch_size", default=1, type=int,
                    help="Input batch size for calibration and inference.")
parser.add_argument("--pad_max_length", default=512, type=int,
                    help="Pad input ids to max length.")
parser.add_argument("--calib_iters", default=512, type=int,
                    help="Number of calibration iterations.")
parser.add_argument("--tasks", default="lambada_openai,hellaswag,winogrande,piqa", type=str,
                    help="Tasks for accuracy validation.")
parser.add_argument("--peft_model_id", type=str, default=None,
                    help="Model name or path of peft model")

# ============WeightOnly configs===============
parser.add_argument("--woq_algo", default="RTN",
                    choices=['RTN', 'AWQ', 'TEQ', 'GPTQ', 'AutoRound', 'AutoTune'],
                    help="Specify the algorithm for weight-only quantization. Choices include: RTN, AWQ, TEQ, GPTQ, AutoRound, AutoTune")
parser.add_argument("--woq_bits", type=int, default=8,
                    help="Number of bits used to weights.")
parser.add_argument("--woq_dtype", type=str, default="int",
                    choices=['int', 'nf4', 'fp4'],
                    help="Data type for weights.  Choices include: int, nf4, fp4")
parser.add_argument("--woq_group_size", type=int, default=-1,
                    help="Size of weight groups, group_size=-1 refers to per output channel quantization.")
parser.add_argument("--woq_group_dim", type=int, default=1,
                    help="Dimension for grouping, group_dim=1 means grouping by input channels.")
parser.add_argument("--woq_scheme", default="sym",
                    help="Indicates whether weights are symmetric or asymmetric.")
parser.add_argument("--woq_use_mse_search", action="store_true",
                    help="Enables mean squared error (MSE) search.")
parser.add_argument("--woq_use_full_range", action="store_true",
                    help="Enables full range for activations.")
parser.add_argument("--quant_lm_head", action="store_true",  
                    help="Whether to quant the lm_head layer in transformers")
parser.add_argument("--use_hf_format", action="store_true",  
                    help="Whether to save & load quantized model in huggingface format")

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
                        'and your dataset builder args: args.pad_max_length.')

# =============AWQ configs====================
parser.add_argument("--use_auto_scale", action="store_true",
                    help="Enables best scales search based on activation distribution.")
parser.add_argument("--use_auto_clip", action="store_true",
                    help="Enables clip range search.")
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
parser.add_argument("--autoround_seq_len", default=2048, type=int, help="The sequence length for autoround calibration.")
parser.add_argument("--autoround_nsamples", default=128, type=int, help="num samples for autoround calibration.")
parser.add_argument("--autoround_attn_implementation", default="eager", type=str, help="The attention implementation.")
parser.add_argument(
    "--disable_quanted_input",
    action="store_true",
    help="whether to use the output of quantized block to tune the next block",
)

# =============DoubleQuant configs====================
parser.add_argument("--double_quant_type", type=str, default=None,
                    choices=['GGML_TYPE_Q4_K', 'BNB_NF4'],
                    help="""A key value to use preset configuration, 
                        GGML_TYPE_Q4_K refers to llama.cpp double quant configuration, 
                        while BNB_NF4 refers to bitsandbytes double quant configuration.""")
parser.add_argument("--double_quant_dtype", type=str, default="fp32",
                    help="Data type for double quant scale.")
parser.add_argument("--double_quant_bits", type=int, default=8,
                    help="Number of bits used to represent double_quant scale.")
parser.add_argument("--double_quant_use_sym", type=bool, default=True,
                    help="Indicates whether double quant scale are symmetric.")
parser.add_argument("--double_quant_group_size", type=int, default=256,
                    help="Size of double quant groups.")
# =======================================

args = parser.parse_args()
calib_size = 1


def compare_versions(v1, v2):
    return version.parse(v1) >= version.parse(v2)


def torch_version_at_least(version_string):
    return compare_versions(torch.__version__, version_string)


TORCH_VERSION_AT_LEAST_2_4 = torch_version_at_least("2.4.0")


def check_torch_compile_with_hpu_backend():
    assert TORCH_VERSION_AT_LEAST_2_4, "Please use torch>=2.4.0 to use torch compile with HPU backend."
    if os.environ.get("PT_HPU_LAZY_MODE") != "0":
        raise ValueError("Please set `PT_HPU_LAZY_MODE=0` to use torch compile with HPU backend.")
    if os.environ.get("PT_ENABLE_INT64_SUPPORT") != "1":
        raise ValueError("Please set `PT_ENABLE_INT64_SUPPORT=1` to use torch compile with HPU backend.")


def set_envs_for_torch_compile_with_hpu_backend():
    import torch._dynamo.config as dynamo_config
    import torch._inductor.config as inductor_config

    inductor_config.force_disable_caches = True
    dynamo_config.inline_inbuilt_nn_modules = True


@lru_cache(None)
def is_habana_framework_installed():
    """Check if Habana framework is installed.

    Only check for the habana_frameworks package without importing it to avoid
    initializing lazy-mode-related components.
    """
    from importlib.util import find_spec

    package_spec = find_spec("habana_frameworks")
    return package_spec is not None

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
    if args.woq_algo in ["AWQ", "TEQ"]:
        torchscript = True
    if args.woq_algo == "AutoRound" and is_habana_framework_installed():
        print("Quantizing model with AutoRound on HPU")
        if args.quantize:
            check_torch_compile_with_hpu_backend()
            set_envs_for_torch_compile_with_hpu_backend()
        user_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.autoround_attn_implementation,
            revision=args.revision,
        )
    else:
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

def eval_fn(user_model=None):
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
        device=device,
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
    return acc

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
        TuningConfig,
        autotune,
        get_woq_tuning_config,
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
                    "quant_lm_head": args.quant_lm_head,
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
                quant_lm_head=args.quant_lm_head,
            )
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
        from neural_compressor.torch.utils import get_model_device, move_input_device
        from tqdm import tqdm
        def run_fn_for_gptq(model, dataloader_for_calibration, *args):
            for batch in tqdm(dataloader_for_calibration):
                device = get_model_device(model)
                batch = move_input_device(batch, device=device)
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
                    "quant_lm_head": args.quant_lm_head,
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
                quant_lm_head=args.quant_lm_head,
            )
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
            quant_lm_head=args.quant_lm_head,
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
            quant_lm_head=args.quant_lm_head,
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
                seqlen=args.autoround_seq_len,
                nsamples=args.autoround_nsamples,
                iters=args.autoround_iters,
            )
        quant_config.set_local("lm_head", AutoRoundConfig(dtype="fp32"))
        from neural_compressor.torch.algorithms.weight_only.autoround import get_dataloader
        dataloader = get_dataloader(tokenizer=tokenizer,
                                                seqlen=args.autoround_seq_len,
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
    elif args.woq_algo == "AutoTune":
        from utils import DataloaderPreprocessor
        dataloaderPreprocessor = DataloaderPreprocessor(
            dataloader_original=calib_dataloader,
            use_max_length=args.gptq_use_max_length,
            max_seq_length=args.gptq_max_seq_length,
        )
        dataloader = dataloaderPreprocessor.get_prepared_dataloader()
        custom_tune_config = TuningConfig(config_set=get_woq_tuning_config())
        from neural_compressor.torch.utils import get_model_device, move_input_device
        from tqdm import tqdm
        def run_fn_for_gptq(model, dataloader_for_calibration, *args):
            for batch in tqdm(dataloader_for_calibration):
                device = get_model_device(model)
                batch = move_input_device(batch, device=device)
                if isinstance(batch, tuple) or isinstance(batch, list):
                    model(batch[0])
                elif isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)
            return
        example_inputs = torch.ones([1, args.pad_max_length], dtype=torch.long)
        user_model = autotune(
            model=user_model,
            tune_config=custom_tune_config,
            eval_fn=eval_fn,
            run_fn=run_fn_for_gptq,
            run_args=(dataloader, True),  # run_args should be a tuple,
            example_inputs=example_inputs,
        )

    print("saving weight-only quantized model")
    if args.use_hf_format:
        user_model.save(args.output_dir, format="huggingface")
        tokenizer.save_pretrained(args.output_dir)
    else:
        user_model.save(args.output_dir)
    print("saved weight-only quantized model")


if args.load:
    print("load weight-only quantized model")

    from neural_compressor.torch.quantization import load
    if args.use_hf_format:
        user_model = load(args.model, format="huggingface", device=device)
    else:
        user_model, _ = get_user_model()
        config = AutoConfig.from_pretrained(args.model)
        user_model = load(
            os.path.abspath(os.path.expanduser(args.output_dir)),
            user_model,
            device=device,
        )
        setattr(user_model, "config", config)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
else:
    user_model, tokenizer = get_user_model()


if is_hpex_available():
    from habana_frameworks.torch.hpu.graphs import wrap_in_hpu_graph
    user_model = user_model.to(torch.bfloat16)
    wrap_in_hpu_graph(user_model, max_graphs=10)


if args.accuracy:
    user_model.eval()
    from neural_compressor.evaluation.lm_eval import evaluate, LMEvalParser
    eval_args = LMEvalParser(
        model="hf",
        user_model=user_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        tasks=args.tasks,
        device=device,
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
    batch_size, input_leng = args.batch_size, 512
    example_inputs = torch.ones((batch_size, input_leng), dtype=torch.long)
    print("Batch size = {:d}".format(batch_size))
    print("The length of input tokens = {:d}".format(input_leng))
    import time

    total_iters = args.iters
    warmup_iters = 5
    with torch.no_grad():
        for i in range(total_iters):
            if i == warmup_iters:
                start = time.time()
            user_model(example_inputs)
        end = time.time()
    latency = (end - start) / ((total_iters - warmup_iters) * args.batch_size)
    throughput = ((total_iters - warmup_iters) * args.batch_size) / (end - start)
    print("Latency: {:.3f} ms".format(latency * 10**3))
    print("Throughput: {:.3f} samples/sec".format(throughput))
