import argparse
import datasets
import json
import logging
import math
import os
import sys
sys.path.insert(0, './neural-compressor')
sys.path.insert(0, './')

import random
import numpy as np
from itertools import chain
from pathlib import Path

import torch
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
set_seed(42)
def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn.functional import pad

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    OPTForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from neural_compressor.training import prepare_compression
from neural_compressor.training import WeightPruningConfig
from timers import CPUTimer, GPUTimer
from neural_compressor.compression.pruner import model_slim
from neural_compressor.compression.pruner import parse_auto_slim_config

check_min_version("4.27.0.dev0")
logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class Evaluator:
    def __init__(self, dataset, tokenizer, device, batch_size=16):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataloader = INCDataloader(dataset, tokenizer, self.device, batch_size)

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        if torch.cuda.is_available():
            my_timer = GPUTimer(timelogs = [])
        else:
            my_timer = CPUTimer(timelogs = [])
        warmup_steps = 10
        step = 0
        for input_ids, label, label_indices in tqdm(self.dataloader):
            with torch.no_grad():
                # if step == 0:
                #     model = torch.jit.trace(model, input_ids)
                step += 1
                # timing
                if step > warmup_steps: my_timer.__enter__()
                outputs = model(input_ids)
                if step > warmup_steps: my_timer.__exit__()
                last_token_logits = outputs[0][torch.arange(len(label_indices)), label_indices, :]
                pred = last_token_logits.argmax(dim=-1)
                total += label.size(0)
                hit += (pred == label).sum().item()
                if step % 100 == 0:
                    logger.info(f"eval step:{step}  accuracy:{float(hit/total)}")
        avg_latency = my_timer.get_avg_time()
        del my_timer
        accuracy = hit / total
        return accuracy, avg_latency


class INCDataloader():
    def __init__(self, dataset, tokenizer, device, batch_size=1):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        import math
        self.length = math.ceil(len(dataset) / self.batch_size)
        self.pad_len = 196

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    def pad_input(self, input):
        input_id = input['input_ids'].unsqueeze(0).to(self.device)
        label = input_id[:, -1].to(self.device)
        pad_len = self.pad_len - input_id.shape[1]
        label_index = -2 - pad_len
        input_id = pad(input_id, (0, pad_len), value=1)

        return (input_id, label, label_index)

    def __iter__(self):
        input_ids = None
        labels = None
        label_indices = None
        for idx, batch in enumerate(self.dataset):
            input_id, label, label_index = self.pad_input(batch)

            if input_ids is None:
                input_ids = input_id
                labels = label
                label_indices = [label_index]
            else:
                input_ids = torch.cat((input_ids, input_id), 0).to(self.device)
                labels = torch.cat((labels, label), 0).to(self.device)
                label_indices.append(label_index)

            if (idx + 1) % self.batch_size == 0:
                yield (input_ids, labels, label_indices)
                input_ids = None
                labels = None
                label_indices = None
        if (idx + 1) % self.batch_size != 0:
            yield (input_ids, labels, label_indices)

    def __len__(self):
        return self.length

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--calibration_dataset_name",
        type=str,
        default=None,
        help="The name of the pruning dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--evaluation_dataset_name",
        type=str,
        default=None,
        help="The name of the evaluation dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--device", default=0, type=str,
        help="device gpu int number, or 'cpu' ",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=42,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    # pruning config
    parser.add_argument(
        "--do_prune", action="store_true",
        help="Whether or not to prune the model"
    )
    parser.add_argument(
        "--max_pruning_steps",
        type=int,
        default=None,
        help="Total number of pruning steps to perform. If provided",
    )
    parser.add_argument(
        "--pruning_pattern",
        type=str, default="1x1",
        help="pruning pattern type, we support NxM and N:M."
    )
    parser.add_argument(
        "--target_sparsity",
        type=float, default=0.8,
        help="Target sparsity of the model."
    )
    parser.add_argument(
        "--pruning_frequency",
        type=int, default=-1,
        help="Sparse step frequency for iterative pruning, default to a quarter of pruning steps."
    )
    parser.add_argument(
        "--auto_slim", action="store_true",
        help="Whether or not to auto slim the model after pruning."
    )
    parser.add_argument(
        "--auto_config", action="store_true",
        help="Whether to automatically generate pruning configs."
    )
    parser.add_argument(
        "--max_length",
        type=int, default=2048,
        help="Maximum data length the model can receive."
    )
    parser.add_argument(
        "--trust_remote_code", default=True,
        help="Transformers parameter: use the external repo")
    ### DDP mode config
    parser.add_argument(
        "--local_rank",
        type=int, default=-1,
        help="Automatic DDP Multi-GPU argument, do not modify")
    
    args = parser.parse_args()
        
    # Sanity checks
    if args.calibration_dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args

def main():
    args = parse_args()

    # DDP Mode
    local_rank = args.local_rank
    if local_rank != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        batch_size = args.per_device_train_batch_size
        assert batch_size % WORLD_SIZE == 0, f'--batch-size {batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > local_rank, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(local_rank)
        # device = torch.device('cuda', local_rank)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if args.calibration_dataset_name is not None:
        # Downloading and loading a dataset from the hub.i
        if "wiki" in args.calibration_dataset_name:
            raw_datasets = load_dataset('wikitext', args.calibration_dataset_name, args.dataset_config_name)
        else:
            raw_datasets = load_dataset(args.calibration_dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset( #use the_pile's validation set for retraining pruning
                args.calibration_dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]"
            )
            raw_datasets["train"] = load_dataset(
                args.calibration_dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
            
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, torchscript=True)
    elif args.model_name_or_path:
        # torchscript will force `return_dict=False` to avoid jit errors
        config = AutoConfig.from_pretrained(args.model_name_or_path,
                                            torchscript=True, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    is_llama = bool("llama" in args.model_name_or_path)
    is_t5 = bool("t5" in args.model_name_or_path)
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        if is_llama:
            tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_name_or_path)
        else :
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if is_t5:
            model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                config=config,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=args.trust_remote_code,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    )

    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if local_rank != -1:
        device = torch.device("cuda", local_rank)
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], max_length=args.max_length, truncation=True) #padding
    #   return tokenizer(examples[text_column_name])

    if RANK in {-1, 0}:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    if RANK in {-1, 0}:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    train_dataset = lm_datasets["train"]
    
    # DataLoaders creation:
    train_dataset = train_dataset.shuffle(seed=42).select(range(128))
    total_batch_size = args.per_device_train_batch_size
    if local_rank != -1:
        total_batch_size *= WORLD_SIZE
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, \
                                                       sampler=train_sampler)
    else:
        train_dataloader = DataLoader(
                train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
        
    logger.info("***** Running pruning *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train/prune batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    
    if not args.auto_config:
        pruning_configs=[
            {
                "pruning_type": "sparse_gpt",
                "op_names": [".attn", "_proj", ".fc", "key", "dense", "_h"],
            }
        ]
    else:
        # auto slim config
        pruning_configs=[]
        auto_slim_configs = parse_auto_slim_config(
            model,
            ffn2_sparsity = args.target_sparsity,
            mha_sparsity = args.target_sparsity,
            pruning_type = "sparse_gpt",
            pattern = args.pruning_pattern
        )
        pruning_configs += auto_slim_configs
    configs = WeightPruningConfig(
        pruning_configs,
        target_sparsity=args.target_sparsity,
        pattern=args.pruning_pattern,
    )

    if args.do_prune:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        use_cache = model.config.use_cache
        model.config.use_cache = False
        # if torch.cuda.is_available():     # Larger models(e.g. 80G+) may not load into the video card memory.
        #     model = model.cuda()
        device = args.device
        if device != 'cpu':
            device = "cuda:"+str(device)
        from neural_compressor.compression.pruner import prepare_pruning
        pruning = prepare_pruning(model, configs,  dataloader=train_dataloader, device=device)
        model.config.use_cache = use_cache
        
    if args.output_dir is not None:
        ###TODO set ddp save method
        output_dir = args.output_dir
        if args.auto_slim:
            output_dir += "/before_slim"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    if args.evaluation_dataset_name != None:
        dataset_eval = load_dataset( 
            # for example:use the_pile's validation set for pruning, and lambada dataset for eval
            args.evaluation_dataset_name,
            args.dataset_config_name,
            split=f"validation",
        )
    else:
        dataset_eval = raw_datasets["validation"]
    dataset_eval = dataset_eval.shuffle(seed=42)
    evaluator = Evaluator(dataset_eval, tokenizer, model.device, batch_size=args.per_device_eval_batch_size)
    def eval_func(model):
        acc, avg_latency = evaluator.evaluate(model)
        return acc, avg_latency

    if not args.auto_slim:
        # only eval
        logger.info(f"***** Running Evaluation *****")
        acc, _ = eval_func(model)
        logger.info(f"pruned model accuracy:{acc}")
    else:
        logger.info(f"***** Running Evaluation before ffn auto slim*****")
        accuracy, avg_latency = eval_func(model)
        logger.info(f"accuracy:{accuracy}  avg_latency:{avg_latency}")
        model = model_slim(model, round_multiplier=32)

        logger.info(f"***** Running Evaluation after ffn auto_slim*****")
        accuracy, avg_latency = eval_func(model)
        logger.info(f"accuracy:{accuracy}  avg_latency:{avg_latency}")
    
    if RANK in {-1, 0}:
        if args.output_dir is not None and args.auto_slim:
            model.to('cpu')
            torch.save(model, args.output_dir+"/slimed_model.pt")
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of auto slim", auto_lfs_prune=True)
    
if __name__ == "__main__":
    main()
