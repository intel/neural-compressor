#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from accelerate.utils import set_seed
set_seed(42)
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
import argparse
import json
import logging
import math
import os
import sys
sys.path.insert(0, './neural-compressor')
sys.path.insert(0, './')
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
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

    
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
    
    
class Net(torch.nn.Module):
    def __init__(self, ori_model):
        super(Net, self).__init__()
        self.model = ori_model
    def forward(self, input_ids, pastkv, mask):
        return self.model(input_ids=input_ids, attention_mask=mask, past_key_values=pastkv, return_dict=False)
        
def trace_model(model, tokenizer):
    from optimum.utils import NormalizedConfigManager
    normalized_config = NormalizedConfigManager.get_normalized_config_class(model.config.model_type)(model.config)
    num_layers = normalized_config.num_layers
    num_attention_heads = normalized_config.num_attention_heads
    hidden_size = normalized_config.hidden_size
    d_k = hidden_size // num_attention_heads
    model_type = model.config.model_type
    model = model.cpu()
    model.eval()
    prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
    " She wanted to go to places and meet new people, and have fun."
    init_input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    traced_model = None
    if 'llama' in model_type:
        input_ids = init_input_ids.clone()
        attention_mask = torch.ones(len(input_ids)+1)
        attention_mask[0] = 0
        input_ids = input_ids[0:1].unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        past_key_value = tuple([(torch.zeros([1,32,34,128]), torch.zeros([1,32,34,128])) for i in range(32)])
        if 'llama_13b' in model_type:
            past_key_value = tuple([(torch.zeros([1,40,34,128]), torch.zeros([1,40,34,128])) for i in range(40)])
        net = model
        traced_model = torch.jit.trace(net, (input_ids, attention_mask, past_key_value))
    else:
        input_ids = init_input_ids.clone().unsqueeze(0)
        attention_mask = torch.ones(len(input_ids)).unsqueeze(0)
        past_key_value = tuple([(torch.zeros([1,num_attention_heads,0,d_k]),
                                    torch.zeros([1,num_attention_heads,0,d_k])) for i in range(num_layers)])
        net = Net(model)
        traced_model = torch.jit.trace(net, (input_ids, past_key_value, attention_mask))
    return traced_model
    

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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        "--cooldown_epochs",
        type=int, default=0,
        help="Cooling epochs after pruning."
    )
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
        type=str, default="channelx1",
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

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None: # Already set at the beginning of the file
    #     set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.calibration_dataset_name is not None:
        # Downloading and loading a dataset from the hub.i
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
        config = AutoConfig.from_pretrained(args.model_name_or_path, torchscript=True)
        # config = None
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    
    model_name = model.config.model_type
    
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        if 'llama' in model_name:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
        else :
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], max_length=args.max_length, truncation=True) #padding
    #   return tokenizer(examples[text_column_name])
    

    with accelerator.main_process_first():
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

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    max_sample_num = args.max_pruning_steps * total_batch_size
    train_dataset = train_dataset.shuffle(seed=42).select(range(max_sample_num))
    train_dataloader = DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Prepare everything with our `accelerator`.
    model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Pruning!
    logger.info("***** Running Pruning *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total pruning steps = {args.max_pruning_steps}")

    # Pruning preparation 
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_iterations = num_update_steps_per_epoch
    num_warm = args.num_warmup_steps
    total_iterations = args.max_pruning_steps
    frequency = int((total_iterations - num_warm + 1) / 40) if args.pruning_frequency == -1 \
                                                           else args.pruning_frequency
    pruning_start = max(num_warm, 1)
    pruning_end = max(total_iterations - 1, pruning_start)
    if not args.do_prune:
        pruning_start = args.max_pruning_steps + 1
        pruning_end = pruning_start
        
    if not args.auto_config:
        pruning_configs=[
            {
                "pruning_type": "retrain_free",
                "pruning_scope": "global",
                "op_names": ['.fc', '.mlp'],
                "excluded_op_names": [".attn"],
                "sparsity_decay_type": "exp",
                "pattern": "channelx1",
                "pruning_op_types": ["Linear"],
                "max_sparsity_ratio_per_op": 0.98,
            },
        ]
    else:
        # auto config
        pruning_configs=[]
        auto_configs = parse_auto_slim_config(
            model,
            ffn2_sparsity = args.target_sparsity,
            mha_sparsity = 0,
            pruning_scope = "global",
            pruning_type = "retrain_free",
        )
        pruning_configs += auto_configs
        
    configs = WeightPruningConfig(
        pruning_configs,
        target_sparsity=args.target_sparsity,
        pattern=args.pruning_pattern,
        pruning_frequency=frequency,
        start_step=pruning_start,
        end_step=pruning_end,
    )
    
    from neural_compressor.compression.pruner import prepare_pruning
    pruning = prepare_pruning(configs, model, dataloader=train_dataloader)
    
    model.eval()
    if args.evaluation_dataset_name != None:
        dataset_eval = load_dataset( # for example:use the_pile's validation set for retraining-free pruning, and lambada dataset for eval
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

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        output_dir = args.output_dir
        if args.auto_slim:
            output_dir += "/before_slim"
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of pruning", auto_lfs_prune=True)
    
    if not args.auto_slim:
        # only eval
        logger.info(f"***** Running Evaluation *****")
        acc, _ = eval_func(model)
        logger.info(f"total_steps:{args.max_pruning_steps} accuracy:{acc}")
    else:
        if 'bloom' not in model_name:
            logger.info(f"***** Running Evaluation before ffn auto slim*****")
            accuracy, avg_latency = eval_func(model)
            logger.info(f"accuracy:{accuracy}  avg_latency:{avg_latency}")
            model = model_slim(model, round_multiplier=32)

            logger.info(f"***** Running Evaluation after ffn auto_slim*****")
            accuracy, avg_latency = eval_func(model)
            logger.info(f"accuracy:{accuracy}  avg_latency:{avg_latency}")
            
            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                traced_model = trace_model(model, tokenizer)
                logger.info(f"Save silmed jit model")
                torch.jit.save(traced_model, args.output_dir+"/slimed_jit_model.pt")
        else:
            logger.info(f"Trace on BLOOM MODEL is not supported yet.")
            
    
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()

