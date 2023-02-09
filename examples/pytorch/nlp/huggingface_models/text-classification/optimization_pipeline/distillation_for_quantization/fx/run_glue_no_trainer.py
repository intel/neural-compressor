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
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import shutil
import math
import os
import random
import datasets
from datasets import load_dataset, load_metric
# from neural_compressor import set_distillation_record
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm.auto import tqdm

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
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
    parser.add_argument('--use_auth_token', action='store_true', help="use authentic token")
    parser.add_argument("--resume", type=str, default=None, help="Where to resume from the provided model.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument('--do_eval', action='store_true',
                        help="evaluate model")
    parser.add_argument('--do_quantization', action='store_true',
                        help="do quantization aware training on model")
    parser.add_argument('--do_distillation', action='store_true',
                        help="do distillation with pre-trained teacher model")
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models"
             " to be the teacher model."
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def move_input_to_device(input, device):
    if isinstance(input, torch.Tensor):
        return input if input.device == device else input.to(device)
    elif isinstance(input, tuple):
        return tuple([move_input_to_device(ele, device) for ele in input])
    elif isinstance(input, list):
        return [move_input_to_device(ele, device) for ele in input]
    elif isinstance(input, dict):
        return {key:move_input_to_device(input[key], device) for key in input}
    else:
        assert False, "only support input type of torch.Tensor, tuple, list and dict."

def gather_results(predictions, gt):
    if rank != -1:
        pred_list = [predictions.clone() for _ in range(world)] if rank == 0 else []
        gt_list = [gt.clone() for _ in range(world)] if rank == 0 else []
        dist.gather(predictions, gather_list=pred_list)
        dist.gather(gt, gather_list=gt_list)
        return pred_list[0], gt_list[0]
    else:
        return predictions, gt

def evaluation(model, eval_dataloader, metric):
    logger.info("***** Running eval *****")
    logger.info(f"  Num examples = {len(eval_dataloader) }")
    model.eval()
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")
    model_device = next(model.parameters()).device
    for step, batch in enumerate(eval_dataloader):
        batch = move_input_to_device(batch, model_device)
        outputs = model(**batch)['logits']
        predictions = outputs.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )

    eval_metric = metric.compute()
    logger.info(f"eval_metric : {eval_metric}")
    return eval_metric['accuracy']

def save_checkpoint(state, is_best, save_dir):
    """Saves checkpoint to disk"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = save_dir + "/checkpoint.pth"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_dir + '/model_best.pth')

def train(args, model, train_dataloader, lr_scheduler, compression_manager, optimizer, eval_dataloader,
          metric):
    # Train!
    total_batch_size = args.batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    completed_steps = 0
    best_prec = 0

    model_device = next(model.parameters()).device
    for epoch in range(args.num_train_epochs):
        model.train()
        train_dataloader = tqdm(train_dataloader, desc="Training")
        compression_manager.callbacks.on_epoch_begin(epoch)
        for step, batch in enumerate(train_dataloader):
            batch = move_input_to_device(batch, model_device)
            compression_manager.callbacks.on_step_begin(step)
            outputs = model(**batch)
            loss = compression_manager.callbacks.on_after_compute_loss(batch, outputs['logits'], outputs['loss'])
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                compression_manager.callbacks.on_before_optimizer_step()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            compression_manager.callbacks.on_step_end()
            if completed_steps >= args.max_train_steps:
                break
        compression_manager.callbacks.on_epoch_end()
        best_score = evaluation(model, eval_dataloader, metric)
        is_best = best_score > best_prec
        best_prec = max(best_score, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
        }, is_best, args.output_dir)
    model.load_state_dict(torch.load(args.output_dir + "/model_best.pth")["state_dict"])


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=num_labels,
                                        finetuning_task=args.task_name,
                                        use_auth_token=args.use_auth_token)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=not args.use_slow_tokenizer,
                                              use_auth_token=args.use_auth_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config, use_auth_token=args.use_auth_token
    )
    if args.resume:
        try:
            model.load_state_dict(torch.load(args.resume))
            logger.info('Resumed model from {}'.format(args.resume))
        except:
            raise TypeError('Provided {} is not a valid checkpoint file, '
                            'please provide .pt file'.format(args.resume))
    model.to(device)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_seq_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size, drop_last=True)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)

    combs = []
    if args.do_distillation:
        from neural_compressor.config import DistillationConfig, \
                                             IntermediateLayersKnowledgeDistillationLossConfig
        layer_mappings = [
            [['bert.encoder.layer.0.output', ]],
            [['bert.encoder.layer.0.attention', '1']],
            [['bert.encoder.layer.1.output', ]],
            [['bert.encoder.layer.1.attention', '1']],
            [['bert.encoder.layer.2.output', ]],        
            [['bert.encoder.layer.2.attention', '1']],      
            [['bert.encoder.layer.3.output', ]],        
            [['bert.encoder.layer.3.attention', '1']],      
            [['bert.encoder.layer.4.output', ]],        
            [['bert.encoder.layer.4.attention', '1']],      
            [['bert.encoder.layer.5.output', ]],        
            [['bert.encoder.layer.5.attention', '1']],      
            [['bert.encoder.layer.6.output', ]],        
            [['bert.encoder.layer.6.attention', '1']],      
            [['bert.encoder.layer.7.output', ]],        
            [['bert.encoder.layer.7.attention', '1']],      
            [['bert.encoder.layer.8.output', ]],        
            [['bert.encoder.layer.8.attention', '1']],      
            [['bert.encoder.layer.9.output', ]],        
            [['bert.encoder.layer.9.attention', '1']],
            [['bert.encoder.layer.10.output', ]],
            [['bert.encoder.layer.10.attention', '1']],
            [['bert.encoder.layer.11.output', ]],
            [['bert.encoder.layer.11.attention', '1']],
            [['classifier', ]],
        ]
        loss_weights = [1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1]
        distillation_criterion = IntermediateLayersKnowledgeDistillationLossConfig(
            layer_mappings=layer_mappings,
            loss_weights=loss_weights)

        teacher_config = AutoConfig.from_pretrained(args.teacher_model_name_or_path, \
                            num_labels=num_labels, finetuning_task=args.task_name)
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path, \
                            use_fast=not args.use_slow_tokenizer)
        assert teacher_tokenizer.vocab == tokenizer.vocab, \
                'teacher model and student model should have same tokenizer.'
        teacher_model = AutoModelForSequenceClassification.from_pretrained(
            args.teacher_model_name_or_path,
            from_tf=bool(".ckpt" in args.teacher_model_name_or_path),
            config=teacher_config,
        )
        teacher_model.to(device)
        para_counter = lambda model:sum(p.numel() for p in model.parameters())
        logger.info("***** Number of teacher model parameters: {:.2f}M *****".format(\
                    para_counter(teacher_model)/10**6))
        logger.info("***** Number of student model parameters: {:.2f}M *****".format(\
                    para_counter(model)/10**6))

        # output attentions of models for intermediate layers distillation
        model.config.output_attentions = True
        teacher_model.config.output_attentions = True

        d_conf = DistillationConfig(teacher_model=teacher_model, criterion=distillation_criterion)
        combs.append(d_conf)

    if args.do_quantization:
        from neural_compressor import QuantizationAwareTrainingConfig
        q_conf = QuantizationAwareTrainingConfig()
        combs.append(q_conf)

    if len(combs) == 0:
        assert False, "Please set at least one of do_distillation and do_quantization."

    from neural_compressor.training import prepare_compression
    compression_manager = prepare_compression(model, combs)
    compression_manager.callbacks.on_train_begin()
    model = compression_manager.model
    train(args, model, train_dataloader, lr_scheduler, compression_manager, optimizer, eval_dataloader,
          metric)
    compression_manager.callbacks.on_train_end()

    model = model.model

    if args.do_eval:
        eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")
        model.eval()
        model_device = next(model.parameters()).device
        for step, batch in enumerate(eval_dataloader):
            batch = move_input_to_device(batch, model_device)
            outputs = model(**batch)
            predictions = outputs['logits'].argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        logger.info(f"eval_metric: {eval_metric}")

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.batch_size, drop_last=True
        )
        model.eval()
        model_device = next(model.parameters()).device
        for step, batch in enumerate(eval_dataloader):
            batch = move_input_to_device(batch, model_device)
            outputs = model(**batch)
            predictions = outputs['logits'].argmax(dim=-1)
            pred, gt = gather_results(predictions, batch["labels"])
            metric.add_batch(predictions=pred, references=gt)

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

if __name__ == "__main__":
    rank, world = int(os.environ.get('PMI_RANK', -1)), int(os.environ.get('PMI_SIZE', 1))
    if rank != -1:
        logger.warning('start distributed training...')
        dist.init_process_group(backend='mpi')
    main()
