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
Fine-tuning a 🤗 Transformers model on question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
# from transformers.utils import check_min_version
# from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.8.0.dev0")
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
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
        "--preprocessing_num_workers", type=int, default=4, help="A csv or a json file containing the training data."
    )
    parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="The threshold used to select the null answer: if the best answer has a score that is less than "
        "the score of the null answer minus this threshold, the null answer is selected for this example. "
        "Only useful when `version_2_with_negative=True`.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        type=bool,
        default=False,
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this "
        "value if set.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument('--use_auth_token', action='store_true', help="use authentic token")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resume", type=str, default=None, help="Where to resume from the provided model.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument('--do_train', action='store_true',
                        help="fine-tune model")
    parser.add_argument('--do_prune', action='store_true',
                        help="prune model")
    parser.add_argument('--do_eval', action='store_true',
                        help="evaluate model")
    parser.add_argument('--do_quantization', action='store_true',
                        help="do quantization aware training on model")
    parser.add_argument('--do_distillation', action='store_true',
                        help="do distillation with pre-trained teacher model")
    parser.add_argument('--run_teacher_logits', action='store_true',
                        help="do evaluation on training data with teacher model "
                        "to accelerate distillation.")
    parser.add_argument("--prune_config", default='prune.yaml', help="pruning config")
    parser.add_argument("--quantization_config", default='qat.yaml', help="quantization config")
    parser.add_argument("--distillation_config", default='distillation.yaml', help="pruning config")
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        help="Path to pretrained teacher model or it's identifier from huggingface.co/models.",
    )

    parser.add_argument("--temperature", default=1, type=float,
                        help='temperature parameter of distillation')
    parser.add_argument("--loss_types", default=['CE', 'KL'], type=str, nargs='+',
                        help='loss types of distillation, should be a list of length 2, '
                        'first for student targets loss, second for teacher student loss.')
    parser.add_argument("--loss_weights", default=[0.5, 0.5], type=float, nargs='+',
                        help='loss weights of distillation, should be a list of length 2, '
                        'and sum to 1.0, first for student targets loss weight, '
                        'second for teacher student loss weight.')
    args = parser.parse_args()

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
        and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

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

def train(args, model, train_dataloader, lr_scheduler, criterion, optimizer, \
          agent, accelerator, eval_dataloader, metric):
    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0

    if agent:
        agent.on_train_begin()
        model = agent.model.model
    for epoch in range(args.num_train_epochs):
        model.train()
        train_dataloader = tqdm(train_dataloader, desc="Training")
        if agent:
            agent.on_epoch_begin(epoch)
        for step, batch in enumerate(train_dataloader):
            if agent:
                agent.on_step_begin(step)
            teacher_logits = None
            if 'teacher_logits' in batch:
                teacher_logits = torch.vstack(list(batch['teacher_logits']))
                del batch['teacher_logits']
            outputs = model(**batch)
            outputs_for_kd = torch.vstack([torch.vstack([sx, ex]) \
                for sx, ex in zip(outputs['start_logits'], outputs['end_logits'])])
            labels = torch.hstack([torch.tensor([sx, ex]).to(outputs_for_kd.device) \
                for sx, ex in zip(batch["start_positions"], batch["end_positions"])])
            if criterion is None:
                loss = outputs['loss']
            else:
                if teacher_logits is not None:
                    criterion.teacher_outputs = teacher_logits
                else:
                    criterion.teacher_model_forward(batch)
                loss = criterion(outputs_for_kd, labels)
                loss = agent.on_after_compute_loss(batch, outputs, loss, teacher_logits)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if agent:
                    agent.on_before_optimizer_step()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if agent:
                agent.on_step_end()
            if completed_steps >= args.max_train_steps:
                break
        if agent:
            agent.on_epoch_end()
        evaluation(args, model, accelerator, eval_dataloader, metric)
    if agent:
        agent.on_train_end()

# Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat

# Post-processing:
def post_processing_function(args, examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=args.version_2_with_negative,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=args.null_score_diff_threshold,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def evaluation(args, model, accelerator, eval_dataloader, metric):
    # Evaluation
    logger.info("***** Running Evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.batch_size}")

    all_start_logits = []
    all_end_logits = []
    model_device = next(model.parameters()).device
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            # due to fx model must take 'start_positions' and 'end_positions' as input
            fake_input = torch.zeros(batch['input_ids'].shape[0], dtype=torch.int64)
            batch['start_positions'] = fake_input
            batch['end_positions'] = fake_input
            batch = move_input_to_device(batch, model_device)
            outputs = model(**batch)
            if torch.is_tensor(outputs):
                start_logits = torch.vstack([x for x in outputs[0::2]])
                end_logits = torch.vstack([x for x in outputs[1::2]])
            else:
                start_logits = outputs['start_logits']
                end_logits = outputs['end_logits']

            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(args, eval_examples, eval_dataset, outputs_numpy)
    eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    logger.info(f"Evaluation metrics: {eval_metric}")
    return eval_metric['exact_match']

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, use_auth_token=args.use_auth_token)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, use_auth_token=args.use_auth_token)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True, use_auth_token=args.use_auth_token)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True, use_auth_token=args.use_auth_token)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            use_auth_token=args.use_auth_token
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config)

    if args.resume:
        try:
            model.load_state_dict(torch.load(args.resume))
            logger.info('Resumed model from {}'.format(args.resume))
        except:
            raise TypeError('Provided {} is not a valid checkpoint file, '
                            'please provide .pt file'.format(args.resume))

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.

    column_names = raw_datasets["train"].column_names

    global answer_column_name
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if args.max_train_samples is not None:
        # We will select sample from whole data if agument is specified
        train_dataset = train_dataset.select(range(args.max_train_samples))
    # Create train feature from dataset
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )
    if args.max_train_samples is not None:
        # Number of samples might increase during Feature Creation, We select only specified max samples
        train_dataset = train_dataset.select(range(args.max_train_samples))
     
    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    global eval_examples, eval_dataset
    eval_examples = raw_datasets["validation"]
    if args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(args.max_eval_samples))
    
    # fx model must take input with predefined shape, evaluation of QA model 
    # need lengthes of dataset and dataloader be the same, 
    # so here to make length of eval_examples to multiples of batch_size. 
    eval_examples = eval_examples.select(range((len(eval_examples) // args.batch_size) * args.batch_size))

    # Validation Feature Creation
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    if args.max_eval_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    # fx model must take input with predefined shape, evaluation of QA model 
    # need lengthes of dataset and dataloader be the same, 
    # so here to make length of eval_dataset to multiples of batch_size. 
    eval_dataset = eval_dataset.select(range((len(eval_dataset) // args.batch_size) * args.batch_size))

    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(args.max_predict_samples))

        # fx model must take input with predefined shape, evaluation of QA model 
        # need lengthes of dataset and dataloader be the same, 
        # so here to make length of predict_examples to multiples of batch_size. 
        predict_examples = predict_examples.select(range((len(predict_examples) // args.batch_size) * args.batch_size))

        # Predict Feature Creation
        predict_dataset = predict_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )
        if args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            predict_dataset = predict_dataset.select(range(args.max_predict_samples))

        # fx model must take input with predefined shape, evaluation of QA model 
        # need lengthes of dataset and dataloader be the same, 
        # so here to make length of predict_dataset to multiples of batch_size. 
        predict_dataset = predict_dataset.select(range((len(predict_dataset) // args.batch_size) * args.batch_size))

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
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    if args.do_distillation:
        if args.teacher_model_name_or_path:
            config = AutoConfig.from_pretrained(args.teacher_model_name_or_path, use_auth_token=args.use_auth_token)
            teacher_model = AutoModelForQuestionAnswering.from_pretrained(
                args.teacher_model_name_or_path,
                from_tf=bool(".ckpt" in args.teacher_model_name_or_path),
                config=config,
                use_auth_token=args.use_auth_token
            )
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                                    args.teacher_model_name_or_path, use_fast=True, use_auth_token=args.use_auth_token)
            assert teacher_tokenizer.vocab == tokenizer.vocab, \
                'teacher model and student model should have same tokenizer.'
        else:
            raise ValueError("Please provide a teacher model for distillation.")

        class QAModel_output_reshaped(torch.nn.Module):
            def __init__(self, model):
                super(QAModel_output_reshaped, self).__init__()
                self.model = model

            def forward(self, *args, **kwargs):
                outputs = self.model(*args, **kwargs)
                outputs_reshaped = torch.vstack([torch.vstack([sx, ex]) \
                        for sx, ex in zip(outputs['start_logits'], outputs['end_logits'])])
                return outputs_reshaped
        
        teacher_model = QAModel_output_reshaped(teacher_model)
        teacher_model = accelerator.prepare(teacher_model)
        para_counter = lambda model:sum(p.numel() for p in model.parameters())
        logger.info("***** Number of teacher model parameters: {:.2f}M *****".format(\
                    para_counter(teacher_model)/10**6))
        logger.info("***** Number of student model parameters: {:.2f}M *****".format(\
                    para_counter(model)/10**6))

        # get logits of teacher model
        if args.run_teacher_logits and args.loss_weights[1] > 0:
            assert args.pad_to_max_length, 'to run teacher logits must open pad_to_max_length due to padding issue'
            def get_logits(teacher_model, train_dataset):
                logger.info("***** Getting logits of teacher model *****")
                logger.info(f"  Num examples = {len(train_dataset) }")
                teacher_model.eval()
                npy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '{}.{}.npy'.format(args.dataset_name, args.teacher_model_name_or_path.replace('/', '.')))
                if os.path.exists(npy_file):
                    teacher_logits = [list(x) for x in np.load(npy_file, allow_pickle=True)]
                else:
                    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, batch_size=args.batch_size)
                    train_dataloader = accelerator.prepare(train_dataloader)
                    train_dataloader = tqdm(train_dataloader, desc="Evaluating")
                    teacher_logits = []
                    for step, batch in enumerate(train_dataloader):
                        outputs = teacher_model(**batch).cpu().detach().numpy()
                        teacher_logits += [[s,e] for s,e in zip(outputs[0::2], outputs[1::2])]
                    np.save(npy_file, teacher_logits, allow_pickle=True)
                return train_dataset.add_column('teacher_logits', teacher_logits)
            with torch.no_grad():
                train_dataset = get_logits(teacher_model, train_dataset)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size, drop_last=True
    )

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.batch_size, drop_last=True
    )

    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(
            predict_dataset_for_model, collate_fn=data_collator, batch_size=args.batch_size, drop_last=True
        )

    metric = load_metric("squad_v2" if args.version_2_with_negative else "squad")

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

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

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
    
    def train_func(model):
        return train(args, model, train_dataloader, lr_scheduler, criterion, \
                     optimizer, agent, accelerator, eval_dataloader, metric)

    def eval_func(model):
        return evaluation(args, model, accelerator, eval_dataloader, metric)

    if args.do_prune:
        # Pruning!
        from neural_compressor.experimental import Pruning, common
        agent = Pruning(args.prune_config)
        criterion = None # use huggingface's loss
        if args.do_distillation:
            logger.info('='*30 + 'Teacher model on validation set' + '='*30)
            evaluation(args, teacher_model, accelerator, eval_dataloader, metric)  

            # from neural_compressor.experimental import Distillation
            from neural_compressor.experimental.common.criterion import PyTorchKnowledgeDistillationLoss
            criterion = PyTorchKnowledgeDistillationLoss(
                                    temperature=args.temperature,
                                    loss_types=args.loss_types,
                                    loss_weights=args.loss_weights)
            criterion.teacher_model = teacher_model
            
        if args.do_quantization:
            # transforming the student model to fx mode for QAT
            from transformers.utils.fx import symbolic_trace
            for input in train_dataloader:
                input_names = list(input.keys())
                if 'teacher_logits' in input_names:
                    input_names.remove('teacher_logits')
                break
            model = symbolic_trace(model, input_names=input_names, \
                                   batch_size=args.batch_size, \
                                   sequence_length=args.max_seq_length)
                                   
            from neural_compressor.experimental.scheduler import Scheduler
            from neural_compressor.experimental import Quantization
            combs = [agent, Quantization(args.quantization_config)]
            scheduler = Scheduler()                         
            scheduler.model = common.Model(model)
            agent = scheduler.combine(*combs)
            agent.train_func = train_func
            agent.eval_func = eval_func
            print(agent)
            scheduler.append(agent)
            model = scheduler.fit()
        else:
            agent.model = common.Model(model)
            agent.pruning_func = train_func
            agent.eval_func = eval_func
            model = agent()
        model.save(args.output_dir)
        # change to framework model for further use
        model = model.model
        
    # Prediction
    if args.do_predict:
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(predict_dataset)}")
        logger.info(f"  Batch size = {args.batch_size}")

        all_start_logits = []
        all_end_logits = []
        model_device = next(model.parameters()).device
        for step, batch in enumerate(predict_dataloader):
            with torch.no_grad():
                # due to fx model must take 'start_positions' and 'end_positions' as input
                fake_input = torch.zeros(batch['input_ids'].shape[0], dtype=torch.int64)
                batch['start_positions'] = fake_input
                batch['end_positions'] = fake_input
                batch = move_input_to_device(batch, model_device)
                outputs = model(**batch)
                start_logits = outputs['start_logits']
                end_logits = outputs['end_logits']

                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
        predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f"Predict metrics: {predict_metric}")

    # if args.output_dir is not None:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
