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
from neural_compressor.utils.logger import log
import math
import os
import random
import copy
import datasets
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import numpy as np

import transformers
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
        default=64,
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
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
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
    parser.add_argument('--do_train', action='store_true',
                        help="train model")
    parser.add_argument('--do_eval', action='store_true',
                        help="evaluate model")
    parser.add_argument('--do_distillation', action='store_true',
                        help="do distillation with pre-trained model on Bi-LSTM.")
    parser.add_argument("--config", default='distillation.yaml', help="pruning config")
    parser.add_argument("--core_per_instance", type=int, default=-1, help="cores per instance.")
    
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

def take_eval_steps(model, eval_dataloader, metric):
    logger.info("***** Running eval *****")
    logger.info(f"  Num examples = {len(eval_dataloader) }")
    model.eval()
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        outputs = model(**batch)
        predictions = outputs.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )

    eval_metric = metric.compute()
    logger.info(f"eval_metric : {eval_metric}")
    return eval_metric['accuracy']

def take_train_steps(args, model, train_dataloader, lr_scheduler, distiller):
    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running distillation *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    completed_steps = 0

    distiller.pre_epoch_begin()
    for epoch in range(args.num_train_epochs):
        model.train()
        train_dataloader = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(train_dataloader):
            teacher_logits = None
            if 'teacher_logits' in batch:
                teacher_logits = batch['teacher_logits']
                del batch['teacher_logits']
            outputs = model(**batch)
            distiller.on_post_forward(batch, teacher_logits)
            loss = distiller.criterion(outputs, batch["labels"])
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                distiller.optimizer.step()
                lr_scheduler.step()
                distiller.optimizer.zero_grad()
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        distiller.on_epoch_end()

def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

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
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

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

    if args.loss_weights[1] > 0:
        augmented_sst2_dataset = load_dataset("jmamou/augmented-glue-sst2")
        index_start = len(raw_datasets['train'])
        augmented_sst2_dataset = augmented_sst2_dataset['train'].add_column('idx', \
            np.array(range(index_start, index_start+len(augmented_sst2_dataset['train'])), dtype=np.int32)).remove_columns('prediction')
        raw_datasets['train'] = datasets.concatenate_datasets([raw_datasets['train'], augmented_sst2_dataset])
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

    if args.do_distillation:
        embedding_dim = 50
        def load_glove_embeddings(vocab, embedding_dim=embedding_dim):
            # load Glove 
            # define dict to hold a word and its vector
            glove = {}
            # read the word embeddings file ~820MB
            with open(os.path.join(os.path.dirname(__file__), 'glove.6B.50d.txt'), \
                      encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    glove[word] = coefs
            embedding_matrix = np.zeros((len(vocab), embedding_dim), dtype=np.float32)
            for word, i in vocab.items():
                embedding_vector = glove.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
            return torch.tensor(embedding_matrix)
        class BidirectionalLSTM(torch.nn.Module):
            def __init__(self, vocab, hidden_dim, num_layers, \
                         output_dim, dropout=0.2, embedding_dim=embedding_dim):
                super(BidirectionalLSTM, self).__init__()
                self.embedding = torch.nn.Embedding.from_pretrained(
                                     load_glove_embeddings(vocab, embedding_dim), freeze=False)
                self.lstm = torch.nn.LSTM(input_size=embedding_dim,
                                          hidden_size=hidden_dim,
                                          num_layers=num_layers,
                                          dropout=dropout,
                                          bidirectional=True,
                                          batch_first=True)
                self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        
            def forward(self, attention_mask, input_ids, labels):
                embedding_output = self.embedding(input_ids)
                lstm_output, states = self.lstm(embedding_output)
                hidden_states, cell_states = states
                output = torch.reshape(hidden_states[-2:,].permute(1, 0, 2), 
                                       [input_ids.shape[0], -1])
                output = self.fc(output)
                return output

        class BertModelforLogitsOutputOnly(torch.nn.Module):
            def __init__(self, model):
                super(BertModelforLogitsOutputOnly, self).__init__()
                self.model = model
            def forward(self, *args, **kwargs):
                output = self.model(*args, **kwargs)
                return output.logits

        para_counter = lambda model:sum(p.numel() for p in model.parameters())
        teacher_model = BertModelforLogitsOutputOnly(model)
        logger.info("***** Number of teacher model parameters: {:.2f}M *****".format(\
                    para_counter(model)/10**6))
        model = BidirectionalLSTM(vocab=tokenizer.vocab, hidden_dim=64, \
                                  num_layers=2, output_dim=num_labels)
        logger.info("***** Number of student model parameters: {:.2f}M *****".format(\
                    para_counter(model)/10**6))

        # get logits of teacher model
        if args.loss_weights[1] > 0:
            def get_logits(teacher_model, train_dataset):
                logger.info("***** Getting logits of teacher model *****")
                logger.info(f"  Num examples = {len(train_dataset) }")
                teacher_model.eval()
                npy_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '{}+augmented-glue-sst2.{}.npy'.format(args.task_name, args.model_name_or_path.replace('/', '.')))
                if os.path.exists(npy_file):
                    teacher_logits = [x for x in np.load(npy_file)]
                else:
                    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, \
                                           batch_size=args.per_device_eval_batch_size)
                    train_dataloader = tqdm(train_dataloader, desc="Evaluating")
                    teacher_logits = []
                    for step, batch in enumerate(train_dataloader):
                        outputs = teacher_model(**batch)
                        teacher_logits += [x for x in outputs.numpy()]
                    np.save(npy_file, np.array(teacher_logits))
                return train_dataset.add_column('teacher_logits', teacher_logits)
            with torch.no_grad():
                train_dataset = get_logits(teacher_model, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
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

    # Do distillation
    if args.do_distillation:
        from neural_compressor.experimental import Distillation, common
        from neural_compressor.experimental.common.criterion import PyTorchKnowledgeDistillationLoss
        def train_func(model):
            return take_train_steps(args, model, train_dataloader, lr_scheduler, distiller)

        def eval_func(model):
            return take_eval_steps(model, eval_dataloader, metric)

        # eval datasets.
        distiller = Distillation(args.config)
        distiller.teacher_model = common.Model(teacher_model)
        distiller.student_model = common.Model(model)
        distiller.train_func = train_func
        distiller.eval_func = eval_func
        distiller.optimizer = optimizer
        distiller.criterion = PyTorchKnowledgeDistillationLoss(
                                temperature=args.temperature,
                                loss_types=args.loss_types,
                                loss_weights=args.loss_weights)
        model = distiller()
        model.save(args.output_dir)
        # change to framework model for further use
        model = model.model

    if args.do_train:
        # Train!
        total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps))
        completed_steps = 0

        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break
        # required when actual steps and max train steps doesn't match
        progress_bar.close()
        torch.save(model.state_dict(), args.output_dir + '/retrained_model.pth')

    if args.do_eval:
        eval_dataloader = tqdm(eval_dataloader, desc="Evaluating")
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        logger.info(f"eval_metric: {eval_metric}")

if __name__ == "__main__":
    main()
