#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import json
import os
import random
import re
import time

import bigcode_eval  # pylint: disable=E0611, E0401
import numpy as np
import torch
from accelerate import Accelerator
from bigcode_eval.arguments import EvalArguments  # pylint: disable=E0611, E0401
from bigcode_eval.evaluator import Evaluator  # pylint: disable=E0611, E0401
from bigcode_eval.tasks import ALL_TASKS  # pylint: disable=E0611, E0401


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns."""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def evaluate(
    model,
    tokenizer,
    tasks,
    batch_size,
    args,
):
    """Instantiate and evaluate a model on a list of tasks."""
    try:
        import datasets
        import transformers

        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()
    except:
        pass
    if tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(tasks.split(","), ALL_TASKS)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}

    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")

    try:
        tokenizer.pad_token = tokenizer.eos_token

    # Some models like CodeGeeX2 have pad_token as a read-only property
    except AttributeError:
        print("Not setting pad_token to eos_token")
        pass

    evaluator = Evaluator(accelerator, model, tokenizer, args)

    if args.load_generations_intermediate_paths and len(args.load_generations_intermediate_paths) != len(task_names):
        raise ValueError(
            "If passing --load_generations_intermediate_paths, \
            must pass equal number of files as number of tasks"
        )

    for idx, task in enumerate(task_names):
        intermediate_generations = None
        if args.load_generations_intermediate_paths:
            with open(args.load_generations_intermediate_paths[idx], "r") as f_in:
                # intermediate_generations: list[list[str | None]] of len n_tasks
                # where list[i] = generated codes or empty
                intermediate_generations = json.load(f_in)

        if args.generation_only:
            if accelerator.is_main_process:
                print("generation mode only")
            generations, references = evaluator.generate_text(task, intermediate_generations=intermediate_generations)
            if accelerator.is_main_process:
                save_generations_path = f"{os.path.splitext(args.save_generations_path)[0]}_{task}.json"
                save_references_path = f"references_{task}.json"
                evaluator.save_json_files(
                    generations,
                    references,
                    save_generations_path,
                    save_references_path,
                )
        else:
            results[task] = evaluator.evaluate(task, intermediate_generations=intermediate_generations)

    # Save all args to config
    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if accelerator.is_main_process:
            print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)

    return results
