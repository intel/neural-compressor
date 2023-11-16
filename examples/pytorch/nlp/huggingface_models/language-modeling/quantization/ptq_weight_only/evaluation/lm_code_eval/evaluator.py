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

import os
import random
import re
import time
import numpy as np
import json
import fnmatch
from accelerate import Accelerator
import torch
import lm_eval # pylint: disable=E0611, E0401
from lm_eval.arguments import EvalArguments # pylint: disable=E0611, E0401
from lm_eval.evaluator import Evaluator # pylint: disable=E0611, E0401
from lm_eval.tasks import ALL_TASKS # pylint: disable=E0611, E0401


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def evaluate(model,
             tokenizer,
             tasks,
             batch_size,
             args,
            ):
    """Instantiate and evaluate a model on a list of tasks.

    """
    try:
        import transformers
        import  datasets
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

    tokenizer.pad_token = tokenizer.eos_token
    evaluator = Evaluator(accelerator, model, tokenizer, args)

    for task in task_names:
        if args.generation_only:
            if accelerator.is_main_process:
                print("generation mode only")
            generations, references = evaluator.generate_text(task)
            if accelerator.is_main_process:
                with open(args.save_generations_path, "w") as fp:
                    json.dump(generations, fp)
                    print(f"generations were saved at {args.save_generations_path}")
                if args.save_references:
                    with open("references.json", "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")
        else:
            results[task] = evaluator.evaluate(task)

    results["config"] = {
        "model": model.config.model_type,
        "temperature": args.temperature,
        "n_samples": args.n_samples,
    }
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if accelerator.is_main_process:
            print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)
    
    return results
