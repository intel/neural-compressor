# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Union

import lm_eval
import numpy as np
from lm_eval import evaluator, utils
from lm_eval.loggers import WandbLogger
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table, simple_parse_args_string

DEFAULT_RESULTS_FILE = "results.json"


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def cli_evaluate(args) -> None:
    if args.wandb_args:
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(task_manager.all_tasks)))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks,"
                    + " or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file():
            raise FileExistsError(f"File already exists at {path}")
        output_path_file = path.joinpath(DEFAULT_RESULTS_FILE)
        if output_path_file.is_file():
            eval_logger.warning(f"File {output_path_file} already exists. Results will be overwritten.")
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)

    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if args.trust_remote_code:
        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = str(args.trust_remote_code)
        args.model_args = args.model_args + f",trust_remote_code={os.environ['HF_DATASETS_TRUST_REMOTE_CODE']}"

    eval_logger.info(f"Selected Tasks: {task_names}")
    eval_logger.info("Loading selected tasks...")

    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests)

    ### update model with user_model ###
    if args.model_args is None:
        args.model_args = ""
    # replace HFLM.
    from .models.huggingface import HFLM

    lm_eval.api.registry.MODEL_REGISTRY["hf-auto"] = HFLM
    lm_eval.api.registry.MODEL_REGISTRY["hf"] = HFLM
    lm_eval.api.registry.MODEL_REGISTRY["huggingface"] = HFLM

    if args.user_model is not None:
        # use tiny model to built lm.
        print(
            "We use 'pretrained=Muennighoff/tiny-random-bert'"
            + "to build `LM` instance, the actually run model is user_model you passed."
        )
        lm = lm_eval.api.registry.get_model(args.model).create_from_arg_string(
            "pretrained=Muennighoff/tiny-random-bert",
            {
                "batch_size": args.batch_size,
                "max_batch_size": args.max_batch_size,
                "device": args.device,
            },
        )
        lm._model = args.user_model
        if args.tokenizer is not None:
            lm.tokenizer = args.tokenizer
        else:
            assert False, "Please provide tokenizer in evaluation function"
    elif isinstance(args.model_args, dict):
        lm = lm_eval.api.registry.get_model(args.model).create_from_arg_obj(
            args.model_args,
            {
                "batch_size": args.batch_size,
                "max_batch_size": args.max_batch_size,
                "device": args.device,
            },
        )
    else:
        lm = lm_eval.api.registry.get_model(args.model).create_from_arg_string(
            args.model_args,
            {
                "batch_size": args.batch_size,
                "max_batch_size": args.max_batch_size,
                "device": args.device,
            },
        )
    lm.pad_to_buckets = args.pad_to_buckets
    lm.buckets = args.buckets

    results = evaluator.simple_evaluate(
        model=lm,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging
        if args.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if args.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        if args.output_path:
            output_path_file.open("w", encoding="utf-8").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(re.sub("/|=", "__", args.model_args), task_name)
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.write_text(samples_dumped, encoding="utf-8")

        print(
            f"{args.model} ({args.model_args}),"
            f" gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if args.wandb_args:
            # Tear down wandb run once all the logging is done.
            wandb_logger.run.finish()

    return results
