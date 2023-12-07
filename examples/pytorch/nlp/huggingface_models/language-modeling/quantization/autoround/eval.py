import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path
import torch
import torch.nn as nn
from parse_results import result_parser
import pprint
import json
import shutil
import transformers
import time

EXT_TASKS = ['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new']
fewshots_dict = {}
fewshots_dict['paper'] = {
    "lambada_openai": [0],
    "hellaswag": [0],
    "winogrande": [0],
    "piqa": [0],
    "hendrycksTest-*": [0],
    "wikitext": [0],
    "truthfulqa_mc": [0],
    "openbookqa": [0],
    "boolq": [0],
    "rte": [0],
    "arc_easy": [0],
    "arc_challenge": [0],
}
fewshots_dict['leadboard'] = {
    "hellaswag": [10],
    "winogrande": [5],
    "arc_easy": [25],
    "arc_challenge": [25],
    "hendrycksTest-*": [5],
    "drop": [3],
    "gsm8k": [5],
}
fewshots_dict['all'] = {
    "lambada_openai": [0],
    "hellaswag": [0, 10],
    "winogrande": [0, 5],
    "piqa": [0],
    "coqa": [],  ## coqa is not enabled in llamav1 models
    "truthfulqa_mc": [0],
    "openbookqa": [0],
    "boolq": [0],
    "rte": [0],
    "arc_easy": [0, 25],
    "arc_challenge": [0, 25],
    "hendrycksTest-*": [0, 5],
    "wikitext": [0],
    "drop": [3],
    "gsm8k": [5]
}


def simple_evaluate(
        model,
        model_args=None,
        tasks=[],
        num_fewshot=0,
        batch_size=None,
        max_batch_size=None,
        device=None,
        no_cache=True,  ##changed by wenhua
        limit=None,
        bootstrap_iters=100000,
        description_dict=None,
        check_integrity=False,
        decontamination_ngrams_path=None,
        write_out=False,
        output_base_path=None,
        lm=None  ##changed by wenhua
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model, transformers.PreTrainedModel object, or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write details about prompts and logits to json for all tasks
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir.
    :return
        Dictionary of results
    """
    import collections
    import itertools
    import random

    import lm_eval.metrics
    import lm_eval.models
    import lm_eval.tasks
    import lm_eval.base
    from lm_eval.utils import positional_deprecated, run_task_tests
    from lm_eval.models.gpt2 import HFLM

    import numpy as np
    import transformers
    from lm_eval.evaluator import evaluate
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"
    if lm == None:
        if isinstance(model, str):
            if model_args is None:
                model_args = ""
            lm = lm_eval.models.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
        elif isinstance(model, transformers.PreTrainedModel):
            lm = lm_eval.models.get_model("hf-causal")(
                pretrained=model,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
            )
            no_cache = True
        else:
            assert isinstance(model, lm_eval.base.LM)
            lm = model

        if not no_cache:
            lm = lm_eval.base.CachingLM(
                lm,
                "lm_cache/"
                + (model if isinstance(model, str) else model.model.config._name_or_path)
                + "_"
                + model_args.replace("=", "-").replace(",", "_").replace("/", "-")
                + ".db",
            )

    # if isinstance(lm.tokenizer, transformers.LlamaTokenizerFast):
    #     if lm.tokenizer.pad_token is None:
    #         lm.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     else:
    #         lm.tokenizer.pad_token = '[PAD]'

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        output_base_path=output_base_path,
    )

    # add info about the model and few shot config
    model_name = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, transformers.PreTrainedModel):
        model_name = "pretrained=" + model.config._name_or_path
    results["config"] = {
        "model": model_name,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "batch_sizes": list(lm.batch_sizes.values())
        if hasattr(lm, "batch_sizes")
        else [],
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "description_dict": description_dict,
    }

    return results, lm


def eval_model(output_dir=None, model=None, tokenizer=None,
               tasks=["lambada_openai", "hellaswag", "winogrande", "piqa"],
               eval_bs=32, use_accelerate=True, dtype="float16", limit=None,
               device="cuda:0", seed=0, nsamples=128, eval_orig_float=False, mark="paper", excel_file="tmp.xlsx"):
    print("evaluation with official lm-eval", flush=True)
    try:
        import lm_eval
        from lm_eval import evaluator
        from lm_eval.tasks import ALL_TASKS, get_task_dict
    except:
        raise ImportError("""follow requirements to install dependencies.""")

    org_s = time.time()
    ##save model
    if output_dir is None:
        output_dir = "./tmp_signround"

    if os.path.exists(output_dir) and not eval_orig_float:
        shutil.rmtree(output_dir)
    if not eval_orig_float:
        model = model.to(torch.float16)
        model = model.to("cpu")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    if (hasattr(model, 'config') and model.config.torch_dtype is torch.bfloat16):
        dtype = 'bfloat16'

    external_tasks = []
    for each in EXT_TASKS:
        if each in tasks:
            external_tasks.append(each)
            tasks.remove(each)
    #
    # lm = lm_eval.models.get_model("hf-causal-experimental").create_from_arg_string(
    #         model_args,
    #         {
    #             "batch_size": eval_bs,
    #             "max_batch_size": eval_bs,
    #             "device": device}
    #         )

    results = {}
    model = None
    lm = None
    for tmp_tasks in tasks:
        try:
            num_fewshot = fewshots_dict[mark][tmp_tasks]
            task_names = lm_eval.utils.pattern_match([tmp_tasks], ALL_TASKS)
            # task_dict = get_task_dict(task_names)

            # for lm-eval internal tasks
            print(f'********* {tmp_tasks} evaluate ************')
            task_s = time.time()
            for shot in num_fewshot:
                # tmp_results = evaluator.evaluate(
                #         lm=lm,
                #         task_dict=task_dict,
                #         num_fewshot=shot,
                #         limit=limit,
                #         bootstrap_iters=100000,
                #         description_dict=None,
                #         decontamination_ngrams_path=None,
                #         write_out=False,
                #         output_base_path=None,
                #         )
                # tmp_results, model = simple_evaluate(model="hf-causal", model_args=model_args, tasks=task_names,
                #                                      num_fewshot=shot, limit=limit,batch_size=eval_bs,max_batch_size=eval_bs)

                model_args = f'pretrained={output_dir},tokenizer="{output_dir}",dtype={dtype},use_accelerate={use_accelerate},trust_remote_code=True'
                model_type = "hf-causal-experimental"
                # else:
                #     model_args = f'pretrained={output_dir},tokenizer="{output_dir}",dtype={dtype}'
                #     model_type = "hf-causal"

                if "wikitext" in task_names:
                    tmp_eval_bs = 1
                else:
                    tmp_eval_bs = eval_bs

                tmp_results, lm = simple_evaluate(model=model_type, model_args=model_args, tasks=task_names,
                                                  num_fewshot=shot, limit=limit, batch_size=tmp_eval_bs,
                                                  max_batch_size=tmp_eval_bs, lm=lm)

                sub_name = f'{tmp_tasks} {shot}-shot'
                print(f'{sub_name}: ')
                pprint.pprint(tmp_results["results"])
                print(f"\n{sub_name} cost time: {time.time() - task_s}\n")
                results[sub_name] = tmp_results
        except Exception as e:
            print(f'********* {tmp_tasks} ERROR ************')
            print(str(e))
            continue

    # if isinstance(lm.tokenizer, transformers.LlamaTokenizerFast):
    #     lm.tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir, use_fast=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir, use_fast=False, trust_remote_code=True)
    model = lm.model
    # for external tasks
    # maybe adjust for specific model
    # if hasattr(lm.model.config, "max_position_embeddings"):
    #     lm.model.seqlen = lm.model.config.max_position_embeddings
    # else:
    #     ## for llama-1, opt
    #     lm.model.seqlen = 2048

    # if "opt" in model_name:
    #     seqlen = model.config.max_position_embeddings
    #     model.seqlen = model.config.max_position_embeddings
    # else:
    #     seqlen = 2048
    #     model.seqlen = seqlen

    model.seqlen = 2048
    from utils import get_loaders, eval_ppl_same_with_gptq
    for dataset in external_tasks:
        try:
            dataloader, testloader = get_loaders(
                dataset, nsamples=nsamples, seed=seed,
                tokenizer=tokenizer, seqlen=model.seqlen
            )
            ppl = eval_ppl_same_with_gptq(model, testloader, device)
            print(dataset, ppl)

            results.update({dataset: ppl})
        except Exception as e:
            print(str(e))
            continue

    print(results, flush=True)
    new_results = result_parser(results)
    print("cost time: ", time.time() - org_s)
    import pickle
    from collections import OrderedDict
    new_dict = OrderedDict()
    new_dict["model"] = "tmp"
    new_dict["paper-avg"] = 0
    new_dict["leaderboard-avg"] = 0
    new_dict["wikitext2"] = 0
    new_dict["ptb-new"] = 0
    new_dict["c4-new"] = 0
    new_dict["wikitext 0-shot_word_perplexity"] = 0
    new_dict["wikitext 0-shot_byte_perplexity"] = 0
    new_dict["wikitext 0-shot_bits_per_byte"] = 0
    for key in new_results.keys():
        if key == "model" or key == "paper-avg" or key == "leaderboard-avg":
            continue
        data = new_results[key]
        if not isinstance(data, dict):
            new_dict[key] = new_results[key]
            continue
        for sub_key in data.keys():
            for sub_sub_key in data[sub_key].keys():
                if "hendry" in key:
                    new_key = key + "-" + sub_key + "-" + sub_sub_key
                else:
                    new_key = key + "_" + sub_sub_key
                if "std" in new_key:
                    continue
                # if "norm" in new_key:
                #     continue
                new_dict[new_key] = data[sub_key][sub_sub_key]

    import pandas as pd

    df = pd.DataFrame(data=new_dict, index=[0])

    df.to_excel(excel_file)

    # if output_dir == "./tmp_signround":
    #     shutil.rmtree(output_dir)


if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="/models/opt-125m"
    )
    parser.add_argument(
        "--bs", default=1,
    )

    args = parser.parse_args()
    s = time.time()
    # 'wikitext2', 'ptb-new', 'c4-new', 'lambada_openai',
    #               'hellaswag', 'winogrande', 'piqa', 'coqa', 'drop', 'gsm8k','truthfulqa_mc',
    # "lambada_openai": [0],
    # "hellaswag": [0],
    # "winogrande": [0],
    # "piqa": [0],
    # "hendrycksTest-*": [0],
    # "wikitext": [0],
    # "truthfulqa_mc": [0],
    # "openbookqa": [0],
    # "boolq": [0],
    # "rte": [0],
    # "arc_easy": [0],
    # "arc_challenge": [0],

    test_tasks = [
        "hendrycksTest-*", 'lambada_openai', "wikitext2", "ptb-new", "c4_new"

    ]

    test_tasks = ['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
     "hendrycksTest-*", "wikitext", "truthfulqa_mc", "openbookqa", "boolq", "rte", "arc_easy", "arc_challenge"]
    test_tasks = ['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai', 'hellaswag', 'winogrande', 'piqa',
  ]
    excel_name = (args.model_name).split('/')[-1] + ".xlsx"

    # test_tasks = ['wikitext2', 'ptb-new', 'c4-new', 'lambada_openai']
    eval_model(output_dir=args.model_name,
               tasks=test_tasks,
               eval_bs=args.bs, eval_orig_float=True, limit=None, excel_file=excel_name)

    print("cost time: ", time.time() - s)
