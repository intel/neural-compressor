import collections
import itertools
import random
import sys
sys.path.append("/data2/lkk/eval_all/lm-evaluation-harness")

try:
    import lm_eval.metrics
    import lm_eval.models
    import lm_eval.tasks
    import lm_eval.base
    from lm_eval.utils import positional_deprecated, run_task_tests
    from lm_eval.models.gpt2 import HFLM
    from lm_eval import tasks, evaluator, utils
except:
    raise ImportError("""git clone https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    pip install -e .""")

import numpy as np
import transformers

import argparse
import json
import logging
import os
import torch
import torch.nn as nn

EXT_TASKS = ['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None, help="init gpu")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )

    return parser.parse_args()


@torch.no_grad()
def eval_ppl_same_with_gptq(model, testenc, dev):
    print('Evaluating ...', flush=True)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        lm_logits = model(batch).logits
        # to main gpu for multi-gpus
        shift_logits = lm_logits[:, :-1, :].contiguous().to(dev)
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:].to(dev)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    model.config.use_cache = use_cache
    return ppl.item()


def main():
    args = parse_args()

    internal_tasks = args.tasks.split(",")
    external_tasks = []
    for each in internal_tasks:
        if each in EXT_TASKS:
            external_tasks.append(each)
            internal_tasks.remove(each)

    print("lm-eval internal tasks: ", internal_tasks)
    print("external tasks: ", external_tasks)

    task_names = utils.pattern_match(internal_tasks, tasks.ALL_TASKS)
    task_dict = lm_eval.tasks.get_task_dict(task_names)

    if isinstance(args.model, str):
        if args.model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(args.model).create_from_arg_string(
            args.model_args, {"batch_size": args.batch_size, "max_batch_size": args.max_batch_size, "device": args.device}
        )
    elif isinstance(args.model, transformers.PreTrainedModel):
        lm = lm_eval.models.get_model("hf-causal")(
                pretrained=args.model,
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                )
        no_cache = True
    else:
        assert isinstance(args.model, lm_eval.base.LM)
        lm = model

    # for lm-eval internal tasks
    bootstrap_iters=100000
    description_dict=None
    decontamination_ngrams_path=None
    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        bootstrap_iters=bootstrap_iters,
        description_dict=description_dict,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    # for external tasks

    # maybe adjust for specific model
    lm.model.seqlen = 2048

    from datautils import get_loaders

    for dataset in external_tasks:
        dataloader, testloader = get_loaders(
                dataset, nsamples=args.nsamples, seed=args.seed, tokenizer=lm.tokenizer, seqlen=lm.model.seqlen
                )
        ppl = eval_ppl_same_with_gptq(lm.model, testloader, args.device)

        results.update({dataset: ppl})

    print(results)


if __name__ == "__main__":
    main()
