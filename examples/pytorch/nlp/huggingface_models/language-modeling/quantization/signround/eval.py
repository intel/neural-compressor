import os.path
import torch
import torch.nn as nn
from parse_results import result_parser

import fnmatch
def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            if pattern == "json" or pattern.startswith("json="):
                task_names.add(pattern)

            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return sorted(list(task_names))
        

def eval_model(model, output_dir, tokenizer, tasks=["lambada_openai", "hellaswag", "winogrande", "piqa"], eval_bs=32, use_accelerate=False, device=None):
    if device is None: 
        device = str(model.device)
    if str(device) == "cpu"  or (hasattr(model, 'config') and model.config.torch_dtype is torch.bfloat16):
        model = model.to(torch.bfloat16)
        dtype = 'bfloat16'
    else:
        dtype = 'float16'
        model = model.half()
        
    results = []
    try:
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate as lm_evaluate
        print("evaluation with itrex lm-eval", flush=True)
        
        if use_accelerate or "coqa" in tasks or "hendrycksTest" in tasks:
            raise ValueError
        
        model_args = f'pretrained="{model_name}",tokenizer="{model_name}",dtype={dtype}'
        model.eval()
        results = lm_evaluate(model="hf-causal",
                              model_args=model_args,
                              user_model=model,
                              tasks=tasks,
                              device=str(device),
                              batch_size=eval_bs)
    except:
        print("evaluation with official lm-eval", flush=True)
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.tasks import ALL_TASKS
        import json
        import shutil

        ##save model
        output_dir = "./tmp_signround"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if output_dir is not None:
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
        if use_accelerate:
            model_args = f'pretrained={output_dir},tokenizer="{output_dir}",dtype={dtype},use_accelerate=True'
        else:
            model_args = f'pretrained="{output_dir}",tokenizer="{output_dir}",dtype={dtype}'
            
        task_names = pattern_match(tasks, ALL_TASKS)
        results = simple_evaluate(model="hf-causal", 
                                  model_args=model_args,
                                  tasks=task_names,
                                  device=str(device),
                                  batch_size=eval_bs,
                                  no_cache=True)
        # dumped = json.dumps(results, indent=2)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            
    result_parser(results)
    
    

