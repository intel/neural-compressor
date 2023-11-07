import os.path
import torch
import torch.nn as nn


def eval_model(model, model_name, tokenizer, tasks=["lambada_openai", "hellaswag", "winogrande", "piqa"], eval_bs=32, use_accelerate=False, device=None):
    if device is None: 
        device = str(model.device)
    if str(device) == "cpu"  or (hasattr(model, 'config') and model.config.torch_dtype is torch.bfloat16):
        model = model.to(torch.bfloat16)
        dtype = 'bfloat16'
    else:
        dtype = 'float16'
        model = model.half()
        
    try:
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate as lm_evaluate
        print("evaluation with itrex lm-eval", flush=True)
            
        if use_accelerate:
            user_model = None
            model_args = f'pretrained={model_name},tokenizer="{model_name}",dtype={dtype},use_accelerate=True'
        else:
            user_model = model
            model_args = f'pretrained="{model_name}",tokenizer="{model_name}",dtype={dtype}'
        model.eval()
        results = lm_evaluate(model="hf-causal",
                              model_args=model_args,
                              user_model=user_model,
                              tasks=tasks,
                              device=str(device),
                              batch_size=eval_bs)

    except:
        print("evaluation with official lm-eval", flush=True)
        from lm_eval.evaluator import simple_evaluate
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
            
        results = simple_evaluate(model="hf-causal", 
                                  model_args=model_args,
                                  tasks=tasks,
                                  device=str(device),
                                  batch_size=eval_bs,
                                  no_cache=True)
        dumped = json.dumps(results, indent=2)
        print(dumped)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
