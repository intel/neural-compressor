import os.path
import torch
import torch.nn as nn

def get_layers(model):
    layers = []
    search_flag = False
    def unfoldLayer(module):
        nonlocal search_flag
        nonlocal layers
        if search_flag:
            return
        if hasattr(type(module),"__name__") and 'ModuleList' in type(module).__name__:
            layers = module
            search_flag = True
        layer_list = list(module.named_children())
        for item in layer_list:
            module = item[1]
            if isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)
    return layers

def set_parameter_device(model, device):
    required_ops = [model]
    flag = False
    while not flag and required_ops:
        module = required_ops.pop()
        inner_flag = False
        for name, ops in module.named_children():
            inner_flag = True
            if hasattr(type(ops), "__name__") and 'ModuleList' in type(ops).__name__:
                flag = True
                continue
            required_ops.append(ops)
        if not inner_flag:
            module = module.to(device)
        
    while required_ops:
        module = required_ops.pop()
        module = module.to(device)
    return model


def set_device_hooks(model, device):
    layers = get_layers(model)
    max_idx = len(layers)
    handles = []
    
    for idx, layer in enumerate(layers):
        layer.index = idx

    def layer_hook(self, inps, outs):
        cur_idx = self.index
        prev_idx = (cur_idx - 1 + max_idx) % max_idx
        next_idx = (cur_idx + 1 + max_idx) % max_idx
        layers[prev_idx] = layers[prev_idx].cpu()
        layers[next_idx] = layers[next_idx].to(device)
        
    for layer in layers:
        handles.append(layer.register_forward_hook(layer_hook))
        
    layers[0] = layers[0].to(device)
    return handles


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
        # if device is None:    
        #     device = str(model.device)
        # if str(device) == "cpu" or (hasattr(model, 'config') and model.config.torch_dtype is torch.bfloat16):
        #     dtype = 'bfloat16'
        # else:
        #     dtype = 'float16'
            
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

    @torch.no_grad()
    def eval_same_with_gptq(model, testenc, dev):
        print('Evaluating ...', flush=True)
        # model.eval()
        # model.to(dev)

        testenc = testenc.input_ids
        nsamples = testenc.numel() // model.seqlen

        use_cache = model.config.use_cache
        model.config.use_cache = False

        testenc = testenc.to(dev)
        nlls = []
        for i in range(nsamples):
            batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
            lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = testenc[
                           :, (i * model.seqlen):((i + 1) * model.seqlen)
                           ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print(ppl.item())

        model.config.use_cache = use_cache
        return ppl.item()
    
    handles = None
    if use_accelerate and dtype != 'bfloat16': # Use block-wise evaluation for large models.
        model = model.cpu()
        model = set_parameter_device(model, device)
        handles = set_device_hooks(model, device)

    datasets = ['wikitext2', 'ptb-new', 'c4-new']

    from gptq_data_loader import get_loaders
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=0, model=model_name, seqlen=model.seqlen
        )
        print(dataset, flush=True)
        ppl = eval_same_with_gptq(model, testloader, device)
        results.update({dataset: ppl})
    
    if handles is not None:
        for handle in handles:
            handle.remove()
