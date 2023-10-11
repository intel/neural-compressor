import os.path
import torch
import torch.nn as nn

def eval_model(model, model_name,tokenizer, tasks=["lambada_openai", "hellaswag", "winogrande", "piqa"], eval_bs=32):
    try:
        from intel_extension_for_transformers.llm.evaluation.lm_eval import evaluate as lm_evaluate
        print("evaluation with itrex lm-eval", flush=True)
        # ours
        if str(model.device) == "cpu":
            model = model.to(torch.bfloat16)
            results = lm_evaluate(model="hf-causal",
                                  model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=bfloat16',
                                  user_model=model,
                                  tasks=tasks,
                                  device=str(model.device),
                                  batch_size=eval_bs)
        else:
            model = model.half()
            model.eval()
            results = lm_evaluate(model="hf-causal",
                                  model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=float16',
                                  user_model=model,
                                  tasks=tasks,
                                  device=str(model.device),
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


        if str(model.device) == "cpu":
            results = simple_evaluate(model="hf-causal",
                                      model_args=f'pretrained="{output_dir}",tokenizer="{output_dir}",dtype=bfloat16',
                                      tasks=tasks,
                                      device=str(model.device),
                                      batch_size=eval_bs,
                                      no_cache=True)
        else:
            results = simple_evaluate(model="hf-causal",
                                      model_args=f'pretrained="{output_dir}",tokenizer="{output_dir}",dtype=float16',
                                      tasks=tasks,
                                      device=str(model.device),
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
        model.to(dev)

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

    datasets = ['wikitext2', 'ptb-new', 'c4-new']

    from gptq_data_loader import get_loaders
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=0, model=model_name, seqlen=model.seqlen
        )
        print(dataset, flush=True)
        ppl = eval_same_with_gptq(model, testloader, str(model.device))
        results.update({dataset: ppl})
