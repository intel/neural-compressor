import torch

import torch.nn as nn

from evaluation import evaluate as lm_evaluate


def eval_model(model, model_name, tasks=["lambada_openai", "hellaswag", "winogrande", "piqa"], eval_bs=32):
    if str(model.device) == "cpu":
        print("eval on cpu")
        model = model.to(torch.bfloat16)  ##TODO support BF16 evaluation
        model.eval()
        results = lm_evaluate(model="hf-causal",
                              model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=bfloat16',
                              user_model=model, tasks=tasks,
                              device=str(model.device),
                              batch_size=eval_bs)
    else:
        model = model.half()  ##TODO support BF16 evaluation
        model.eval()
        results = lm_evaluate(model="hf-causal",
                              model_args=f'pretrained="{model_name}",tokenizer="{model_name}",dtype=float16',
                              user_model=model, tasks=tasks,
                              device=str(model.device),
                              batch_size=eval_bs)

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
