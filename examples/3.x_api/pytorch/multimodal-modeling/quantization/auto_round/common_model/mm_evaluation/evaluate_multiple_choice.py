import argparse
import itertools
import json
import os
from functools import partial

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

multiple_choices = ['A', 'B', 'C', 'D', 'E']

ds_collections = {
    'scienceqa_test_img': {
        'test': 'data/scienceqa/scienceqa_test_img.jsonl',
    }
}


def collate_fn(batches, pad_token_id):

    input_tokens = [_['input_tokens'] for _ in batches]
    target_lengths = [_['target_lengths'] for _ in batches]
    answers = [_['answer'] for _ in batches]

    chunk_sizes = [len(_) for _ in input_tokens]

    input_tokens = [_ for _ in itertools.chain.from_iterable(input_tokens)]

    max_lengths = max([len(_) for _ in input_tokens])
    input_tokens = [[pad_token_id] * (max_lengths - len(_)) + _
                    for _ in input_tokens]
    input_tokens = torch.LongTensor(input_tokens)

    attention_mask = 1 - input_tokens.eq(pad_token_id).float()

    return input_tokens, attention_mask, target_lengths, answers, chunk_sizes


class MultipleChoiceDataste(torch.utils.data.Dataset):

    def __init__(self, test, prompt, tokenizer):
        self.data = open(test).readlines()
        self.prompt = prompt
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = json.loads(self.data[idx].strip())
        image = data['image']
        hint = data['hint'] if data['hint'] else 'N/A'
        question = data['question']

        choices = data['choices']
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append('{}. {}'.format(multiple_choices[i], c))
        choice_txt = '\n'.join(choice_list)

        prompt = self.prompt.format(image, hint, question, choice_txt)

        prompt_tokens = self.tokenizer(prompt).input_ids
        target_tokens = [
            self.tokenizer(' ' + _).input_ids
            for _ in multiple_choices[:len(choices)]
        ]

        return {
            'input_tokens': [prompt_tokens + _ for _ in target_tokens],
            'target_lengths': [len(_) for _ in target_tokens],
            'answer': data['answer'],
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def scienceQA_evaluation(model_name, dataset_name, dataset_path=None, tokenizer=None,
                       batch_size=1, few_shot=0, seed=0, trust_remote_code=True, device="cuda:0"):
    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     world_size=int(os.getenv('WORLD_SIZE', '1')),
    #     rank=int(os.getenv('RANK', '0')),
    # )
    if "cuda" in device:
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    if isinstance(model_name, str):
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=trust_remote_code).eval()
        model = model.to(torch.device(device))
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
    else:
        assert tokenizer is not None, "Two types of parameter passing are supported:model_path or model with tokenizer."
        model = model_name

    prompt = '<img>{}</img>Context: {}\nQuestion: {}\nOptions: {}\nAnswer:'

    dataset = MultipleChoiceDataste(test=ds_collections[dataset_name]['test'],
                                    prompt=prompt,
                                    tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        # sampler=InferenceSampler(len(dataset)),
        batch_size=batch_size,
        # num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.eod_id),
    )

    results = []
    with torch.no_grad():
        for _, (input_tokens, attention_mask, target_lengths, answer,
                chunk_sizes) in tqdm(enumerate(dataloader)):

            outputs = model(
                input_ids=input_tokens[:, :-1].to(device),
                attention_mask=attention_mask[:, :-1].to(device),
                return_dict=True,
            )
            losses = torch.nn.functional.cross_entropy(outputs.logits.permute(
                0, 2, 1),
                                                       input_tokens[:,
                                                                    1:].to(device),
                                                       reduction='none')

            losses = losses.split(chunk_sizes, dim=0)

            for loss, target_length, answer in zip(losses, target_lengths,
                                                   answer):

                target_loss = loss.mean(-1)
                for _ in range(len(target_length)):
                    target_loss[_] = loss[_, -target_length[_]:].mean()
                pred = target_loss.argmin().item()
                if pred == answer:
                    results.append(1)
                else:
                    results.append(0)

    # torch.distributed.barrier()

    # world_size = torch.distributed.get_world_size()
    # merged_results = [None for _ in range(world_size)]
    # torch.distributed.all_gather_object(merged_results, results)
    merged_results = [json.dumps(results)]
    merged_results = [json.loads(_) for _ in merged_results]
    merged_results = [_ for _ in itertools.chain.from_iterable(merged_results)]

    # if torch.distributed.get_rank() == 0:
    print(f"Evaluating {dataset_name} ...")
    print(f'Acc@1: {sum(merged_results) / len(merged_results)}')

    # torch.distributed.barrier()




if __name__ == "__main__":
    import sys
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="Qwen/Qwen-VL"
    )
    parser.add_argument(
        "--dataset_name", default="scienceqa_test_img"
    )
    parser.add_argument(
        "--eval_bs", default=4,
    )
    parser.add_argument(
        "--trust_remote_code", action='store_true',
        help="Whether to enable trust_remote_code"
    )
    args = parser.parse_args()
    s = time.time()
    evaluator = scienceQA_evaluation(
        args.model_name,
        dataset_name=args.dataset_name,
        # dataset_path=args.eval_path,
        batch_size=args.eval_bs,
        trust_remote_code=args.trust_remote_code
    )
    print("cost time: ", time.time() - s)

    
