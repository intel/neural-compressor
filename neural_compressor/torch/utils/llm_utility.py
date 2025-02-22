# -*- coding: utf-8 -*-
# Copyright (c) 2025 Intel Corporation
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


def initialize_model_and_tokenizer(model_name_or_path, use_load=False, device="cpu"):
    import transformers

    from neural_compressor.torch.utils import local_rank, logger, world_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    if use_load:
        from neural_compressor.torch.quantization import load

        model = load(model_name_or_path, format="huggingface", device=device)
        model, tokenizer = update_tokenizer(model, tokenizer)
        return model, tokenizer
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    # using memory mapping with torch_dtype=config.torch_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=config.torch_dtype)
    model, tokenizer = update_tokenizer(model, tokenizer)
    # shard model for multi-cards and enable hpu graph
    if world_size > 1:
        if "hpu" in device:
            logger.warning("The model will be loaded via memory mapping, so the device settings are useless")
        ds_inference_kwargs = {
            "dtype": config.torch_dtype,
            "tensor_parallel": {"tp_size": world_size},
            "keep_module_on_host": True,
        }
        import deepspeed

        ds_model = deepspeed.init_inference(model, **ds_inference_kwargs)
        model = ds_model.module
    model.eval()
    return model, tokenizer


def update_tokenizer(model, tokenizer):
    if model.config.model_type in ["llama", "mixtral"]:
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
    return model, tokenizer


def get_default_llm_dataloader(tokenizer, dataset_name="NeelNanda/pile-10k", bs=8, nsamples=128, seq_len=128, seed=42):
    """Generate dataloader based on dataset name and other configurations.

    Args:
        tokenizer (obj): tokenizer object.
        seq_len (int, optional): _description_. Defaults to 128.
        dataset_name (str, optional): dataset name. Defaults to "NeelNanda/pile-10k".
        seed (int, optional): random seed. Defaults to 42.
        bs (int, optional): batch size. Defaults to 8.
        nsamples (int, optional): number of samples. Defaults to 128.

    Returns:
        dataloader: dataloader
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader, Dataset

    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=seed).select(range(nsamples))

    class TokenizedDataset(Dataset):
        def __init__(self, dataset, tokenizer, seq_len):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.seq_len = seq_len

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            text = self.dataset[idx]["text"]
            inputs = self.tokenizer(
                text, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt"
            )
            return {key: val.squeeze(0) for key, val in inputs.items()}

    tokenized_dataset = TokenizedDataset(dataset, tokenizer, seq_len)
    dataloader = DataLoader(tokenized_dataset, batch_size=bs, shuffle=True)
    return dataloader


def llm_benchmark(model, batch_size, input_length, warmup_iters=3, total_iters=20):
    import time

    import torch

    from neural_compressor.torch.utils import get_accelerator, logger

    cur_accelerator = get_accelerator()
    # this is a simple example to show the performance benefit of quantization
    example_inputs = torch.ones((batch_size, input_length), dtype=torch.long)
    logger.info("Batch size = {:d}".format(batch_size))
    logger.info("The length of input tokens = {:d}".format(input_length))

    with torch.no_grad():
        for i in range(total_iters):
            if i == warmup_iters:
                start = time.perf_counter()
            model(example_inputs)
            cur_accelerator.synchronize()
        end = time.perf_counter()
    latency = (end - start) / ((total_iters - warmup_iters) * batch_size)
    throughput = ((total_iters - warmup_iters) * batch_size) / (end - start)
    logger.info("Latency: {:.3f} ms".format(latency * 10**3))
    logger.info("Throughput: {:.3f} samples/sec".format(throughput))
