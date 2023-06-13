from optimum.onnxruntime import ORTModelForCausalLM
from transformers import PretrainedConfig
import os
import json
import time
import numpy as np
import psutil
import onnxruntime as ort
from itertools import chain
from transformers import LlamaForCausalLM, LlamaTokenizer
import argparse

def get_memory_usage(name):
    memory_allocated = round(psutil.Process().memory_info().rss/1024**3, 3)
    print(name, 'memory used total:', memory_allocated, 'GB')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alexnet fine-tune examples for image classification tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--onnx_model_path',
        type=str,
        help="onnx_model_path"
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help="model_name_or_path"
    )
    parser.add_argument(
        '--search_algorithm',
        type=str,
        default='beam',
        help="search_algorithm"
    )
    parser.add_argument(
        '--intra_op_num_threads',
        type=int,
        default=16,
        help="intra_op_num_threads"
    )

    args = parser.parse_args()

    token_latency = False
    device = 'cpu'

    if args.search_algorithm == 'beam':
        # beam search = 4i
        generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
    elif args.search_algorithm == 'greedy':
        generate_kwargs = dict(do_sample=False, num_beams=1)

    model = args.onnx_model_path
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    #tokenizer = LlamaTokenizer.from_pretrained('/lfs/opa01/mengfeil/LLaMA/13B')
    config = PretrainedConfig.from_pretrained(args.model_name_or_path)
    #config = PretrainedConfig.from_pretrained('/lfs/opa01/mengfeil/LLaMA/13B')
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.intra_op_num_threads
    sessions = ORTModelForCausalLM.load_model(
            os.path.join(model, 'decoder_model.onnx'), 
            #os.path.join(model, 'decoder_with_past_model.onnx'), 
            session_options=sess_options)
    model = ORTModelForCausalLM(
                sessions[0],
                config, 
                model, 
                sessions[1],
                use_cache=False)

    input_tokens = '32'
    max_new_tokens = 32
    with open('prompt.json') as f:
        prompt_pool = json.load(f)
    if input_tokens in prompt_pool:
        prompt = prompt_pool[input_tokens]
    else:
        raise SystemExit('[ERROR] Plese use --prompt if want to use custom input.')

    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
    print("---- Prompt size:", input_size)
    if token_latency:
        generate_kwargs["token_latency"] = True

    total_time = 0.0
    num_iter = 100
    num_warmup = 10
    batch_size = 1
    prompt = [prompt] * batch_size
    total_list = []

    for i in range(num_iter):
        get_memory_usage("Iteration: " + str(i))
        tic = time.time()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, **generate_kwargs
        )
        gen_ids = output[0] if token_latency else output
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        toc = time.time()
        print(gen_text, flush=True)
        if i >= num_warmup:
            total_time += toc - tic
            if token_latency:
                total_list.append(output[1])

    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter - num_warmup)
    print(args)
    print("Inference latency: %.3f sec." % latency)
    if token_latency:
        first_latency = np.mean([x[0] for x in total_list])
        average_2n = list(chain(*[x[1:] for x in total_list]))
        average_2n.sort()
        average_2n_latency = np.mean(average_2n)
        p90_latency = average_2n[int(len(average_2n) * 0.9)]
        print("First token average latency: %.3f sec." % first_latency)
        print("Average 2... latency: %.3f sec." % average_2n_latency)
        print("P90 2... latency: %.3f sec." % p90_latency)
