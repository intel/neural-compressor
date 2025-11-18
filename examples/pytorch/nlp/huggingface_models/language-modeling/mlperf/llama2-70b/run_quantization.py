from argparse import ArgumentParser
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
#from auto_round import AutoRound
from dataset import Dataset
import torch
torch.use_deterministic_algorithms(True, warn_only=True)

#=========================
import os
import subprocess
import pandas as pd

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from transformers import set_seed

import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#=========================

def get_args():
    parser = ArgumentParser(description="AutoRound quantization script")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to quantize")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset for quantization")
    parser.add_argument("--bits", type=int, default=4, help="Number of bits for quantization")
    parser.add_argument("--group_size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--sym", action='store_true', help="Use symmetric quantization")
    parser.add_argument("--nsamples", type=int, default=512, help="Number of samples for quantization")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations for quantization")
    parser.add_argument("--output_dir", type=str, default="./tmp_autoround", help="Directory to save the quantized model")
    parser.add_argument("--quant_lm_head", action='store_true', help="Quantize the language model head")
    parser.add_argument("--act_bits", type=int, default=8, help="Number of bits for activation quantization")
    parser.add_argument("--device", type=str, default="xpu", help="Device to run the quantization on")
    return parser.parse_args()

def main():
    args = get_args()
    model_name = args.model_name
    dataset_path = args.dataset_path
    nsamples = args.nsamples
    iters = args.iters
    bits = args.bits
    group_size = args.group_size
    sym = args.sym
    output_dir = args.output_dir

    # Load dataset
    calib_dataset = Dataset(model_name=model_name, dataset_path=dataset_path, total_sample_count=nsamples)
    tokenizer = calib_dataset.tokenizer
    calib_dataset = [torch.tensor(input).reshape(1, -1) for input in calib_dataset.input_ids[:nsamples]]

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True, 
                                                trust_remote_code=True,)
    
    model = model.eval()  # Set model to evaluation mode
    model = model.to(torch.float16)  # Convert model to float16 for quantization
    #model.seqlen = 3072

    layer_config = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, transformers.modeling_utils.Conv1D):
            if m.weight.shape[0] % 32 != 0 or m.weight.shape[1] % 32 != 0:
                layer_config[n] = {"bits": 32}
                print(
                    f"{n} will not be quantized due to its shape not being divisible by 32, resulting in an exporting issue to autogptq")
    lm_head_layer_name = "lm_head"
    for n, _ in model.named_modules():
        lm_head_layer_name = n
    if args.quant_lm_head:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=not args.disable_trust_remote_code)
        if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
            tied_keys = model._tied_weights_keys
            for item in tied_keys:
                if lm_head_layer_name in item:  ##TODO extend to encoder-decoder layer, seq classification model
                    args.quant_lm_head = False
                    print(
                        f"warning, disable quant_lm_head as quantizing lm_head with tied weights has not been "
                        f"supported currently")
                    break
    if args.quant_lm_head:
        layer_config[lm_head_layer_name] = {"bits": args.bits}
        transformers_version = [int(item) for item in transformers.__version__.split('.')[:2]]
        if transformers_version[0] == 4 and transformers_version[1] < 38:
            error_message = "Please upgrade transformers>=4.38.0 to support lm-head quantization."
            raise EnvironmentError(error_message)
        
    from auto_round import AutoRound

    qmodel = AutoRound(model=model,
                       tokenizer=tokenizer,
                       bits=bits,
                       group_size=group_size,
                       sym=sym,
                       dataset=calib_dataset,
                          seqlen=256,
                          iters=iters,
                          device=args.device,
                          gradient_accumulate_steps=1,
                          batch_size=1,
                          layer_config=layer_config,
                          model_dtype="float16"
                          )
    
    export_dir = output_dir + "/" + model_name.split('/')[-1] + f"-autoround-w{bits}g{group_size}-iters{iters}-{args.device}"
    
    qmodel.quantize_and_save(export_dir, format='auto_awq')

if __name__ == "__main__":
    main()
