import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor, pipeline

from auto_round import AutoRound

model_path = "/model/whisper-large-v3"
bits, group_size, sym, act_bits = 8, -1, True, 8

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, dtype=torch.float32, low_cpu_mem_usage=False, use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# quantize the model
'''
autoround = AutoRound(model, tokenizer, processor,
                        bits=bits, group_size=group_size, sym=sym, act_bits=act_bits,
                        iters=0, 
                        layer_config={
                            "proj_out": {
                                "bits": bits, 
                                "group_size": group_size, 
                                "sym": sym, 
                                "act_bits": act_bits,
                            }
                        },
                    )
autoround.quantize()
'''
autoround = AutoRound(model, tokenizer, scheme="w4a16", iters=0, group_size=group_size, sym=sym, processor=processor)

# save the quantized model, set format='auto_gptq' or 'auto_awq' to use other formats
output_dir = f"/model/whisper-large-v3-w4a16" # w{bits}a{act_bits}g{group_size}"
# autoround.save_quantized(output_dir, format="auto_awq", inplace=True)
autoround.quantize_and_save(output_dir, format="auto_round", inplace=True)