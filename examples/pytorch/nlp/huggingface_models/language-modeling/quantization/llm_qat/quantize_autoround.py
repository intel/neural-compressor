
from auto_round import AutoRound

model_name_or_path = "./llama3.1-finetuned"
output_dir = "./Llama-3.1-8B-Instruct_autoround_rtn_mxfp4"

# Available schemes: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4" (no real kernels), "GGUF:Q4_K_M", etc.
ar = AutoRound(model_name_or_path, scheme="MXFP4", iters=0)

ar.quantize_and_save(output_dir=output_dir, format="llm_compressor")
