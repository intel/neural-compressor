
from auto_round import AutoRound

# Load a model (supports FP8/BF16/FP16/FP32)
model_name_or_path = "build/models/Llama2/Llama-2-70b-chat-hf/"
output_dir = "build/models/Llama2/Llama-2-70b-chat-hf-quantized"

#ar = AutoRound(model_name_or_path, scheme="MXFP4", iters=200, static_kv_dtype="fp8",)
ar = AutoRound(model_name_or_path, scheme="MXFP4", iters=200)

# Highest accuracy (4–5× slower).
# `low_gpu_mem_usage=True` saves ~20GB VRAM but runs ~30% slower.
# ar = AutoRound(model_name_or_path, nsamples=512, iters=1000, low_gpu_mem_usage=True)

# Faster quantization (2–3× speedup) with slight accuracy drop at W4G128.
# ar = AutoRound(model_name_or_path, nsamples=128, iters=50, lr=5e-3)

# Supported formats: "auto_round" (default), "auto_gptq", "auto_awq", "llm_compressor", "gguf:q4_k_m", etc.
# ar.quantize_and_save(output_dir="./tmp_autoround", format="llm_compressor")
ar.quantize_and_save(output_dir=output_dir, format="llm_compressor")


