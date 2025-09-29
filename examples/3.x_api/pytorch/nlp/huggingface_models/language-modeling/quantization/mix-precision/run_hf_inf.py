import torch
import sys


quantized_model_path = sys.argv[1]
print("model name or path:", quantized_model_path)
with torch.no_grad(), torch.device("cuda"):
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(
        quantized_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(quantized_model_path)
    prompt = "Solve the following math problem step by step: What is 25 + 37? Please answer directly with the result."

    encode = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_tokens = model.generate(
            encode,
            max_length=200,
        )
        output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        assert output is not None, "Output should not be None"