from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import argparse

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--saved_model_path", type=str, required=True)
args = parser.parse_args()

quantized_model_path=args.saved_model_path

model = AutoModelForCausalLM.from_pretrained(
    quantized_model_path,
    device_map="auto",
    dtype="bfloat16",
)

tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
text = "Solve the following math problem: What is 25 + 37? Please answer directly with the result."

inputs = tokenizer(text, return_tensors="pt").to(model.device)
res = tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0], skip_special_tokens=True)
print(res)
assert "62" in res, "Inference output does not contain the expected result."