from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = TFAutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
model.save_pretrained('./gpt-j-6B', saved_model=True)