import torch
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
# MODEL_ID = "openai/whisper-large-v3"
MODEL_ID = "/models/whisper-large-v3"

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(MODEL_ID)

# Configure processor the dataset task.
processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

# recipe = QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
recipe = QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])

# Apply quantization.
oneshot(model=model, recipe=recipe)



# that's where you have a lot of windows in the south no actually that's passive solar
# and passive solar is something that was developed and designed in the 1960s and 70s
# and it was a great thing for what it was at the time but it's not a passive house

# Save to disk compressed.
#SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W8A8-INT8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
