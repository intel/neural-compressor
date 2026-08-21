#/bin/bash

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer

model_id = "openai/whisper-large-v3"
model_path = "/model/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(model_path)
processor.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
