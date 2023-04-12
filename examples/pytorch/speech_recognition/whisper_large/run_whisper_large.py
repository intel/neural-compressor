from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load


model_name = 'openai/whisper-large'
librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test")

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features)[0]
    transcription = processor.decode(predicted_ids)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

result = librispeech_test_clean.map(map_to_pred)


wer = load("wer")
wer_result = {100 * wer.compute(references=result["reference"], predictions=result["prediction"])}
print(f"Result wer: {wer_result}")




