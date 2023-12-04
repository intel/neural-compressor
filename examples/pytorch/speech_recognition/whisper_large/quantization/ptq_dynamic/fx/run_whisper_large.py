import os
import time
import argparse
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load

parser = argparse.ArgumentParser()
parser.add_argument('--int8', dest='int8', action='store_true')
parser.add_argument('--tune', dest='tune', action='store_true', 
                    help='tune best int8 model with Neural Compressor on calibration dataset')
parser.add_argument('--accuracy_only', dest='accuracy_only', action='store_true',
                    help='run accuracy_only')
parser.add_argument('--benchmark', dest='benchmark', action='store_true',
                    help='run benchmark')
parser.add_argument('--batch_size', default=1, type=int,
                    help='For accuracy measurement only.')
parser.add_argument('--iters', default=0, type=int,
                    help='For accuracy measurement only.')
parser.add_argument('--warmup_iters', default=5, type=int,
                    help='For benchmark measurement only.')
parser.add_argument('--output_dir', default="saved_results", type=str,
                    help='the folder path to save the results.')
parser.add_argument('--cache_dir', default=None, type=str,
                    help='the folder path to save the results.')

args = parser.parse_args()
model_name = 'openai/whisper-large'
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
# dataset
librispeech_test_clean = load_dataset("librispeech_asr", "clean", split="test", cache_dir=args.cache_dir)

# metric
wer = load("wer")

def eval_func(model):
    predictions = []
    references = []
    for batch in librispeech_test_clean:
        audio = batch["audio"]
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
        reference = processor.tokenizer._normalize(batch['text'])
        references.append(reference)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)[0]
        transcription = processor.decode(predicted_ids)
        prediction = processor.tokenizer._normalize(transcription)
        predictions.append(prediction)
    wer_result = wer.compute(references=references, predictions=predictions)
    print(f"Result wer: {wer_result * 100}")
    accuracy = 1 - wer_result
    print("Accuracy: %.5f" % accuracy)
    return accuracy

if args.tune:
    from neural_compressor import PostTrainingQuantConfig, quantization
    op_type_dict = {
            "Embedding": {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}
            }
    conf = PostTrainingQuantConfig(approach="dynamic", op_type_dict=op_type_dict)
    q_model = quantization.fit(model,
                               conf=conf,
                               eval_func=eval_func)
    q_model.save(args.output_dir)
    exit(0)

#benchmark
if args.int8:
    from neural_compressor.utils.pytorch import load
    model = load(
            os.path.abspath(os.path.expanduser(args.output_dir)), model)

if args.accuracy_only:
    eval_func(model)
    exit(0)

if args.benchmark:
    from neural_compressor.config import BenchmarkConfig
    from neural_compressor import benchmark
    def b_func(model):
        total_time = 0
        for i, batch in enumerate(librispeech_test_clean):
            if i > args.iters:
                break
            audio = batch["audio"]
            input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
            tic = time.time()
            with torch.no_grad():
                predicted_ids = model.generate(input_features)[0]
            toc = time.time()
            if i >= args.warmup_iters:
                total_time += (toc - tic)
        latency = total_time / (args.iters - args.warmup_iters)
        print('Latency: %.3f ms' % (latency * 1000))
        print('Throughput: %.3f images/sec' % (args.batch_size / latency))
        print('Batch size = %d' % args.batch_size)
    b_conf = BenchmarkConfig(
                                cores_per_instance=4,
                                num_of_instance=1
                                )
    benchmark.fit(model, b_conf, b_func=b_func)
    exit(0)
