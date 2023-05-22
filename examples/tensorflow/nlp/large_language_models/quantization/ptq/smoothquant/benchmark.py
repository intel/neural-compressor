import os.path
import transformers
import tensorflow as tf
from tqdm import tqdm
import sys
import argparse
from datasets import load_dataset
import numpy as np
import time

sys.path.insert(0, './')

parser = argparse.ArgumentParser()
parser.add_argument('--int8', action='store_true', help="eval fp32 model or int8 model")
parser.add_argument('--model_name_or_path', type=str, default='facebook/opt-125m')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--warmup', type=int, default=10)
args = parser.parse_args()

class Evaluator:
    def __init__(self, dataset, tokenizer, device, batch_size=args.batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataloader = INCDataloader(dataset, tokenizer, batch_size, device)

    def evaluate(self, model):
        # model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        index = 1
        for input_ids, label, label_indices in tqdm(self.dataloader):
            # TFCausalLMOutputWithPast len: 2
            # first element shape (16, 196, 50272)
            # second element shape (16, 12, 196, 64)
            outputs = model(input_ids)
            last_token_logits = outputs[0].numpy()[np.arange(len(label_indices)), label_indices, :]
            pred = last_token_logits.argmax(axis=-1)
            total += label.shape[0]
            hit += (pred == label.numpy()).sum().item()
            index += 1
        acc = hit / total
        print(acc, flush=True)
        return acc
    
    def get_attention_mask(self, input_ids):
        return tf.constant(1 - (input_ids==1).numpy().astype(int))
    
    def evaluate_tf_v1(self, model):
        # return 0.99 # TODO debug remove
        total, hit = 0, 0
        index = 1
        infer = model.signatures["serving_default"]
        overall_infer_duration = 0
        for input_ids, label, label_indices in tqdm(self.dataloader):
            attention_mask = self.get_attention_mask(input_ids)
            input_ids = tf.constant(input_ids.numpy(), dtype=infer.inputs[0].dtype)
            attention_mask = tf.constant(attention_mask.numpy(), dtype=infer.inputs[0].dtype)
            start = time.time()
            results = infer(input_ids=input_ids, attention_mask=attention_mask) # len: 25 Identity: [16, 196, 50272], Identity_1: [16, 12, 196, 64]
            batch_infer_time = time.time() - start
            if index > args.warmup:
                overall_infer_duration += batch_infer_time
            last_token_logits = results['Identity'].numpy()[np.arange(len(label_indices)), label_indices, :]
            pred = last_token_logits.argmax(axis=-1)
            total += label.shape[0]
            hit += (pred == label.numpy()).sum().item()
            index += 1
        acc = hit / total
        print("\nEvaluation result: ")
        print(f"Batch size = {args.batch_size}")
        print(f"Accuracy: {acc}")
        print(
            f"Throughput: {(len(self.dataloader) - args.warmup * args.batch_size) / overall_infer_duration} samples/sec"
        )

class INCDataloader:
    # for_calib=True in quantization, only input_id is needed, =False in evaluation need label
    def __init__(self, dataset, tokenizer, batch_size=1, device='cpu', for_calib=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.for_calib = for_calib
        import math
        self.length = math.ceil(len(dataset) / self.batch_size) # batch number
        self.pad_len = 196
        
        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example
        
        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='tensorflow', columns=['input_ids'])
    def get_attention_mask(self, input_ids):
        return 1 - (input_ids==1).numpy().astype(int)
    def pad_input(self, input): # input: a record
        input_id = input['input_ids']
        if input_id.numpy().shape[0] > self.pad_len: # truncate the sequence to pad_len if the sequence is longer than pad_len
            input_id = input_id[:self.pad_len]
        label = input_id[-1]
        pad_len = self.pad_len - input_id.numpy().shape[0]
        label_index = -2 - pad_len  # last logit index
        input_id = tf.pad(input_id, tf.constant([[0,pad_len]]), constant_values=1)
        input_id = tf.expand_dims(input_id, axis=0)
        label = tf.expand_dims(label, axis=0)
        return (input_id, label, label_index)
    
    def __iter__(self):
        if self.for_calib:
            labels = None
            # label_indices = None
            for idx, record in enumerate(self.dataset):
                input_id, label, label_index = self.pad_input(record)
                attention_mask = self.get_attention_mask(input_id)
                # compose attention_mask and input_id together
                # during the calibration, it requires to yield a <attention_mask, input_id>
                # cur_input = tf.constant(np.append(attention_mask, input_id.numpy(), axis=0))
                cur_input = {"input_ids": input_id.numpy(), "attention_mask": attention_mask}
                assert self.batch_size == 1
                yield (cur_input, label)
        else:
            input_ids = None
            labels = None
            label_indices = None
            for idx, record in enumerate(self.dataset):
                input_id, label, label_index = self.pad_input(record)
                if input_ids is None:
                    input_ids = input_id
                    labels = label
                    label_indices = [label_index]
                else:
                    input_ids = tf.concat([input_ids, input_id], 0)
                    labels = tf.concat([labels, label], 0)
                    
                    label_indices.append(label_index)

                if (idx + 1) % self.batch_size == 0:
                    yield (input_ids, labels, label_indices)
                    input_ids = None
                    labels = None
                    label_indices = None
            if (idx + 1) % self.batch_size != 0:
                yield (input_ids, labels, label_indices)

    def __len__(self):
        return self.length

from datasets import load_dataset

model_name = args.model_name_or_path
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
)
eval_dataset = load_dataset('lambada', split='validation')

evaluator = Evaluator(eval_dataset, tokenizer, 'cpu')

if args.int8:
    print("benchmarking int8 model")
    int8_folder = model_name.split('/')[-1] + "_int8"
    if not os.path.exists(int8_folder):
        print(f"could not find int8 folder {int8_folder} ")
        exit()
    model = tf.saved_model.load(int8_folder)    # tensorflow.python.trackable.autotrackable.AutoTrackable object
else:
    print("benchmaking fp32 model")
    model = transformers.TFAutoModelForCausalLM.from_pretrained(model_name)
    # fp32_folder = model_name.split('/')[-1] + "_fp32"
    # model.save(fp32_folder)
    # model = tf.keras.models.load_model(fp32_folder)
    from neural_compressor.experimental import common
    def keras2SavedModel(model):
        model = common.Model(model)
        return model.model
    model = keras2SavedModel(model) # tensorflow.python.trackable.autotrackable.AutoTrackable object

# TODO current neural_compressor.benchmark does not support AutoTrackable model, we will write our own
# from neural_compressor.benchmark import fit
# from neural_compressor.config import BenchmarkConfig
# conf = BenchmarkConfig(cores_per_instance=28, num_of_instance=1)
# fit(model, conf, b_func=evaluator.evaluate_tf_v1)
evaluator.evaluate_tf_v1(model)
