#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os.path
import transformers
import tensorflow as tf
from tqdm import tqdm
import sys
import argparse
from datasets import load_dataset
import numpy as np

sys.path.insert(0, './')

parser = argparse.ArgumentParser()
parser.add_argument('--sq', action='store_true', default=False, help="whether to use smooth quant")
# parser.add_argument('--calib_num', type=int, default=100, help="calibration num for sq")
parser.add_argument('--model_name_or_path', type=str, default="facebook/opt-125m")
# TODO auto tuning not supported currently for TF backend
# parser.add_argument('--alpha', default=0.5, help="Set alpha=auto to use alpha tuning.")
parser.add_argument('--alpha', type=float, default=0.5, help="alpha value for smoothing.")
parser.add_argument('--log_frequency', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--kl', action='store_true', default=False, help="whether to use kl divergence for calibration")
parser.add_argument('--fallback_add', action='store_true', default=False, help="Whether to add fp32 fallback option" )
args = parser.parse_args()

class Evaluator:
    def __init__(self, dataset, tokenizer, device, batch_size=args.batch_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataloader = CustomDataloader(dataset, tokenizer, batch_size, device)

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
            if index % args.log_frequency == 0:
                print(hit / total, flush=True)
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
        for input_ids, label, label_indices in tqdm(self.dataloader):
            attention_mask = self.get_attention_mask(input_ids)
            input_ids = tf.constant(input_ids.numpy(), dtype=infer.inputs[0].dtype)
            attention_mask = tf.constant(attention_mask.numpy(), dtype=infer.inputs[0].dtype)
            results = infer(input_ids=input_ids, attention_mask=attention_mask) # len: 25 Identity: [16, 196, 50272], Identity_1: [16, 12, 196, 64]
            last_token_logits = results['Identity'].numpy()[np.arange(len(label_indices)), label_indices, :]
            pred = last_token_logits.argmax(axis=-1)
            total += label.shape[0]
            hit += (pred == label.numpy()).sum().item()
            if index % args.log_frequency == 0:
                print(hit / total, flush=True)
            index += 1
        acc = hit / total
        print(acc, flush=True)
        return acc

class CustomDataloader:
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
        input_id = tf.pad(input_id, tf.constant([[0,pad_len]]), constant_values=1)  # TODO need to check why pad with 1
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


model_name = args.model_name_or_path

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.TFAutoModelForCausalLM.from_pretrained(model_name)
eval_dataset = load_dataset('lambada', split='validation')

# model.eval()

evaluator = Evaluator(eval_dataset, tokenizer, 'cpu')

calib_dataset = load_dataset('lambada', split='train')
# calib_dataset = eval_dataset  # TODO for debug
calib_dataset = calib_dataset.shuffle(seed=42)
calib_dataloader = CustomDataloader(calib_dataset, tokenizer, device='cpu', batch_size=1, for_calib=True)

def eval_func(model):
    acc = evaluator.evaluate_tf_v1(model)
    return acc

from neural_compressor import PostTrainingQuantConfig
from neural_compressor.config import AccuracyCriterion

from neural_compressor import quantization

recipes = {}
if args.sq:
    recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': args.alpha}}
op_type_dict = {}
if args.kl:
    op_type_dict = {'linear': {'activation': {'algorithm': ['kl']}}}
if args.fallback_add:
    op_type_dict["add"] = {"weight": {"dtype": ["fp32"]}, "activation": {"dtype": ["fp32"]}}
conf = PostTrainingQuantConfig(quant_level=1, excluded_precisions=["bf16"],##use basic tuning
                                recipes=recipes,
                                op_type_dict=op_type_dict, accuracy_criterion=AccuracyCriterion(
tolerable_loss=0.011,      # TODO remove for debug
))

q_model = quantization.fit(model,
                            conf,
                            calib_dataloader=calib_dataloader,
                            eval_func=eval_func)
save_model_name = model_name.split("/")[-1]
q_model.save(f"{save_model_name}_int8")
