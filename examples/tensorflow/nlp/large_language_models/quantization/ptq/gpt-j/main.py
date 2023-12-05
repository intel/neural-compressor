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

import os
import time
import math
import numpy as np
import logging
import datasets
import tensorflow as tf
from typing import Optional
from itertools import chain
from datasets import load_dataset
from collections import defaultdict
from dataclasses import dataclass, field

import transformers
from transformers import (
    TF_MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForCausalLM,
    TFTrainingArguments,
    set_seed,
)
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt")
MODEL_CONFIG_CLASSES = list(TF_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to use.
    """

    model_name_or_path: Optional[str] = field(
        default="EleutherAI/gpt-j-6B",
        metadata={
            "help": (
                "The model checkpoint for GPT-J weights."
            )
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    precision: Optional[str] = field(
        default="fp32",
        metadata={"help": "The precision that we want to run with."},
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for evaluation.
    """

    dataset_name: Optional[str] = field(
        default="EleutherAI/lambada_openai", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class RunningArguments:
    """
    Arguments for options of running.
    """

    input_model: Optional[str] = field(
        default="./gpt-j-6B",
        metadata={
            "help": (
                "The path of input model."
            )
        },
    )
    output_model: Optional[str] = field(
        default="./nc_int8_gpt-j-6B",
        metadata={
            "help": (
                "The path save quantized gpt-j-6B int8 model."
            )
        },
    )
    tune: bool = field(
        default=False,
        metadata={"help": "Whether to apply quantization."},
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether to apply benchmarking."},
    )
    mode:  Optional[str] = field(
        default="performance",
        metadata={"help": ("Evaluate performance or accuracy benchmark." 
        "Set it to be accuracy or performance.")},
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    iteration: Optional[int] = field(
        default=200,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )



parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments, RunningArguments))
model_args, data_args, train_args, run_args = parser.parse_args_into_dataclasses()

logger.setLevel(logging.INFO)
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()

if train_args.seed is not None:
    set_seed(train_args.seed)

raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.checkpoint,
        use_auth_token=None,
    )
    
config = AutoConfig.from_pretrained(model_args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
column_names = raw_datasets["test"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

mydata = tokenizer(raw_datasets["test"][text_column_name], return_tensors="np").input_ids


def prepare_attention_mask_for_generation(
    inputs: tf.Tensor,
    pad_token_id=50256,
    eos_token_id=50256,
) -> tf.Tensor:
    """Generate attention_mask from input_ids.

    Args:
        inputs (tf.Tensor): The tensor of input_ids.

    Returns:
        attention_mask (tf.Tensor): The tensor of attention_mask.
    """
    is_input_ids = len(inputs.shape) == 2 and inputs.dtype in (tf.int32, tf.int64)
    is_pad_token_in_inputs = (pad_token_id is not None) and tf.math.reduce_any(inputs == pad_token_id)
    is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)

    # Check if input is input_ids and padded -> only then is attention_mask defined
    attention_mask = tf.cast(tf.math.not_equal(inputs, pad_token_id), dtype=tf.int32) \
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id \
            else tf.ones(inputs.shape[:2], dtype=tf.int32)

    return attention_mask

class MyDataloader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = math.ceil(len(dataset) / self.batch_size)

    def generate_data(self, data, pad_token_id=50256):
        input_ids = tf.convert_to_tensor([data[:-1]], dtype=tf.int32)
        cur_len = len(data)-1
        input_ids_padding = tf.ones((self.batch_size, 1), dtype=tf.int32) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        model_kwargs = {'attention_mask': prepare_attention_mask_for_generation(input_ids)}
        if model_kwargs.get("past_key_values") is None:
            input_ids = generated[:, :cur_len]
        else:
            input_ids = tf.expand_dims(generated[:, cur_len - 1], -1)
        return input_ids, model_kwargs['attention_mask']
    
    def __iter__(self):
        labels = None
        for _, data in enumerate(self.dataset):
            cur_input = self.generate_data(data)
            yield (cur_input, labels)

    def __len__(self):
        return self.length

def postprocess(outputs, generated, batch_size, cur_len):
    """The function that processes the inference outputs to prediction"""
    finished_sequences = tf.convert_to_tensor([False])
    next_token_logits = outputs['logits'][:, -1]
    # pre-process distribution
    next_tokens_scores = next_token_logits
    # argmax
    next_tokens = tf.argmax(next_tokens_scores, axis=-1, output_type=tf.int32)

    pad_token_id = 50256
    eos_token_id = [50256]

    unfinished_seq = 1 - tf.cast(finished_sequences, tf.int32)
    next_tokens = next_tokens * unfinished_seq + pad_token_id * (1 - unfinished_seq)
    next_token_is_eos = tf.math.reduce_any(
        tf.equal(
            tf.broadcast_to(next_tokens, (len(eos_token_id), batch_size)), tf.expand_dims(eos_token_id, -1)
        ),
        axis=0,
    )
    finished_sequences = finished_sequences | next_token_is_eos

    # update `generated` and `cur_len`
    update_indices = tf.stack([tf.range(batch_size), tf.broadcast_to(cur_len, [batch_size])], axis=-1)
    return tf.tensor_scatter_nd_update(tensor=generated, indices=update_indices, updates=next_tokens)

def evaluate(model, tf_eval_dataset=mydata):
    """Evaluate function that inference the model to apply calibration or benchmarking.

    Args:
        model (tf.python.training.tracking.tracking.AutoTrackable): The model to be evaluated.
            The object is usually gotten by using tf.saved_model.load(model_dir) API.

    Returns:
        accuracy (float): The accuracy result.
    """
    warmup = 5
    batch_size = run_args.batch_size
    pad_token_id = 50256
    iteration = run_args.iteration
    correct = 0
    latency_list = []
    infer = model.signatures["serving_default"]
    for idx, data in enumerate(tf_eval_dataset):
        print('Running Iteration: ', idx)
        input_ids = tf.convert_to_tensor([data[:-1]], dtype=tf.int32)
        cur_len = len(data)-1
        input_ids_padding = tf.ones((batch_size, 1), dtype=tf.int32) * (pad_token_id or 0)
        generated = tf.concat([input_ids, input_ids_padding], axis=-1)
        input_ids = generated[:, :cur_len]
        attention_mask = prepare_attention_mask_for_generation(input_ids)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

        start = time.time()
        outputs = infer(**inputs)
        end = time.time()
        dur = end-start

        predictions = postprocess(outputs, generated, batch_size, cur_len)
        if data[-1] == predictions[0][-1].numpy():
            correct+=1

        latency_list.append(dur)
        if idx >= iteration:
            break
    latency = np.array(latency_list[warmup:]).mean() / 1
    acc = correct/(iteration+1)
    if run_args.benchmark and run_args.mode == 'performance':
        print("Batch size = {}".format(run_args.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    return acc

def weight_name_mapping(name):
    """The function that maps name from AutoTrackable variables to graph nodes"""
    name = name.replace('tfgptj_for_causal_lm', 'StatefulPartitionedCall')
    name = name.replace('kernel:0', 'Tensordot/ReadVariableOp')
    return name

def main(): 
    with train_args.strategy.scope():
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        from neural_compressor import Model
        model = Model(run_args.input_model, modelType='llm_saved_model')

        if run_args.tune:
            from neural_compressor.config import AccuracyCriterion
            from neural_compressor import quantization, PostTrainingQuantConfig

            calib_dataloader = MyDataloader(mydata, batch_size=run_args.batch_size)  
            recipes = {"smooth_quant": True, "smooth_quant_args": {'alpha': 0.52705}}
            conf = PostTrainingQuantConfig(quant_level=1, 
                                            excluded_precisions=["bf16"],##use basic tuning
                                            recipes=recipes,
                                            calibration_sampling_size=[1],
                                            accuracy_criterion=AccuracyCriterion()
                                            )
            
            model.weight_name_mapping = weight_name_mapping
            q_model = quantization.fit( model,
                                        conf,
                                        eval_func=evaluate,
                                        calib_dataloader=calib_dataloader)

            q_model.save(run_args.output_model)
        if run_args.benchmark:
            if run_args.mode == "performance":
                from neural_compressor.benchmark import fit
                from neural_compressor.config import BenchmarkConfig
                conf = BenchmarkConfig(warmup=10, iteration=run_args.iteration, cores_per_instance=4, num_of_instance=1)
                fit(model, conf, b_func=evaluate)
            elif run_args.mode == "accuracy":
                acc_result = evaluate(model.model)
                print("Batch size = %d" % run_args.batch_size)
                print("Accuracy: %.5f" % acc_result)

if __name__ == "__main__":
    main()
    