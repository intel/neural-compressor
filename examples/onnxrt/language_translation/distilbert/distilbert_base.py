"""
MRPC with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained DistilBERT parameters for
for Microsoft Research Paraphrase Corpus (MRPC) task.

@article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}
"""

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation


from __future__ import absolute_import, division, print_function

import os
import time
import logging
import argparse

import numpy as np
import onnx
import onnxruntime
import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm
from transformers import DistilBertTokenizer
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features


logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

def parse_dummy_input(model, benchmark_nums, max_seq_length):
    session = onnxruntime.InferenceSession(model.SerializeToString(), None)
    shapes = []
    lows = []
    highs = []
    for i in range(len(session.get_inputs())):
        input_name = session.get_inputs()[i].name
        input_shapes = session.get_inputs()[i].shape
        shape = [benchmark_nums]
        for input_shape in input_shapes:
            if 'seq' in input_shape :
                shape.append(max_seq_length)
            if input_name == "input_ids":
                low = 0.0
                high = 1000.0
            else:
                low = 0.0
                high = 2.0
        shapes.append(tuple(shape))
        lows.append(low)
        highs.append(high)
    return shapes, lows, highs

def evaluate_onnxrt(args, model, tokenizer, eval_dataloader):
    session = onnxruntime.InferenceSession(model.SerializeToString(), None)
    output_mode = output_modes[args.task_name]

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name

    results = {}

    # Eval!
    logger.info("***** Running evaluation  *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    #eval_loss = 0.0
    #nb_eval_steps = 0
    preds = None
    out_label_ids = None
    latencies = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.detach().cpu().numpy()  \
                            if not isinstance(t, np.ndarray) else t \
                            for t in batch)
        ort_inputs = {
                            session.get_inputs()[0].name:  batch[0],
                            session.get_inputs()[1].name: batch[1]
                        }
        logits = np.reshape(session.run(None, ort_inputs)[0], (-1,2))
        if preds is None:
            preds = logits
            out_label_ids = batch[2]
        else:
            preds = np.append(preds, logits, axis=0)
            out_label_ids = np.append(out_label_ids, batch[2], axis=0)
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)
    return results["acc"]

def load_and_cache_examples(args, task, tokenizer, evaluate=False):

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if not os.path.exists("./dataset_cached"):
        os.makedirs("./dataset_cached")
    cached_features_file = os.path.join("./dataset_cached", 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir) if evaluate else \
            processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    return dataset

if __name__ == "__main__":
    logger.info('Evaluating ONNXRuntime full precision accuracy and performance:')
    parser = argparse.ArgumentParser(
    description='DistilBERT fine-tune examples for classification/regression tasks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=8,
        help='Batch size for dev set and test set')
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='ratio of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='report interval')
    parser.add_argument(
        '--seed', type=int, default=2, help='Random seed')
    parser.add_argument(
        '--gpu', type=int, default=None, help='Which gpu for finetuning.')
    parser.add_argument(
        '--task_name',
        type=str,
        choices=['mnli', 'mrpc'],
        help='The name of the task to fine-tune. Choices include MRPC, QQP, '
            'QNLI, RTE, STS-B, CoLA, MNLI, WNLI, SST.')
    parser.add_argument(
        '--bert_dataset',
        type=str,
        default='MRPC',
        choices=['MRPC'],
        help='The dataset DistilBERT pre-trained with.')
    parser.add_argument(
        '--model_type',
        type=str,
        default='distilbert')
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Pre-trained distilbert model onnx file.')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='./',
        help='The input directory where the model params are stored.')
    parser.add_argument(
        '--only_inference',
        action='store_true',
        help='If set, we skip training and only perform inference on dev and test data.')
    parser.add_argument('--max_seq_length', type=int,
                        default=128,
                        help='max seq length')
    parser.add_argument('--model_name_or_path', type=str,
                        default='distilbert-base-uncased',
                        help='model name or path')
    parser.add_argument('--data_dir', type=str,
                        help='datseset path')
    parser.add_argument('--tune',action='store_true', default=False,
                        help='Get distilbert tuning quantization model with lpot.')
    parser.add_argument('--config',type=str, default=None,
                        help='Tuning config file path')
    parser.add_argument('--output_model',type=str, default=None,
                        help='output model path and name')
    parser.add_argument('--benchmark',action='store_true', default=False,
                        help='Get benchmark performance of quantized model.')
    parser.add_argument('--benchmark_nums', type=int, default=1000,
                        help="Benchmark numbers of samples")
    parser.add_argument('--mode', type=str, default='performance',
                        choices=['performance', 'accuracy'],
                        help="Mode of benchmark")
    args = parser.parse_args()
    tokenizer = DistilBertTokenizer.from_pretrained(args.input_dir, do_lower_case=True)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, \
        batch_size=args.eval_batch_size)

    def eval_func(model):
        return evaluate_onnxrt(args, model, tokenizer, eval_dataloader)

    if args.benchmark and args.mode == "performance":
        model = onnx.load(args.model_path)
        
        from lpot.experimental.data.datasets.dummy_dataset import DummyDataset
        from lpot.experimental.data.dataloaders.onnxrt_dataloader import ONNXRTDataLoader
        shapes, lows, highs = parse_dummy_input(model, args.benchmark_nums, args.max_seq_length)
        dummy_dataset = DummyDataset(shapes, low=lows, high=highs, dtype="int64")
        dummy_dataloader = ONNXRTDataLoader(dummy_dataset)
        
        from lpot.experimental import Benchmark, common
        evaluator = Benchmark(args.config)
        evaluator.b_dataloader = dummy_dataloader
        evaluator.model = common.Model(model)
        evaluator(args.mode)

    if args.benchmark and args.mode == "accuracy":
        model = onnx.load(args.model_path)
        results = evaluate_onnxrt(args, model, tokenizer, eval_dataloader)
        print("Accuracy: %.5f" % results)

    if args.tune:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
        opt_options = BertOptimizationOptions('bert')
        opt_options.enable_embed_layer_norm = False

        model_optimizer = optimizer.optimize_model(
            args.model_path,
            'bert',
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options)
        model = model_optimizer.model

        from lpot.experimental import Quantization, common
        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        quantize.calib_dataloader = eval_dataloader
        quantize.eval_func = eval_func
        q_model = quantize()
        q_model.save(args.output_model)