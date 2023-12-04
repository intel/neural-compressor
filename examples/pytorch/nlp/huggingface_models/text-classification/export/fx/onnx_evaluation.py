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

from cProfile import label
import logging
import argparse
import onnx
import onnxruntime as ort
import transformers
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

class ONNXRTBertDataset:
    def __init__(self, task, model_name_or_path, max_seq_length=128, data_dir=None):
        raw_dataset = load_dataset('glue', task, cache_dir=data_dir, split='validation')
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        sentence1_key, sentence2_key = task_to_keys[task]
        origin_keys = raw_dataset[0].keys()

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding="max_length", max_length=max_seq_length, truncation=True)
            if  "label" in examples:
                result["label"] = examples["label"]
            return result

        self.dataset = raw_dataset.map(
            preprocess_function, batched=True, load_from_cache_file=True, remove_columns=origin_keys
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        batch = {k: np.asarray(v) for k, v in self.dataset[index].items()}
        label = batch.pop('label')
        return batch, label


class INCDataloader():
    def __init__(self, dataset, batch_size=1):
        import math
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = math.ceil(len(self.dataset) // self.batch_size)
        self.example_input = self.dataset[0][0]

    def __iter__(self):
        batched_input = {k: None for k in self.example_input}
        batched_label = None
        for idx, (input, label) in enumerate(self.dataset):
            label = np.expand_dims(label, axis=0)
            for k, v in input.items():
                v = np.expand_dims(v, axis=0)
                if batched_input[k] is None:
                    batched_input[k] = v
                else:
                    batched_input[k] = np.append(batched_input[k], v, axis=0)
            if batched_label is None:
                batched_label = label
            else:
                batched_label = np.append(batched_label, label, axis=0)
            if (idx+1) % self.batch_size == 0:
                yield batched_input, batched_label
                batched_input = {k: None for k in self.example_input}
                batched_label = None
        if (idx+1) % self.batch_size != 0:
            yield batched_input, batched_label

    def __len__(self):
        return self.length

class ONNXRTGLUE:
    """Computes GLUE score.

    Args:
        task (str, default=mrpc): The name of the task.
                                  Choices include mrpc, qqp, qnli, rte,
                                  sts-b, cola, mnli, wnli.

    """
    def __init__(self, task='mrpc'):
        assert task in ['mrpc', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', \
            'mnli', 'wnli', 'sst2'], 'Unsupported task type'
        self.pred_list = None
        self.label_list = None
        self.task = task
        self.return_key = {
            "cola": "mcc",
            "mrpc": "f1",
            "sts-b": "corr",
            "qqp": "acc",
            "mnli": "mnli/acc",
            "qnli": "acc",
            "rte": "acc",
            "wnli": "acc",
            "sst2": "acc"
        }

    def update(self, preds, labels):
        if self.pred_list is None:
            self.pred_list = preds
            self.label_list = labels
        else:
            self.pred_list = np.append(self.pred_list, preds, axis=0)
            self.label_list = np.append(self.label_list, labels, axis=0)

    def reset(self):
        """clear preds and labels storage"""
        self.pred_list = None
        self.label_list = None

    def result(self):
        """calculate metric"""
        output_mode = transformers.glue_output_modes[self.task]

        if output_mode == "classification":
            processed_preds = np.argmax(self.pred_list, axis=1)
        elif output_mode == "regression":
            processed_preds = np.squeeze(self.pred_list)
        result = transformers.glue_compute_metrics(\
            self.task, processed_preds, self.label_list)
        return result[self.return_key[self.task]]

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

if __name__ == "__main__":
    logger.info('Evaluating ONNXRuntime full precision accuracy and performance:')
    parser = argparse.ArgumentParser(
    description='BERT fine-tune examples for classification/regression tasks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained resnet50 model on onnx file"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true', \
        default=False, \
        help="benchmark mode of performance"
    )
    parser.add_argument(
       '--config',
       type=str,
       help="config yaml path"
    )
    parser.add_argument(
        '--output_model',
        type=str,
        default=None,
        help="output model path"
    )
    parser.add_argument(
        '--accuracy',
        action='store_true', \
        default=False, \
        help="benchmark mode of accuracy"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help="input data path"
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help="pretrained model name or path"
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['mrpc', 'qqp', 'qnli', 'rte', 'sts-b', 'cola', \
                'mnli', 'wnli', 'sst2'],
        help="GLUE task name"
    )
    parser.add_argument(
        '--max_seq_length',
        default=128,
        type=int,
    )
 
    args = parser.parse_args()

    dataset = ONNXRTBertDataset(task=args.task,
                                model_name_or_path=args.model_name_or_path,
                                max_seq_length =args.max_seq_length)
    dataloader = INCDataloader(dataset, args.batch_size)
    metric = ONNXRTGLUE(args.task)

    def eval_func(model):
        metric.reset()
        from tqdm import tqdm
        session = ort.InferenceSession(model.SerializeToString(), None)
        for inputs, labels in tqdm(dataloader):
            predictions = session.run(None, inputs)
            metric.update(predictions[0], labels)
        return metric.result()

    model = onnx.load(args.model_path)
    if args.benchmark:
        from neural_compressor.benchmark import fit
        from neural_compressor.config import BenchmarkConfig
        conf = BenchmarkConfig(iteration=100,
                                cores_per_instance=4,
                                num_of_instance=1)
        fit(model, conf, b_dataloader=dataloader)
    elif args.accuracy:
        acc_result = eval_func(model)
        print("Batch size = %d" % args.batch_size)
        print("Accuracy: %.5f" % acc_result)
