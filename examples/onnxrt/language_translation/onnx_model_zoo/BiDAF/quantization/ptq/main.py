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

import logging
import argparse
import onnx
import json
import numpy as np
import string
from nltk import word_tokenize
import re
from collections import Counter

# answer normalization specific for squad evaluation
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def metric_max_over_ground_truths(prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = exact_match_score(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def preprocess(text):
   tokens = word_tokenize(text)
   # split into lower-case word tokens, in numpy array with shape of (seq, 1)
   words = np.asarray([w.lower() for w in tokens]).reshape(-1, 1)
   # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
   chars = [[c for c in t][:16] for t in tokens]
   chars = [cs+['']*(16-len(cs)) for cs in chars]
   chars = np.asarray(chars).reshape(-1, 1, 1, 16)
   return words, chars

class squadDataset():
    def __init__(self, datapath):
        self.batch_size = 1
        self.data = []
        with open(datapath, "r") as f:
            input_data = json.load(f)["data"]
        for idx, entry in enumerate(input_data):
            for paragraph in entry["paragraphs"]:
                ct = paragraph["context"]
                cw, cc = preprocess(ct)
                for qas in paragraph['qas']:
                    qw, qc = preprocess(qas['question'])
                    self.data.append([[cw, cc, qw, qc], [cw, [i['text'] for i in qas['answers']]]])
                            
    def __iter__(self):
        for data in self.data:
            yield data

    def __len__(self):
        return len(self.data)

class EM:
    def __init__(self):
        self.items = []
      
    def update(self, pred, label):
        start = pred[0].item()
        end = pred[1].item()
        self.items.append([' '.join([w[0] for w in label[0][start:end+1]]), label[1]])
        
    def reset(self):
        self.items = []
        
    def result(self):
        em_sum = 0
        for pred, labels in self.items:
            em = metric_max_over_ground_truths(pred, labels)
            em_sum += 1 if em else 0
        return em_sum / len(self.items)

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

if __name__ == "__main__":
    logger.info('Evaluating ONNXRuntime full precision accuracy and performance:')
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained resnet50 model on onnx file"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help="Pre-trained resnet50 model on onnx file"
    )
    parser.add_argument(
        '--benchmark',
        action='store_true', \
        default=False
    )
    parser.add_argument(
        '--tune',
        action='store_true', \
        default=False,
        help="whether quantize the model"
    )
    parser.add_argument(
        '--config',
        type=str,
        help="config yaml path"
    )
    parser.add_argument(
        '--output_model',
        type=str,
        help="output model path"
    )
    parser.add_argument(
        '--mode',
        type=str,
        help="benchmark mode of performance or accuracy"
    )
    args = parser.parse_args()
    model = onnx.load(args.model_path)
    dataloader = squadDataset(args.data_path)
 
    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        evaluator.b_dataloader = dataloader
        evaluator.metric = EM()
        evaluator(args.mode)

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        quantize.calib_dataloader = dataloader
        quantize.eval_dataloader = dataloader
        quantize.metric = EM()
        q_model = quantize()
        q_model.save(args.output_model)
