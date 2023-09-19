# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from metrics import word_error_rate


class Optimization(Enum):
    """Various levels of Optimization.
    WARNING: This might have effect on model accuracy."""
    nothing = 0
    mxprO0 = 1
    mxprO1 = 2
    mxprO2 = 3
    mxprO3 = 4


AmpOptimizations = {Optimization.mxprO0: "O0",
                    Optimization.mxprO1: "O1",
                    Optimization.mxprO2: "O2",
                    Optimization.mxprO3: "O3"}


def add_blank_label(labels):
    if not isinstance(labels, list):
        raise ValueError("labels must be a list of symbols")
    labels.append("<BLANK>")
    return labels


def __rnnt_decoder_predictions_tensor(tensor, labels):
    """
    Takes output of greedy rnnt decoder and converts to strings.
    Args:
        tensor: model output tensor
        label: A list of labels
    Returns:
        prediction
    """
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over batch
    for ind in range(len(tensor)):
        hypothesis = ''.join([labels_map[c] for c in tensor[ind]])
        hypotheses.append(hypothesis)
    return hypotheses


def __gather_predictions(predictions_list: list, labels: list) -> list:
    results = []
    for prediction in predictions_list:
        results += __rnnt_decoder_predictions_tensor(prediction, labels=labels)
    return results


def __gather_transcripts(transcript_list: list, transcript_len_list: list,
                         labels: list) -> list:
    results = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    for i, t in enumerate(transcript_list):
        target = t.numpy().tolist()
        reference = ''.join([labels_map[c] for c in target])
        results.append(reference)
    return results


def process_evaluation_batch(tensors: dict, global_vars: dict, labels: list):
    """
    Processes results of an iteration and saves it in global_vars
    Args:
        tensors: dictionary with results of an evaluation iteration, e.g. loss, predictions, transcript, and output
        global_vars: dictionary where processes results of iteration are saved
        labels: A list of labels
    """
    for kv, v in tensors.items():
        if kv.startswith('predictions'):
            global_vars['predictions'] += __gather_predictions(
                v, labels=labels)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v

    global_vars['transcripts'] += __gather_transcripts(transcript_list,
                                                       transcript_len_list,
                                                       labels=labels)


def process_evaluation_epoch(global_vars: dict, tag=None):
    """
    Processes results from each worker at the end of evaluation and combine to final result
    Args:
        global_vars: dictionary containing information of entire evaluation
    Return:
        wer: final word error rate
        loss: final loss
    """
    hypotheses = global_vars['predictions']
    references = global_vars['transcripts']

    wer, scores, num_words = word_error_rate(
        hypotheses=hypotheses, references=references)
    return wer


def print_dict(d):
    maxLen = max([len(ii) for ii in d.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(d.items()):
        print(fmtString % keyPair)
