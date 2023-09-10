#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Official evaluation script for v1.1 of the SQuAD dataset.

From https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
"""

from __future__ import print_function

import sys
from collections import Counter

from .f1 import normalize_answer


def f1_score(prediction, ground_truth):
    """Calculate the F1 score of the prediction and the ground_truth.

    Args:
        prediction: The predicted result.
        ground_truth: The ground truth.

    Returns:
        The F1 score of prediction. Float point number.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Calculate the max metric for each ground truth.

    For each answer in ground_truths, evaluate the metric of prediction with
    this answer, and return the max metric.

    Args:
        metric_fn: The function to calculate the metric.
        prediction: The prediction result.
        ground_truths: A list of correct answers.

    Returns:
        The max metric. Float point number.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def exact_match_score(prediction, ground_truth):
    """Compute the exact match score between prediction and ground truth.

    Args:
        prediction: The result of predictions to be evaluated.
        ground_truth: The ground truth.

    Returns:
        The exact match score.
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def evaluate(dataset, predictions):
    """Evaluate the average F1 score and the exact match score for Question-Answering results.

    Args:
        dataset: The dataset to evaluate the prediction. A list instance of articles.
          An article contains a list of paragraphs, a paragraph contains a list of
          question-and-answers (qas), and a question-and-answer contains an id, a question,
          and a list of correct answers. For example:
        predictions: The result of predictions to be evaluated. A dict mapping the id of
          a question to the predicted answer of the question.

    Returns:
        The F1 score and the exact match score.
    """
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}
