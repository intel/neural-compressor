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

import re
import string
from collections import Counter, abc
from typing import Any, Callable, Dict, List, TypeVar

from neural_compressor.utils import logger


def normalize_answer(text: str) -> str:
    """Normalize the answer text.

    Lower text, remove punctuation, articles and extra whitespace,
    and replace other whitespace (newline, tab, etc.) to space.

    Args:
        s: The text to be normalized.

    Returns:
        The normalized text.
    """

    def _remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def _white_space_fix(text):
        return " ".join(text.split())

    def _remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def _lower(text):
        return text.lower()

    return _white_space_fix(_remove_articles(_remove_punc(_lower(text))))


def f1_score(prediction: abc.Sequence, ground_truth: abc.Sequence):
    """Calculate the F1 score of the prediction and the ground_truth.

    Args:
        prediction: the predicted answer.
        ground_truth: the correct answer.

    Returns:
        The F1 score of prediction. Float point number.
    """
    assert isinstance(prediction, abc.Sequence) and isinstance(
        ground_truth, abc.Sequence
    ), "prediction and ground_truth should be Sequence"
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


T = TypeVar("T")


def metric_max_over_ground_truths(
    metric_fn: Callable[[T, T], float], prediction: str, ground_truths: List[str]
) -> float:
    """Calculate the max metric for each ground truth.

    For each answer in ground_truths, evaluate the metric of prediction with
    this answer, and return the max metric.

    Args:
        metric_fn: the function to calculate the metric.
        prediction: the prediction result.
        ground_truths: the list of correct answers.

    Returns:
        The max metric. Float point number.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        score = metric_fn(prediction_tokens, ground_truth_tokens)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(predictions: Dict[str, str], dataset: List[Dict[str, Any]]) -> float:
    """Evaluate the average F1 score of Question-Answering results.

    The F1 score is the harmonic mean of the precision and recall. It can be computed
    with the equation: F1 = 2 * (precision * recall) / (precision + recall).
    For all question-and-answers in dataset, it evaluates the f1-score

    Args:
        predictions: The result of predictions to be evaluated. A dict mapping the id of
          a question to the predicted answer of the question.
        dataset: The dataset to evaluate the prediction. A list instance of articles.
          An article contains a list of paragraphs, a paragraph contains a list of
          question-and-answers (qas), and a question-and-answer contains an id, a question,
          and a list of correct answers. For example:

          [{'paragraphs':
                [{'qas':[{'answers': [{'answer_start': 177, 'text': 'Denver Broncos'}, ...],
                          'question': 'Which NFL team represented the AFC at Super Bowl 50?',
                          'id': '56be4db0acb8001400a502ec'}]}]}]

    Returns:
        The F1 score of this prediction. Float point number in forms of a percentage.
    """
    f1 = total = 0
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    logger.warning(message)
                    continue

                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]

                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    f1 = 100.0 * f1 / total
    return f1
