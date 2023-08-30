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
"""Script for BLEU metric."""

import re
import sys
import unicodedata
from typing import List, Sequence

import six

from .bleu_util import compute_bleu
from .metric import metric_registry


class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols.

    Attributes:
        nondigit_punct_re: The compiled regular expressions to recognize
          punctuation preceded with a digit.
        punct_nondigit_re: The compiled regular expressions to recognize
          punctuation followed by a digit.
        symbol_re: The compiled regular expressions to recognize symbols.
    """

    def __init__(self) -> None:
        """Initialize the regular expressions."""
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix: str) -> str:
        """Collect all Unicode strings starting with a specific prefix.

        Args:
            prefix: The specific prefix.

        Returns:
            punctuation: The join result of all Unicode strings starting
              with a specific prefix.
        """
        punctuation = "".join(
            six.unichr(x) for x in range(sys.maxunicode) if unicodedata.category(six.unichr(x)).startswith(prefix)
        )
        return punctuation


uregex = UnicodeRegex()


def bleu_tokenize(string: str) -> List[str]:
    """Tokenize a string following the official BLEU implementation.

    See https://github.com/moses-smt/mosesdecoder/"
            "blob/master/scripts/generic/mteval-v14.pl#L954-L983

    Args:
        string: The string to be tokenized.

    Returns:
        tokens: A list of tokens.
    """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    tokens = string.split()
    return tokens


@metric_registry("BLEU", "tensorflow, tensorflow_itex")
class BLEU(object):
    """Computes the BLEU (Bilingual Evaluation Understudy) score.

    BLEU is an algorithm for evaluating the quality of text which has
    been machine-translated from one natural language to another.
    This implementent approximate the BLEU score since we do not
    glue word pieces or decode the ids and tokenize the output.
    By default, we use ngram order of 4 and use brevity penalty.
    Also, this does not have beam search.

    Attributes:
        predictions: List of translations to score.
        labels: List of the reference corresponding to the prediction result.
    """

    def __init__(self) -> None:
        """Initialize predictions and labels."""
        self.predictions = []
        self.labels = []

    def reset(self) -> None:
        """Clear the predictions and labels in the cache."""
        self.predictions = []
        self.labels = []

    def update(self, prediction: Sequence[str], label: Sequence[str]) -> None:
        """Add the prediction and label.

        Args:
            prediction: The prediction result.
            label: The reference corresponding to the prediction result.

        Raises:
            ValueError: An error occurred when the length of the prediction
            and label are different.
        """
        if len(label) != len(prediction):
            raise ValueError(
                "Reference and prediction files have different number "
                "of lines. If training only a few steps (100-200), the "
                "translation may be empty."
            )
        label = [x.lower() for x in label]
        prediction = [x.lower() for x in prediction]
        label = [bleu_tokenize(x) for x in label]
        prediction = [bleu_tokenize(x) for x in prediction]
        self.labels.extend(label)
        self.predictions.extend(prediction)

    def result(self) -> float:
        """Compute the BLEU score.

        Returns:
            bleu_score: The approximate BLEU score.
        """
        bleu_score = compute_bleu(self.labels, self.predictions) * 100
        return bleu_score
