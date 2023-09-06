#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Script to compute BLEU score.

Source:
https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
"""

from __future__ import absolute_import, division, print_function

import collections
import math
from typing import List, Sequence, Union

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from neural_compressor.utils.utility import LazyImport

tf = LazyImport("tensorflow")


def _get_ngrams_with_counter(segment: Sequence[str], max_order: List[int]) -> collections.Counter:
    """Extract all n-grams up to a given maximum order from an input segment.

    Args:
        segment: The text segment from which n-grams will be extracted.
        max_order: The maximum length in tokens of the n-grams returned
          by this methods.

    Returns:
        ngram_counts: The Counter containing all n-grams up to max_order
          in segment with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in xrange(1, max_order + 1):
        for i in xrange(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(
    reference_corpus: Union[Sequence[str], Sequence[Sequence[str]]],
    translation_corpus: Sequence[str],
    max_order: int = 4,
    use_bp: bool = True,
) -> float:
    """Compute the BLEU score of translated segments against its references.

    Args:
        reference_corpus: List of references for each translation.
          Each reference should be tokenized into a list of tokens.
        translation_corpus: List of translations to score. Each translation
          should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        use_bp: The flag to decide whether to apply brevity penalty.

    Returns:
        bleu_score: The approximate BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []

    for references, translations in zip(reference_corpus, translation_corpus):
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

        overlap = dict(
            (ngram, min(count, translation_ngram_counts[ngram])) for ngram, count in ref_ngram_counts.items()
        )

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[ngram]

    precisions = [0] * max_order
    smooth = 1.0

    for i in xrange(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        bp = math.exp(1 - 1.0 / ratio) if ratio < 1.0 else 1.0
    bleu_score = np.float32(geo_mean * bp)
    return bleu_score
