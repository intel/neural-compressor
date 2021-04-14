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
""" Official evaluation script for v1.1 of the SQuAD dataset.
https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py """

import re
import six
import sys
import numpy as np
import pandas as pd
import unicodedata
from .bleu_util import compute_bleu
from .metric import metric_registry

class UnicodeRegex(object):
    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
        return "".join(six.unichr(x) for x in range(sys.maxunicode)
                    if unicodedata.category(six.unichr(x)).startswith(prefix))

uregex = UnicodeRegex()

def bleu_tokenize(string):
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()

@metric_registry('BLEU', 'tensorflow')
class BLEU(object):
    """Computes Bilingual Evaluation Understudy Score

    BLEU score computation between labels and predictions. An approximate BLEU scoring 
    method since we do not glue word pieces or decode the ids and tokenize the output. 
    By default, we use ngram order of 4 and use brevity penalty. Also, this does not 
    have beam search
    
    """
    def __init__(self):
        self.translations = []
        self.labels = []

    def reset(self):
        """clear preds and labels storage"""
        self.translations = []
        self.labels = []

    def update(self, pred, label):
        """add preds and labels to storage"""
        if len(label) != len(pred):
            raise ValueError("Reference and translation files have different number "
                             "of lines. If training only a few steps (100-200), the "
                             "translation may be empty.")
        label = [x.lower() for x in label]
        pred = [x.lower() for x in pred]
        label = [bleu_tokenize(x) for x in label]
        pred = [bleu_tokenize(x) for x in pred]
        self.labels.extend(label)
        self.translations.extend(pred)

    def result(self):
        """calculate metric"""
        return compute_bleu(self.labels, self.translations) * 100

