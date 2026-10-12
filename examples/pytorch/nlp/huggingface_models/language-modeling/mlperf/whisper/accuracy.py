# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import array
import json
import sys
import os
from typing import List

from whisper.normalizers import EnglishTextNormalizer

from manifest import Manifest
from legacy_helpers import __levenshtein, __gather_predictions
from helpers import get_expanded_wordlist


max_duration = float(os.environ.get("MAX_DURATION", "30.0"))
labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
dtype_map = {
    "int8": 'b',
    "int16": 'h',
    "int32": 'l',
    "int64": 'q',
}

def word_error_rate(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same length.

    Args:
        hypotheses: list of hypotheses
        references: list of references

    Returns:
        (float) average word error rate
    """
    normalizer = EnglishTextNormalizer()

    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError("In word error rate calculation, hypotheses and reference"
                         " lists must have the same number of elements. But I got:"
                         "{0} and {1} correspondingly".format(len(hypotheses), len(references)))
    for h, r in zip(hypotheses, references):
        h = normalizer(h)
        r = normalizer(r)
        h_list = h.split()
        r_list = r.split()
        h_list = get_expanded_wordlist(h_list, r_list)
        r_list = get_expanded_wordlist(r_list, h_list)
        words += len(r_list)
        scores += __levenshtein(h_list, r_list)
    wer = scores / words
    return wer, scores, words

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dtype", default="int64", choices=dtype_map.keys(), help="Output data type")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    manifest = Manifest(args.dataset_dir, [args.manifest], labels, len(labels), max_duration=max_duration)
    with open(os.path.join(args.log_dir, "mlperf_log_accuracy.json")) as fh:
        results = json.load(fh)
    hypotheses = []
    references = []
    for result in results:
        hypotheses.append(array.array(dtype_map[args.output_dtype], bytes.fromhex(result["data"])).tolist())
        references.append(manifest[result["qsl_idx"]]["transcript"])

    references = __gather_predictions([references], labels=labels)
    hypotheses = __gather_predictions([hypotheses], labels=labels)

    wer, _, _ = word_error_rate(hypotheses=hypotheses, references=references)
    print("Word Error Rate: {:}%, accuracy={:}%".format(wer * 100, (1 - wer) * 100))

if __name__ == '__main__':
    main()