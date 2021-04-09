#!/usr/bin/env python

import argparse
import array
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pytorch"))

from QSL import AudioQSL
from helpers import process_evaluation_epoch, __gather_predictions
from parts.manifest import Manifest

dtype_map = {
    "int8": 'b',
    "int16": 'h',
    "int32": 'l',
    "int64": 'q',
}

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
    labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
    qsl = AudioQSL(args.dataset_dir, args.manifest, labels)
    manifest = qsl.manifest
    with open(os.path.join(args.log_dir, "mlperf_log_accuracy.json")) as fh:
        results = json.load(fh)
    hypotheses = []
    references = []
    for result in results:
        hypotheses.append(array.array(dtype_map[args.output_dtype], bytes.fromhex(result["data"])).tolist())
        references.append(manifest[result["qsl_idx"]]["transcript"])

    references = __gather_predictions([references], labels=labels)
    hypotheses = __gather_predictions([hypotheses], labels=labels)

    d = dict(predictions=hypotheses,
             transcripts=references)
    wer = process_evaluation_epoch(d)
    print("Word Error Rate: {:}%, accuracy={:}%".format(wer * 100, (1 - wer) * 100))

if __name__ == '__main__':
    main()
