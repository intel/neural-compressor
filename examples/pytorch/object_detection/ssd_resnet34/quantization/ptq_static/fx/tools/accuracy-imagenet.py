"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as the images in imagenet2012/val_map.txt.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json

import numpy as np


# pylint: disable=missing-docstring

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--imagenet-val-file", required=True, help="path to imagenet val_map.txt")
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    parser.add_argument("--dtype", default="float32", choices=["float32", "int32", "int64"], help="data type of the label")
    args = parser.parse_args()
    return args

dtype_map = {
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64
}

def main():
    args = get_args()

    imagenet = []
    with open(args.imagenet_val_file, "r") as f:
        for line in f:
            cols = line.strip().split()
            imagenet.append((cols[0], int(cols[1])))

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    seen = set()
    good = 0
    for j in results:
        idx = j['qsl_idx']

        # de-dupe in case loadgen sends the same image multiple times
        if idx in seen:
            continue
        seen.add(idx)

        # get the expected label and image
        img, label = imagenet[idx]

        # reconstruct label from mlperf accuracy log
        data = np.frombuffer(bytes.fromhex(j['data']), dtype_map[args.dtype])
        found = int(data[0])
        if label == found:
            good += 1
        else:
            if args.verbose:
                print("{}, expected: {}, found {}".format(img, label, found))

    print("accuracy={:.3f}%, good={}, total={}".format(100. * good / len(seen), good, len(seen)))
    if args.verbose:
        print("found and ignored {} dupes".format(len(results) - len(seen)))


if __name__ == "__main__":
    main()
