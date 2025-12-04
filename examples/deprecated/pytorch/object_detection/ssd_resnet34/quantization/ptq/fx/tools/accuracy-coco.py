"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's annotations/instances_val2017.json.
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import json
import os

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# pylint: disable=missing-docstring

def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--coco-dir", required=True, help="coco directory")
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    parser.add_argument("--output-file", default="coco-results.json", help="path to output file")
    parser.add_argument("--use-inv-map", action="store_true", help="use inverse label map")
    parser.add_argument("--remove-48-empty-images", action="store_true", help="used in case you removed 48 empty images while preprocessing the dataset")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    cocoGt = COCO(os.path.join(args.coco_dir, "annotations/instances_val2017.json"))

    if args.use_inv_map:
        inv_map = [0] + cocoGt.getCatIds() # First label in inv_map is not used

    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    detections = []
    image_ids = set()
    seen = set()
    no_results = 0
    if args.remove_48_empty_images:        
        im_ids = []
        for i in cocoGt.getCatIds():
            im_ids += cocoGt.catToImgs[i]
        im_ids = list(set(im_ids))
        image_map = [cocoGt.imgs[id] for id in im_ids]
    else:
        image_map = cocoGt.dataset["images"]

    for j in results:
        idx = j['qsl_idx']
        # de-dupe in case loadgen sends the same image multiple times
        if idx in seen:
            continue
        seen.add(idx)

        # reconstruct from mlperf accuracy log
        # what is written by the benchmark is an array of float32's:
        # id, box[0], box[1], box[2], box[3], score, detection_class
        # note that id is a index into instances_val2017.json, not the actual image_id
        data = np.frombuffer(bytes.fromhex(j['data']), np.float32)
        if len(data) < 7:
            # handle images that had no results
            image = image_map[idx]
            # by adding the id to image_ids we make pycoco aware of the no-result image
            image_ids.add(image["id"])
            no_results += 1
            if args.verbose:
                print("no results: {}, idx={}".format(image["coco_url"], idx))
            continue

        for i in range(0, len(data), 7):
            image_idx, ymin, xmin, ymax, xmax, score, label = data[i:i + 7]
            image = image_map[idx]
            image_idx = int(image_idx)
            if image_idx != idx:
                print("ERROR: loadgen({}) and payload({}) disagree on image_idx".format(idx, image_idx))
            image_id = image["id"]
            height, width = image["height"], image["width"]
            ymin *= height
            xmin *= width
            ymax *= height
            xmax *= width
            loc = os.path.join(args.coco_dir, "val2017", image["file_name"])
            label = int(label)
            if args.use_inv_map:
                label = inv_map[label]
            # pycoco wants {imageID,x1,y1,w,h,score,class}
            detections.append({
                "image_id": image_id,
                "image_loc": loc,
                "category_id": label,
                "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
                "score": float(score)})
            image_ids.add(image_id)

    with open(args.output_file, "w") as fp:
        json.dump(detections, fp, sort_keys=True, indent=4)

    cocoDt = cocoGt.loadRes(args.output_file) # Load from file to bypass error with Python3
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.params.imgIds = list(image_ids)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("mAP={:.3f}%".format(100. * cocoEval.stats[0]))
    if args.verbose:
        print("found {} results".format(len(results)))
        print("found {} images".format(len(image_ids)))
        print("found {} images with no results".format(no_results))
        print("ignored {} dupes".format(len(results) - len(seen)))


if __name__ == "__main__":
    main()
