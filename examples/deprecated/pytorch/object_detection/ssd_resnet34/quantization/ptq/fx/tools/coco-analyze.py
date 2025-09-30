"""
coco result analyzer
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import collections
import json
import os
import time
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageFont, ImageDraw, ImageColor


# pylint: disable=missing-docstring


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--input", required=True, help="input json")
    parser.add_argument("--coco", help="coco dataset root")
    args = parser.parse_args()
    return args


def annotate_image(results, cocoGt, output):
    os.makedirs(output, exist_ok=True)

    new_results = collections.defaultdict(list)
    for result in results:
        new_results[result['image_id']].append(result)
    print("Unique images = {}".format(len(new_results)))
    results = new_results

    for k, result in results.items():
        draw = None
        image = None
        for v in result:
            box = v['bbox']
            score = v['score']
            predicted_class = v["category_id"]
            try:
                predicted_class = cocoGt.loadCats(predicted_class)[0]["name"]
            except Exception as ex:
                print("category {} not found, image {}".format(predicted_class, v["image_loc"]))
            # predicted_class = self.class_names[c]
            # "image_loc": "/home/gs/data/coco300/val2017/000000397133.jpg",
            if not draw:
                image = Image.open(v['image_loc'])
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                draw = ImageDraw.Draw(image)
            # font = ImageFont.truetype(font='FreeMono.ttf',
            #            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            try:
                left, top, w, h = box
                bottom = top + h
                right = left + w
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                label = '{} {:.2f}'.format(predicted_class, score)
                # label_size = draw.textsize(label, font)
                label_size = draw.textsize(label)

                if top - label_size[1] >= 0:
                    text_origin = tuple(np.array([left, top - label_size[1]]))
                else:
                    text_origin = tuple(np.array([left, top + 1]))

                color = ImageColor.getrgb("red")
                thickness = 0
                draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
                draw.text(text_origin, label, fill=color)  # , font=font)
            except Exception as ex:
                print("{} failed, ex {}".format(v['image_loc'], ex))
        image.save(os.path.join(output, os.path.basename(v['image_loc'])))
        del draw


def calculate_map(results, cocoGt, output):
    # bbox is expected as:
    # x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]

    cocoDt = cocoGt.loadRes(results)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    all_metrics = {
        "DetectionBoxes_Precision/mAP": cocoEval.stats[0],
        "DetectionBoxes_Precision/mAP@.50IOU": cocoEval.stats[1],
        "DetectionBoxes_Precision/mAP@.75IOU": cocoEval.stats[2],
        "DetectionBoxes_Precision/mAP (small)": cocoEval.stats[3],
        "DetectionBoxes_Precision/mAP (medium)": cocoEval.stats[4],
        "DetectionBoxes_Precision/mAP (large)": cocoEval.stats[5],
        "DetectionBoxes_Recall/AR@1": cocoEval.stats[6],
        "DetectionBoxes_Recall/AR@10": cocoEval.stats[7],
        "DetectionBoxes_Recall/AR@100": cocoEval.stats[8],
        "DetectionBoxes_Recall/AR@100 (small)": cocoEval.stats[9],
        "DetectionBoxes_Recall/AR@100 (medium)": cocoEval.stats[10],
        "DetectionBoxes_Recall/AR@100 (large)": cocoEval.stats[11]
    }

    mAP = all_metrics['DetectionBoxes_Precision/mAP']
    recall = all_metrics['DetectionBoxes_Recall/AR@100']
    print("mAP={}, recall={}".format(mAP, recall))


def main():
    args = get_args()

    with open(args.input, "r") as f:
        results = json.load(f)

    annotation_file = os.path.join(args.coco, "annotations/instances_val2017.json")
    cocoGt = COCO(annotation_file)
    annotate_image(results, cocoGt, args.output)
    calculate_map(args.input, cocoGt, args.output)


if __name__ == "__main__":
    main()
