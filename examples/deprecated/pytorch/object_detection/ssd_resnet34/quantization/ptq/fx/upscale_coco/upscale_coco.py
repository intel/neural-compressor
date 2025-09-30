import argparse
import json
import os
import time

import cv2
import numpy as np
from coco import COCO


def parse_args():
    parser = argparse.ArgumentParser(description="Upscale COCO dataset")
    parser.add_argument('--inputs', '-i', type=str, default='/coco',
                        help='input directory for coco dataset')
    parser.add_argument('--outputs', '-o', type=str, default='/cocoup',
                        help='output directory for upscaled coco dataset')
    parser.add_argument('--images', '-im', type=str, default='val2017',
                        help='image directory')
    parser.add_argument('--annotations', '-a', type=str, default='annotations/instances_val2017.json',
                        help='annotations directory')
    parser.add_argument('--size', required=True, type=int, nargs='+',
                        help='upscaled image sizes (e.g 300 300, 1200 1200')
    parser.add_argument('--format', '-f', type=str, default='jpg',
                        help='image format')
    return parser.parse_args()


def upscale_coco(indir, outdir, image_dir, annotate_file, size, fmt):
    # Build directories.
    print('Building directories...')
    size = tuple(size)
    image_in_path = os.path.join(indir, image_dir)
    image_out_path = os.path.join(outdir, image_dir)
    path, fil = os.path.split(annotate_file)
    annotate_out_path = os.path.join(outdir, path)
    annotate_out_file = os.path.join(annotate_out_path, fil)
    annotate_in_file = os.path.join(indir, annotate_file)
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path)
    if not os.path.exists(annotate_out_path):
        os.makedirs(annotate_out_path)

    # Read annotations.
    print('Reading COCO dataset...')
    coco = COCO(annotate_in_file)
    print(len(coco.imgs), 'images')
    print(len(coco.anns), 'annotations')

    # Upscale annotations.
    print('Upscaling annotations...')
    annotations = []
    for idx in coco.anns:
        ann = coco.anns[idx]
        # Scaling factors
        img = coco.imgs[ann['image_id']]
        sx = size[0] / img['width']
        sy = size[1] / img['height']
        # Bounding boxes
        bb = ann['bbox']
        bb[0] = bb[0] * sx
        bb[1] = bb[1] * sy
        bb[2] = bb[2] * sx
        bb[3] = bb[3] * sy
        # Area
        ann['area'] = ann['area'] * sx * sy
        annotations.append(ann)

    # Upscale images.
    print('Upscaling images...')
    count = 0
    images = []
    for idx in coco.imgs:
        img = coco.imgs[idx]
        # Load, upscale, and save image.
        image = cv2.imread(os.path.join(image_in_path, img['file_name']))
        if len(image.shape) < 3 or image.shape[2] != 3:
            # some images might be grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(image_out_path, img['file_name'][0:-3] + fmt), image)
        # Update image file extension
        img['file_name'] = img['file_name'][0:-3] + fmt
        # Image dimensions
        img['width'] = size[0]
        img['height'] = size[1]
        count = count + 1
        # print(count, end=' ', flush=True)
        images.append(img)

    # Save annotations.
    print('Saving annotations...')
    with open(annotate_in_file) as f:
        dataset = json.load(f)
    dataset['images'] = images
    dataset['annotations'] = annotations
    with open(annotate_out_file, 'w') as outfile:
        json.dump(dataset, outfile)
    print('Done.')


def main():
    # Get arguments.
    args = parse_args()
    # Upscale coco.
    upscale_coco(args.inputs, args.outputs, args.images, args.annotations, args.size, args.format)


main()
