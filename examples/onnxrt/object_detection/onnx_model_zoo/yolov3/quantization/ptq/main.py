# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint:disable=redefined-outer-name,logging-format-interpolation


import logging
import argparse

import onnx
from PIL import Image
import math
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)
logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--data_path',
    type=str,
    help="Path of COCO dataset, it contains val2017 and annotations subfolder"
)
parser.add_argument(
    '--model_path',
    type=str,
    help="Pre-trained model on onnx file"
)
parser.add_argument(
    '--benchmark',
    action='store_true', \
    default=False
)
parser.add_argument(
    '--tune',
    action='store_true', \
    default=False,
    help="whether quantize the model"
)
parser.add_argument(
    '--config',
    type=str,
    help="config yaml path"
)
parser.add_argument(
    '--output_model',
    type=str,
    help="output model path"
)
parser.add_argument(
    '--mode',
    type=str,
    help="benchmark mode of performance or accuracy"
)
args = parser.parse_args()

class Dataloader:
    def __init__(self, root, size=416, img_dir='val2017', \
            anno_dir='annotations/instances_val2017.json'):
        import json
        import os
        import numpy as np
        from pycocotools.coco import COCO
        from neural_compressor.experimental.metric.coco_label_map import category_map
        self.batch_size = 1
        self.image_list = []
        self.model_image_size = (size, size)
        img_path = os.path.join(root, img_dir)
        anno_path = os.path.join(root, anno_dir)
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()
        for idx, img_id in enumerate(img_ids):
            img_info = {}
            bboxes = []
            labels = []
            ids = []
            img_detail = coco.loadImgs(img_id)[0]
            ids.append(img_detail['file_name'].encode('utf-8'))
            pic_height = img_detail['height']
            pic_width = img_detail['width']

            ann_ids = coco.getAnnIds(imgIds=img_id,catIds=cat_ids)
            anns = coco.loadAnns(ann_ids)
            for ann in anns:
                bbox = ann['bbox']
                if len(bbox) == 0:
                    continue
                bboxes.append([bbox[1], bbox[0], bbox[1]+bbox[3], bbox[2]+bbox[0]])
                labels.append(category_map[ann['category_id']].encode('utf8'))
            img_file = os.path.join(img_path, img_detail['file_name'])
            if not os.path.exists(img_file) or len(bboxes) == 0:
                continue

            if filter and not filter(None, bboxes):
                continue
            label = [np.array([bboxes]), np.array([labels]), np.zeros((1,0)), np.array([img_detail['file_name'].encode('utf-8')])]
            with Image.open(img_file) as image:
                image = image.convert('RGB')
                data, label = self.preprocess((image, label))
            self.image_list.append((data, label))

    def __iter__(self):
        for item in self.image_list:
            yield item

    def letterbox_image(self, image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    def preprocess(self, sample):
        image, label = sample
        boxed_image = self.letterbox_image(image, tuple(reversed(self.model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
        return (image_data, image_size), label
    

class Post:
    def __call__(self, sample):
        preds, labels = sample
        boxes, scores, indices = preds
        out_boxes, out_scores, out_classes = [], [], []
        if len(indices) == 0:
            return ([np.zeros((0,4))], [[]], [[]]), labels
        for idx_ in indices:
            out_classes.append(idx_[1])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])
        return ([out_boxes], [out_scores], [out_classes]), labels

if __name__ == "__main__":
    model = onnx.load(args.model_path)
    dataloader = Dataloader(args.data_path)
    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        evaluator.b_dataloader = dataloader
        evaluator.postprocess = common.Postprocess(Post)
        evaluator(args.mode)

    if args.tune:
        from neural_compressor import options
        from neural_compressor.experimental import Quantization, common
        options.onnxrt.graph_optimization.level = 'ENABLE_BASIC'

        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        quantize.eval_dataloader = dataloader
        quantize.calib_dataloader = dataloader
        quantize.postprocess = common.Postprocess(Post)
        q_model = quantize()
        q_model.save(args.output_model)
        
