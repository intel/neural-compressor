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
import os
import onnxruntime as ort
import yaml

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
    help="Path to val2017 folder"
)
parser.add_argument(
    '--label_path',
    type=str,
    default='label_map.yaml',
    help="Path of label map yaml file"
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
parser.add_argument(
    '--quant_format',
    type=str,
    choices=['QOperator', 'QDQ'],
    help="quantization format"
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help="quantization format"
)
args = parser.parse_args()

class Dataloader:
    def __init__(self, root, batch_size=1, size=416, \
            anno_dir='annotations/instances_val2017.json'):
        import json
        import os
        import numpy as np
        from pycocotools.coco import COCO
        from coco_label_map import category_map
        self.batch_size = batch_size
        self.image_list = []
        self.model_image_size = (size, size)
        img_path = root
        anno_path = os.path.join(os.path.dirname(root), anno_dir)
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
    
class COCOmAPv2():
    """Compute mean average precision of the detection task."""

    def __init__(self, 
                 anno_path=None, 
                 iou_thrs='0.5:0.05:0.95', 
                 map_points=101, 
                 map_key='DetectionBoxes_Precision/mAP', 
                 output_index_mapping={'num_detections':-1, 'boxes':0, 'scores':1, 'classes':2}):
        """Initialize the metric.
        Args:
            anno_path: The path of annotation file.
            iou_thrs: Minimal value for intersection over union that allows to make decision
              that prediction bounding box is true positive. You can specify one float value
              between 0 to 1 or string "05:0.05:0.95" for standard COCO thresholds.
            map_points: The way to calculate mAP. 101 for 101-point interpolated AP, 11 for 
              11-point interpolated AP, 0 for area under PR curve.
            map_key: The key that mapping to pycocotools COCOeval. 
              Defaults to 'DetectionBoxes_Precision/mAP'.
            output_index_mapping: The output index mapping. 
              Defaults to {'num_detections':-1, 'boxes':0, 'scores':1, 'classes':2}.
        """
        self.output_index_mapping = output_index_mapping
        from coco_label_map import category_map
        if anno_path:
            assert os.path.exists(anno_path), 'Annotation path does not exists!'
            with open(anno_path, 'r') as f:
                label_map = yaml.safe_load(f.read())
            self.category_map_reverse = {k: v for k,v in label_map.items()}
        else:
            # label: index
            self.category_map_reverse = {v: k for k, v in category_map.items()}
        self.image_ids = []
        self.ground_truth_list = []
        self.detection_list = []
        self.annotation_id = 1
        self.category_map = category_map
        self.category_id_set = set(
            [cat for cat in self.category_map]) #index
        self.iou_thrs = iou_thrs
        self.map_points = map_points
        self.map_key = map_key

    def update(self, predicts, labels, sample_weight=None):
        """Add the predictions and labels.
        Args:
            predicts: The predictions.
            labels: The labels corresponding to the predictions.
            sample_weight: The sample weight. Defaults to None.
        """
        from coco_tools import ExportSingleImageGroundtruthToCoco,\
            ExportSingleImageDetectionBoxesToCoco
        detections = []
        if 'num_detections' in self.output_index_mapping and \
            self.output_index_mapping['num_detections'] > -1:
            for item in zip(*predicts):
                detection = {}
                num = int(item[self.output_index_mapping['num_detections']])
                detection['boxes'] = np.asarray(
                    item[self.output_index_mapping['boxes']])[0:num]
                detection['scores'] = np.asarray(
                    item[self.output_index_mapping['scores']])[0:num]
                detection['classes'] = np.asarray(
                    item[self.output_index_mapping['classes']])[0:num]
                detections.append(detection)
        else:
            for item in zip(*predicts):
                detection = {}
                detection['boxes'] = np.asarray(item[self.output_index_mapping['boxes']])
                detection['scores'] = np.asarray(item[self.output_index_mapping['scores']])
                detection['classes'] = np.asarray(item[self.output_index_mapping['classes']])
                detections.append(detection)

        bboxes, str_labels,int_labels, image_ids = labels
        labels = []
        if len(int_labels[0]) == 0:
            for str_label in str_labels:
                str_label = [
                    x if type(x) == 'str' else x.decode('utf-8')
                    for x in str_label
                ]
                labels.append([self.category_map_reverse[x] for x in str_label])
        elif len(str_labels[0]) == 0:
            for int_label in int_labels:
                labels.append([x for x in int_label])

        for idx, image_id in enumerate(image_ids):
            image_id = image_id if type(
                image_id) == 'str' else image_id.decode('utf-8')
            if image_id in self.image_ids:
                continue
            self.image_ids.append(image_id)

            ground_truth = {}
            ground_truth['boxes'] = np.asarray(bboxes[idx])
            ground_truth['classes'] = np.asarray(labels[idx])

            self.ground_truth_list.extend(
                ExportSingleImageGroundtruthToCoco(
                    image_id=image_id,
                    next_annotation_id=self.annotation_id,
                    category_id_set=self.category_id_set,
                    groundtruth_boxes=ground_truth['boxes'],
                    groundtruth_classes=ground_truth['classes']))
            self.annotation_id += ground_truth['boxes'].shape[0]

            self.detection_list.extend(
                ExportSingleImageDetectionBoxesToCoco(
                    image_id=image_id,
                    category_id_set=self.category_id_set,
                    detection_boxes=detections[idx]['boxes'],
                    detection_scores=detections[idx]['scores'],
                    detection_classes=detections[idx]['classes']))

    def reset(self):
        """Reset the prediction and labels."""
        self.image_ids = []
        self.ground_truth_list = []
        self.detection_list = []
        self.annotation_id = 1

    def result(self):
        """Compute mean average precision.
        Returns:
            The mean average precision score.
        """
        from coco_tools import COCOWrapper, COCOEvalWrapper
        if len(self.ground_truth_list) == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        else:
            groundtruth_dict = {
                'annotations':
                self.ground_truth_list,
                'images': [{
                    'id': image_id
                } for image_id in self.image_ids],
                'categories': [{
                    'id': k,
                    'name': v
                } for k, v in self.category_map.items()]
            }
            coco_wrapped_groundtruth = COCOWrapper(groundtruth_dict)
            coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(
                self.detection_list)
            box_evaluator = COCOEvalWrapper(coco_wrapped_groundtruth,
                                                 coco_wrapped_detections,
                                                 agnostic_mode=False,
                                                 iou_thrs = self.iou_thrs,
                                                 map_points = self.map_points)
            box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
                include_metrics_per_category=False, all_metrics_per_category=False)
            box_metrics.update(box_per_category_ap)
            box_metrics = {
                'DetectionBoxes_' + key: value
                for key, value in iter(box_metrics.items())
            }

            return box_metrics[self.map_key]

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
    dataloader = Dataloader(args.data_path, batch_size=args.batch_size)
    metric = COCOmAPv2(anno_path=args.label_path, output_index_mapping={'boxes':0, 'scores':1, 'classes':2})
    postprocess = Post()

    def eval_func(model):
        metric.reset()
        session = ort.InferenceSession(model.SerializeToString(), 
                                       providers=ort.get_available_providers())
        ort_inputs = {}
        len_inputs = len(session.get_inputs())
        inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
        for idx, (inputs, labels) in enumerate(dataloader):
                if not isinstance(labels, list):
                    labels = [labels]
                if len_inputs == 1:
                    ort_inputs.update(
                        inputs if isinstance(inputs, dict) else {inputs_names[0]: inputs}
                    )
                else:
                    assert len_inputs == len(inputs), 'number of input tensors must align with graph inputs'
                    if isinstance(inputs, dict):
                        ort_inputs.update(inputs)
                    else:
                        for i in range(len_inputs):
                            if not isinstance(inputs[i], np.ndarray):
                                ort_inputs.update({inputs_names[i]: np.array(inputs[i])})
                            else:
                                ort_inputs.update({inputs_names[i]: inputs[i]})
                predictions = session.run(None, ort_inputs)
                predictions, labels = postprocess((predictions, labels))
                metric.update(predictions, labels)
        return metric.result()

    if args.benchmark:
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(iteration=100,
                                   cores_per_instance=4,
                                   num_of_instance=1)
            fit(model, conf, b_dataloader=dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval_func(model)
            print("Batch size = %d" % args.batch_size)
            print("Accuracy: %.5f" % acc_result)

    if args.tune:
        from neural_compressor import quantization
        from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig
        accuracy_criterion = AccuracyCriterion()
        accuracy_criterion.absolute = 0.02
        config = PostTrainingQuantConfig(approach='static', 
                                         quant_format=args.quant_format,
                                         accuracy_criterion=accuracy_criterion,
                                         recipes={'first_conv_or_matmul_quantization': False,
                                                  'last_conv_or_matmul_quantization': False,
                                                  'pre_post_process_quantization': False})
        q_model = quantization.fit(model, config, calib_dataloader=dataloader, eval_func=eval_func)
        q_model.save(args.output_model)
