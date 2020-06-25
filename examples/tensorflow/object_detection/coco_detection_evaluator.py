#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
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
#

import coco_tools
import coco_label_map

class CocoDetectionEvaluator:
    """Class to evaluate COCO detection metrics."""

    def __init__(self):
        self._image_ids = {}
        self._groundtruth_list = []
        self._detection_boxes_list = []
        self._annotation_id = 1
        self._category_id_set = set([cat for cat in coco_label_map.category_map])
        self._groundtruth_list = []
        self._detection_boxes_list = []

    def add_single_ground_truth_image_info(self,
                                           image_id,
                                           groundtruth_dict):
        if image_id in self._image_ids:
            return
        
        self._groundtruth_list.extend(
        coco_tools.ExportSingleImageGroundtruthToCoco(
            image_id=image_id,
            next_annotation_id=self._annotation_id,
            category_id_set=self._category_id_set,
            groundtruth_boxes=groundtruth_dict['boxes'],
            groundtruth_classes=groundtruth_dict['classes']))
        self._annotation_id += groundtruth_dict['boxes'].shape[0]
        
        self._image_ids[image_id] = False
        is_debug = False
        if image_id == '000000059386.jpg':
            is_debug = True
        if is_debug:
            is_debug = False
            print(groundtruth_dict['boxes'])
            print(groundtruth_dict['classes'])
            print(image_id)
    
    def add_single_detected_image_info(self,
                                       image_id,
                                       detections_dict):
        assert (image_id in self._image_ids)
        
        if self._image_ids[image_id]:
            return
        
        self._detection_boxes_list.extend(
            coco_tools.ExportSingleImageDetectionBoxesToCoco(
                image_id=image_id,
                category_id_set=self._category_id_set,
                detection_boxes=detections_dict['boxes'],
                detection_scores=detections_dict['scores'],
                detection_classes=detections_dict['classes']))

        self._image_ids[image_id] = True
        is_debug = False
        if image_id == '000000059386.jpg':
            is_debug = True
        if is_debug:
            is_debug = False
            print(detections_dict['boxes'])
            print(detections_dict['classes'])
            print(detections_dict['classes'])
            print(image_id)

    def evaluate(self):
        groundtruth_dict = {
        'annotations': self._groundtruth_list,
        'images': [{'id': image_id} for image_id in self._image_ids],
        'categories': [{'id': k, 'name': v} for k, v in coco_label_map.category_map.items()]
        }
        coco_wrapped_groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
        coco_wrapped_detections = coco_wrapped_groundtruth.LoadAnnotations(
            self._detection_boxes_list)
        box_evaluator = coco_tools.COCOEvalWrapper(
            coco_wrapped_groundtruth, coco_wrapped_detections, agnostic_mode=False)
        box_metrics, box_per_category_ap = box_evaluator.ComputeMetrics(
            include_metrics_per_category=False,
            all_metrics_per_category=False)
        box_metrics.update(box_per_category_ap)
        box_metrics = {'DetectionBoxes_'+ key: value
                       for key, value in iter(box_metrics.items())}
        return box_metrics
