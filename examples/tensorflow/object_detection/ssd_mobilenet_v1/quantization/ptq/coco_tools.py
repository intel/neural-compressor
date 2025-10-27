#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Intel Corporation
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
"""Wrappers for third party pycocotools to be used within object_detection.

Note that nothing in this file is tensorflow related and thus cannot
be called directly as a slim metric, for example.

TODO(jonathanhuang): wrap as a slim metric in metrics.py


Usage example: given a set of images with ids in the list image_ids
and corresponding lists of numpy arrays encoding groundtruth (boxes and classes)
and detections (boxes, scores and classes), where elements of each list
correspond to detections/annotations of a single image,
then evaluation (in multi-class mode) can be invoked as follows:

  groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
      image_ids, groundtruth_boxes_list, groundtruth_classes_list,
      max_num_classes, output_path=None)
  detections_list = coco_tools.ExportDetectionsToCOCO(
      image_ids, detection_boxes_list, detection_scores_list,
      detection_classes_list, output_path=None)
  groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
  detections = groundtruth.LoadAnnotations(detections_list)
  evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                         agnostic_mode=False)
  metrics = evaluator.ComputeMetrics()
"""

import copy
import time
from collections import OrderedDict
from typing import Any, Dict, List, Set, Union

import numpy as np
from pycocotools import coco, cocoeval, mask

from neural_compressor.utils import logger


class COCOWrapper(coco.COCO):
    """Wrapper for the pycocotools COCO class.

    Attributes:
      dataset: a dictionary holding bounding box annotations in the COCO format.
      detection_type: type of detections being wrapped. Can be one of ['bbox',
        'segmentation']
    """

    def __init__(self, dataset: Dict[str, Any], detection_type: str = "bbox"):
        """Construct a COCOWrapper.

        See http://mscoco.org/dataset/#format for a description of the format.
        By default, the coco.COCO class constructor reads from a JSON file.
        This function duplicates the same behavior but loads from a dictionary,
        allowing us to perform evaluation without writing to external storage.

        Args:
          dataset: a dictionary holding bounding box annotations in the COCO format.
          detection_type: type of detections being wrapped. Can be one of ['bbox',
            'segmentation']

        Raises:
          ValueError: if detection_type is unsupported.
        """
        supported_detection_types = ["bbox", "segmentation"]
        if detection_type not in supported_detection_types:
            raise ValueError(
                "Unsupported detection type: {}. "
                "Supported values are: {}".format(detection_type, supported_detection_types)
            )
        self._detection_type = detection_type
        coco.COCO.__init__(self)
        self.dataset = dataset
        self.createIndex()

    def LoadAnnotations(self, annotations: list) -> coco.COCO:
        """Load annotations dictionary into COCO datastructure.

        See http://mscoco.org/dataset/#format for a description of the annotations
        format.  As above, this function replicates the default behavior of the API
        but does not require writing to external storage.

        Args:
          annotations: python list holding object detection results where each
            detection is encoded as a dict with required keys ['image_id',
            'category_id', 'score'] and one of ['bbox', 'segmentation'] based on
            `detection_type`.

        Returns:
          a coco.COCO datastructure holding object detection annotations results

        Raises:
          ValueError: if (1) annotations is not a list or annotations do not
            correspond to the images contained in self.
        """
        results = coco.COCO()
        results.dataset["images"] = [img for img in self.dataset["images"]]

        logger.info("Load and prepare annotation results.")
        tic = time.time()

        if not isinstance(annotations, list):
            raise ValueError("annotations is not a list of objects")
        annotation_img_ids = [ann["image_id"] for ann in annotations]
        if set(annotation_img_ids) != (set(annotation_img_ids) & set(self.getImgIds())):
            raise ValueError("Results do not correspond to current coco set")
        results.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        if self._detection_type == "bbox":
            for idx, ann in enumerate(annotations):
                bb = ann["bbox"]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = idx + 1
                ann["iscrowd"] = 0
        elif self._detection_type == "segmentation":
            for idx, ann in enumerate(annotations):
                ann["area"] = mask.area(ann["segmentation"])
                ann["bbox"] = mask.toBbox(ann["segmentation"])
                ann["id"] = idx + 1
                ann["iscrowd"] = 0
        logger.info("DONE (t=%0.2fs)", (time.time() - tic))

        results.dataset["annotations"] = annotations
        results.createIndex()
        return results


class COCOEvalWrapper(cocoeval.COCOeval):
    """Wrapper for the pycocotools COCOeval class.

    To evaluate, create two objects (groundtruth_dict and detections_list)
    using the conventions listed at http://mscoco.org/dataset/#format.
    Then call evaluation as follows:

      groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
      detections = groundtruth.LoadAnnotations(detections_list)
      evaluator = coco_tools.COCOEvalWrapper(groundtruth, detections,
                                             agnostic_mode=False)
      metrics = evaluator.ComputeMetrics()
    """

    def __init__(
        self,
        groundtruth: coco.COCO = None,
        detections: coco.COCO = None,
        agnostic_mode=False,
        iou_type: str = "bbox",
        iou_thrs: Union[str, float] = None,
        map_points=None,
    ):
        """Construct a COCOEvalWrapper.

        Note that for the area-based metrics to be meaningful, detection and
        groundtruth boxes must be in image coordinates measured in pixels.

        Args:
          groundtruth: a coco.COCO (or coco_tools.COCOWrapper) object holding
            groundtruth annotations
          detections: a coco.COCO (or coco_tools.COCOWrapper) object holding
            detections
          agnostic_mode: boolean (default: False).  If True, evaluation ignores
            class labels, treating all detections as proposals.
          iou_thrs: Minimal value for intersection over union that allows to
                    make decision that prediction bounding box is true positive.
                    You can specify one float value between 0 to 1 or
                    string "05:0.05:0.95" for standard COCO thresholds.
          iou_type: IOU type to use for evaluation. Supports `bbox` or `segm`.
          map_points: The way to calculate mAP. 101 for 101-point interpolated AP, 11 for
                          11-point interpolated AP, 0 for area under PR curve.
        """
        cocoeval.COCOeval.__init__(self, groundtruth, detections, iouType=iou_type)
        if agnostic_mode:
            self.params.useCats = 0
        if iou_thrs == "0.5:0.05:0.95":
            self.params.iouThrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        elif isinstance(iou_thrs, float):
            self.params.iouThrs = [iou_thrs]

        if map_points == 101:
            self.params.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
        if map_points == 11:
            self.params.recThrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.1)) + 1, endpoint=True)
        if map_points == 0:
            self.params.recThrs = [-1]

    def GetCategory(self, category_id: int) -> dict:
        """Fetch dictionary holding category information given category id.

        Args:
          category_id: integer id

        Returns:
          dictionary holding 'id', 'name'.
        """
        return self.cocoGt.cats[category_id]

    def GetAgnosticMode(self) -> bool:
        """Return whether COCO Eval is configured to evaluate in agnostic mode."""
        return self.params.useCats == 0

    def GetCategoryIdList(self) -> List[int]:
        """Return the list of IDs of all valid categories."""
        return self.params.catIds

    def accumulate(self, p: cocoeval.Params = None):
        """Accumulate evaluation results per image and store it to self.eval.

        Args:
          p: input params for evaluation
        """
        print("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            print("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        print("-pe", _pe)
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e["dtScores"][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate([e["dtMatches"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e["dtIgnore"][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))

                        # calculate precision
                        if R == 1:
                            rc = np.concatenate(([0.0], rc, [1.0]))
                            pr = np.concatenate(([0.0], pr, [0.0]))

                            # compute the precision envelope
                            for i in range(pr.size - 1, 0, -1):
                                pr[i - 1] = np.maximum(pr[i - 1], pr[i])

                            # to calculate area under PR curve, look for points
                            # where X axis (recall) changes value
                            change_point = np.where(rc[1:] != rc[:-1])[0]
                            # and sum (\Delta recall) * recall
                            res = np.sum((rc[change_point + 1] - rc[change_point]) * pr[change_point + 1])
                            precision[t, :, k, a, m] = np.array([res])
                        else:
                            q = np.zeros((R,))

                            # numpy is slow without cython optimization for accessing elements
                            # use python array gets significant speed improvement
                            pr = pr.tolist()
                            q = q.tolist()

                            for i in range(nd - 1, 0, -1):
                                if pr[i] > pr[i - 1]:
                                    pr[i - 1] = pr[i]

                            inds = np.searchsorted(rc, p.recThrs, side="left")
                            try:
                                for ri, pi in enumerate(inds):
                                    q[ri] = pr[pi]
                            except:
                                pass
                            precision[t, :, k, a, m] = np.array(q)

                        # calculate recall
                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # calculate score
                        ss = np.zeros((R,))
                        inds = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        scores[t, :, k, a, m] = np.array(ss)
        # exit(0)
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        toc = time.time()
        print("DONE (t={:0.2f}s).".format(toc - tic))

    def ComputeMetrics(
        self, include_metrics_per_category: bool = False, all_metrics_per_category: bool = False
    ):  # pragma: no cover
        """Compute detection metrics.

        Args:
          include_metrics_per_category: Whether include metrics per category.
          all_metrics_per_category: Whether include all the summery metrics for
            each category in per_category_ap. Be careful with setting it to true if
            you have more than handful of categories, because it will pollute
            your mldash.

        Returns:
          A tuple of (summary_metrics, per_category_ap), in which
            (1) summary_metrics is a dictionary holding:
              'Precision/mAP': mean average precision over classes averaged over IOU
                thresholds ranging from .5 to .95 with .05 increments;
              'Precision/mAP@.50IOU': mean average precision at 50% IOU;
              'Precision/mAP@.75IOU': mean average precision at 75% IOU;
              'Precision/mAP (small)': mean average precision for small objects
                (area < 32^2 pixels);
              'Precision/mAP (medium)': mean average precision for medium sized
                objects (32^2 pixels < area < 96^2 pixels);
              'Precision/mAP (large)': mean average precision for large objects
                (96^2 pixels < area < 10000^2 pixels);
              'Recall/AR@1': average recall with 1 detection;
              'Recall/AR@10': average recall with 10 detections;
              'Recall/AR@100': average recall with 100 detections;
              'Recall/AR@100 (small)': average recall for small objects with 100
                detections;
              'Recall/AR@100 (medium)': average recall for medium objects with 100
                detections;
              'Recall/AR@100 (large)': average recall for large objects with 100
                detections;
            and (2) per_category_ap is a dictionary holding category specific results with
              keys of the form: 'Precision mAP ByCategory/category'
              (without the supercategory part if no supercategories exist).

          For backward compatibility 'PerformanceByCategory' is included in the
            output regardless of all_metrics_per_category. If evaluating class-agnostic
            mode, per_category_ap is an empty dictionary.

        Raises:
          ValueError: If category_stats does not exist.
        """
        self.evaluate()
        self.accumulate()
        self.summarize()

        summary_metrics = OrderedDict(
            [
                ("Precision/mAP", self.stats[0]),
                ("Precision/mAP@.50IOU", self.stats[1]),
                ("Precision/mAP@.75IOU", self.stats[2]),
                ("Precision/mAP (small)", self.stats[3]),
                ("Precision/mAP (medium)", self.stats[4]),
                ("Precision/mAP (large)", self.stats[5]),
                ("Recall/AR@1", self.stats[6]),
                ("Recall/AR@10", self.stats[7]),
                ("Recall/AR@100", self.stats[8]),
                ("Recall/AR@100 (small)", self.stats[9]),
                ("Recall/AR@100 (medium)", self.stats[10]),
                ("Recall/AR@100 (large)", self.stats[11]),
            ]
        )
        if not include_metrics_per_category:
            return summary_metrics, {}
        if not hasattr(self, "category_stats"):
            raise ValueError("Category stats do not exist")
        per_category_ap = OrderedDict([])
        if self.GetAgnosticMode():
            return summary_metrics, per_category_ap
        for category_index, category_id in enumerate(self.GetCategoryIdList()):
            category = self.GetCategory(category_id)["name"]
            # Kept for backward compatilbility
            # pylint: disable=no-member
            per_category_ap["PerformanceByCategory/mAP/{}".format(category)] = self.category_stats[0][category_index]
            if all_metrics_per_category:
                per_category_ap["Precision mAP ByCategory/{}".format(category)] = self.category_stats[0][category_index]
                per_category_ap["Precision mAP@.50IOU ByCategory/{}".format(category)] = self.category_stats[1][
                    category_index
                ]
                per_category_ap["Precision mAP@.75IOU ByCategory/{}".format(category)] = self.category_stats[2][
                    category_index
                ]
                per_category_ap["Precision mAP (small) ByCategory/{}".format(category)] = self.category_stats[3][
                    category_index
                ]
                per_category_ap["Precision mAP (medium) ByCategory/{}".format(category)] = self.category_stats[4][
                    category_index
                ]
                per_category_ap["Precision mAP (large) ByCategory/{}".format(category)] = self.category_stats[5][
                    category_index
                ]
                per_category_ap["Recall AR@1 ByCategory/{}".format(category)] = self.category_stats[6][category_index]
                per_category_ap["Recall AR@10 ByCategory/{}".format(category)] = self.category_stats[7][category_index]
                per_category_ap["Recall AR@100 ByCategory/{}".format(category)] = self.category_stats[8][category_index]
                per_category_ap["Recall AR@100 (small) ByCategory/{}".format(category)] = self.category_stats[9][
                    category_index
                ]
                per_category_ap["Recall AR@100 (medium) ByCategory/{}".format(category)] = self.category_stats[10][
                    category_index
                ]
                per_category_ap["Recall AR@100 (large) ByCategory/{}".format(category)] = self.category_stats[11][
                    category_index
                ]

        return summary_metrics, per_category_ap


def _ConvertBoxToCOCOFormat(box):
    """Convert a box in [ymin, xmin, ymax, xmax] format to COCO format.

    This is a utility function for converting from our internal
    [ymin, xmin, ymax, xmax] convention to the convention used by the COCO API
    i.e., [xmin, ymin, width, height].

    Args:
      box: a numpy array in format of [ymin, xmin, ymax, xmax]

    Returns:
      A list of floats, in COCO format, representing [xmin, ymin, width, height]
    """
    return [float(box[1]), float(box[0]), float(box[3] - box[1]), float(box[2] - box[0])]


def _RleCompress(masks):
    """Compresses mask using Run-length encoding provided by pycocotools.

    Args:
      masks: uint8 numpy array of shape [mask_height, mask_width] with values in
        {0, 1}.

    Returns:
      A pycocotools Run-length encoding of the mask.
    """
    return mask.encode(np.asfortranarray(masks))


def ExportSingleImageGroundtruthToCoco(
    image_id: Union[int, str],
    next_annotation_id: int,
    category_id_set: Set[str],
    groundtruth_boxes: np.array,
    groundtruth_classes: np.array,
    groundtruth_masks: Union[np.array, None] = None,
    groundtruth_is_crowd: Union[np.array, None] = None,
) -> list:
    """Export groundtruth of a single image to COCO format.

    This function converts groundtruth detection annotations represented as numpy
    arrays to dictionaries that can be ingested by the COCO evaluation API. Note
    that the image_ids provided here must match the ones given to
    ExportSingleImageDetectionsToCoco. We assume that boxes and classes are in
    correspondence - that is: groundtruth_boxes[i, :], and
    groundtruth_classes[i] are associated with the same groundtruth annotation.

    In the exported result, "area" fields are always set to the area of the
    groundtruth bounding box.

    Args:
      image_id: a unique image identifier either of type integer or string.
      next_annotation_id: integer specifying the first id to use for the
        groundtruth annotations. All annotations are assigned a continuous integer
        id starting from this value.
      category_id_set: A set of valid class ids. Groundtruth with classes not in
        category_id_set are dropped.
      groundtruth_boxes: numpy array (float32) with shape [num_gt_boxes, 4]
      groundtruth_classes: numpy array (int) with shape [num_gt_boxes]
      groundtruth_masks: optional uint8 numpy array of shape [num_detections,
        image_height, image_width] containing detection_masks.
      groundtruth_is_crowd: optional numpy array (int) with shape [num_gt_boxes]
        indicating whether groundtruth boxes are crowd.

    Returns:
      A list of groundtruth annotations for a single image in the COCO format.

    Raises:
      ValueError: if (1) groundtruth_boxes and groundtruth_classes do not have the
        right lengths or (2) if each of the elements inside these lists do not
        have the correct shapes or (3) if image_ids are not integers
    """
    if len(groundtruth_classes.shape) != 1:
        raise ValueError("groundtruth_classes is " "expected to be of rank 1.")
    if len(groundtruth_boxes.shape) != 2:
        raise ValueError("groundtruth_boxes is expected to be of " "rank 2.")
    if groundtruth_boxes.shape[1] != 4:
        raise ValueError("groundtruth_boxes should have " "shape[1] == 4.")
    num_boxes = groundtruth_classes.shape[0]
    if num_boxes != groundtruth_boxes.shape[0]:
        raise ValueError(
            "Corresponding entries in groundtruth_classes, "
            "and groundtruth_boxes should have "
            "compatible shapes (i.e., agree on the 0th dimension)."
            "Classes shape: %d. Boxes shape: %d. Image ID: %s"
            % (groundtruth_classes.shape[0], groundtruth_boxes.shape[0], image_id)
        )
    has_is_crowd = groundtruth_is_crowd is not None
    if has_is_crowd and len(groundtruth_is_crowd.shape) != 1:
        raise ValueError("groundtruth_is_crowd is expected to be of rank 1.")
    groundtruth_list = []
    for i in range(num_boxes):
        if groundtruth_classes[i] in category_id_set:
            iscrowd = groundtruth_is_crowd[i] if has_is_crowd else 0
            export_dict = {
                "id": next_annotation_id + i,
                "image_id": image_id,
                "category_id": int(groundtruth_classes[i]),
                "bbox": list(_ConvertBoxToCOCOFormat(groundtruth_boxes[i, :])),
                "area": float(
                    (groundtruth_boxes[i, 2] - groundtruth_boxes[i, 0])
                    * (groundtruth_boxes[i, 3] - groundtruth_boxes[i, 1])
                ),
                "iscrowd": iscrowd,
            }
            if groundtruth_masks is not None:
                export_dict["segmentation"] = _RleCompress(groundtruth_masks[i])
            groundtruth_list.append(export_dict)
    return groundtruth_list


def ExportSingleImageDetectionBoxesToCoco(
    image_id: Union[int, str],
    category_id_set: Set[int],
    detection_boxes: np.array,
    detection_scores: np.array,
    detection_classes: np.array,
) -> list:
    """Export detections of a single image to COCO format.

    This function converts detections represented as numpy arrays to dictionaries
    that can be ingested by the COCO evaluation API. Note that the image_ids
    provided here must match the ones given to the
    ExporSingleImageDetectionBoxesToCoco. We assume that boxes, and classes are in
    correspondence - that is: boxes[i, :], and classes[i]
    are associated with the same groundtruth annotation.

    Args:
        image_id: unique image identifier either of type integer or string.
        category_id_set: A set of valid class ids. Detections with classes not in
          category_id_set are dropped.
        detection_boxes: float numpy array of shape [num_detections, 4] containing
          detection boxes.
        detection_scores: float numpy array of shape [num_detections] containing
          scored for the detection boxes.
        detection_classes: integer numpy array of shape [num_detections] containing
          the classes for detection boxes.

    Returns:
        A list of detection annotations for a single image in the COCO format.

    Raises:
        ValueError: if (1) detection_boxes, detection_scores and detection_classes
        do not have the right lengths or (2) if each of the elements inside these
        lists do not have the correct shapes or (3) if image_ids are not integers.
    """
    if len(detection_classes.shape) != 1 or len(detection_scores.shape) != 1:
        raise ValueError("All entries in detection_classes and detection_scores" "expected to be of rank 1.")
    if len(detection_boxes.shape) != 2:
        raise ValueError("All entries in detection_boxes expected to be of " "rank 2.")
    if detection_boxes.shape[1] != 4:
        raise ValueError("All entries in detection_boxes should have " "shape[1] == 4.")
    num_boxes = detection_classes.shape[0]
    if not num_boxes == detection_boxes.shape[0] == detection_scores.shape[0]:
        raise ValueError(
            "Corresponding entries in detection_classes, "
            "detection_scores and detection_boxes should have "
            "compatible shapes (i.e., agree on the 0th dimension). "
            "Classes shape: %d. Boxes shape: %d. "
            "Scores shape: %d" % (detection_classes.shape[0], detection_boxes.shape[0], detection_scores.shape[0])
        )
    detections_list = []
    for i in range(num_boxes):
        if detection_classes[i] in category_id_set:
            detections_list.append(
                {
                    "image_id": image_id,
                    "category_id": int(detection_classes[i]),
                    "bbox": list(_ConvertBoxToCOCOFormat(detection_boxes[i, :])),
                    "score": float(detection_scores[i]),
                }
            )
    return detections_list


def ExportSingleImageDetectionMasksToCoco(
    image_id: Union[str, int],
    category_id_set: Set[int],
    detection_masks: np.array,
    detection_scores: np.array,
    detection_classes: np.array,
) -> list:
    """Export detection masks of a single image to COCO format.

    This function converts detections represented as numpy arrays to dictionaries
    that can be ingested by the COCO evaluation API. We assume that
    detection_masks, detection_scores, and detection_classes are in correspondence
    - that is: detection_masks[i, :], detection_classes[i] and detection_scores[i]
        are associated with the same annotation.

    Args:
        image_id: unique image identifier either of type integer or string.
        category_id_set: A set of valid class ids. Detections with classes not in
        category_id_set are dropped.
        detection_masks: uint8 numpy array of shape [num_detections, image_height,
        image_width] containing detection_masks.
        detection_scores: float numpy array of shape [num_detections] containing
        scores for detection masks.
        detection_classes: integer numpy array of shape [num_detections] containing
        the classes for detection masks.

    Returns:
        A list of detection mask annotations for a single image in the COCO format.

    Raises:
        ValueError: if (1) detection_masks, detection_scores and detection_classes
        do not have the right lengths or (2) if each of the elements inside these
        lists do not have the correct shapes or (3) if image_ids are not integers.
    """
    if len(detection_classes.shape) != 1 or len(detection_scores.shape) != 1:
        raise ValueError("All entries in detection_classes and detection_scores" "expected to be of rank 1.")
    num_boxes = detection_classes.shape[0]
    if not num_boxes == len(detection_masks) == detection_scores.shape[0]:
        raise ValueError(
            "Corresponding entries in detection_classes, "
            "detection_scores and detection_masks should have "
            "compatible lengths and shapes "
            "Classes length: %d.  Masks length: %d. "
            "Scores length: %d" % (detection_classes.shape[0], len(detection_masks), detection_scores.shape[0])
        )
    detections_list = []
    for i in range(num_boxes):
        if detection_classes[i] in category_id_set:
            detections_list.append(
                {
                    "image_id": image_id,
                    "category_id": int(detection_classes[i]),
                    "segmentation": _RleCompress(detection_masks[i]),
                    "score": float(detection_scores[i]),
                }
            )
    return detections_list
