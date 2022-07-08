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

import argparse
import cv2
import logging
import numpy as np
import onnx
import os
from scipy.io import loadmat

class Dataloader:
    def __init__(self, data_path, size=[320,240]):
        self.batch_size = 1
        image_mean=np.array([127, 127, 127], dtype=np.float32)
        image_std = 128.0
        self.data = []
        for parent, dir_names, file_names in os.walk(data_path):
            for file_name in file_names:
                if not file_name.lower().endswith('jpg'):
                    continue
                image = cv2.imread(os.path.join(parent, file_name), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, _ = image.shape
                image = cv2.resize(image, size)
                image = (image - image_mean) / image_std
                label = [(height, width), parent.split('/')[-1], file_name.split('.')[0]]
                self.data.append((np.transpose(image, [2,0,1])[np.newaxis,:], label))
            
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for item in self.data:
            yield item

class Post:
    def __init__(self) -> None:
        self.iou_threshold = 0.3
        self.prob_threshold = 0.1
        self.candidate_size = 800

    def __call__(self, sample):
        preds, labels = sample
        height, width = labels[0]
        confidences, boxes = preds
        boxes = boxes[0]
        confidences = confidences[0]

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            #print(confidences.shape[1])
            probs = confidences[:, class_index]
            #print(probs)
            mask = probs > self.prob_threshold
            probs = probs[mask]

            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            #print(subset_boxes)
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(box_probs,
                iou_threshold=self.iou_threshold,
                top_k=-1,
            )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return ([np.array([]), np.array([]), np.array([])], labels)
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return ([picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]], labels)

def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

class AP:
    def __init__(self, label_dir):
        gt_mat = loadmat(os.path.join(label_dir, 'wider_face_val.mat'))
        easy_mat = loadmat(os.path.join(label_dir, 'wider_easy_val.mat'))
        self.facebox_list = gt_mat['face_bbx_list']
        self.event_list = gt_mat['event_list']
        self.file_list = gt_mat['file_list']
        self.gt_list = easy_mat['gt_list']
        self.pred = {event_name[0].item():{} for event_name in self.event_list}

    def reset(self):
        pass

    def update(self, preds, labels):
        bboxes, scores = preds[0], preds[2]
        event_name, imgname = labels[1:]
        if len(bboxes) > 0:
            self.pred[event_name][imgname] = np.concatenate(
                [np.floor(bboxes[:,:2]), np.ceil(bboxes[:,2:]-bboxes[:,:2]), scores[:,np.newaxis]],
                axis=1)
        else:
            self.pred[event_name][imgname] = []

    def result(self, iou_thresh=0.3):
        norm_score(self.pred)
        event_num = len(self.event_list)
        thresh_num = 1000
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        for i in range(event_num):
            event_name = str(self.event_list[i][0][0])
            img_list = self.file_list[i][0]
            pred_list = self.pred[event_name]
            sub_gt_list = self.gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = self.facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        return ap

def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=query_boxes.dtype)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

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
    help="Path of wider face validation dataset."
)
parser.add_argument(
    '--label_path',
    type=str,
    help="Path to label files.",
    default='./'
)
parser.add_argument(
    '--input_size',
    type=int,
    nargs=2,
    help="Size of input image, [320,240] for version-RFB-320 model, [640,480] for version-RFB-640 model.",
    default=[320,240]
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

if __name__ == "__main__":
    model = onnx.load(args.model_path)
    dataloader  = Dataloader(args.data_path, size=args.input_size)
    if args.benchmark:
        from neural_compressor.experimental import Benchmark, common
        if args.mode == 'accuracy':
            evaluator = Benchmark(args.config)
            evaluator.model = common.Model(model)
            evaluator.b_dataloader = dataloader
            evaluator.postprocess = common.Postprocess(Post)
            evaluator.metric = AP(args.label_path)
            evaluator(args.mode)
        else:
            evaluator = Benchmark(args.config)
            evaluator.model = common.Model(model)
            evaluator(args.mode)

    if args.tune:
        from neural_compressor.experimental import Quantization, common
        quantize = Quantization(args.config)
        quantize.model = common.Model(model)
        quantize.eval_dataloader = dataloader
        quantize.calib_dataloader = dataloader
        quantize.postprocess = common.Postprocess(Post)
        quantize.metric = AP(args.label_path)
        q_model = quantize()
        q_model.save(args.output_model)