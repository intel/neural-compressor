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
import math
import cityscapes_labels
import glob
import os
import numpy as np
import cv2 as cv
import onnxruntime as ort
from PIL import Image

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
crop_sz = (800, 800)
cell_width = 2
stride = 8
rgb_mean = [122.675, 116.669, 104.008]
labels = cityscapes_labels.labels
 
class Dataloader:
    def __init__(self, data_dir, label_dir, batch_size=1):
        self.batch_size = batch_size
        index = 0
        val_lst = []
        all_images = glob.glob(os.path.join(data_dir, '*/*.png'))
        all_images.sort()
        for p in all_images:
            l = p.replace(data_dir, label_dir).replace('leftImg8bit', 'gtFine_labelIds')
            if os.path.isfile(l):
                index += 1
                for i in range(1, 8):
                    val_lst.append([str(index), p, l, "512", str(256 * i)])

        self.data = []
        for frags in val_lst:
            item = list()
            item.append(frags[1])
            item.append(frags[2])
            if len(frags) > 3:
                item.append(frags[3:])
            self.data.append(item)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return self.preprocess(item)

    def __iter__(self):
        for item in self.data:
            yield np.expand_dims(self.preprocess(item)[0], 0), self.preprocess(item)[1]

    def preprocess(self, frags):
        im = cv.imread(frags[0])
        im = im[:, :, [2, 1, 0]]
        im_size = (im.shape[0], im.shape[1])
        
        crop_coor = [int(c) for c in frags[-1]]
        x0 = int(crop_coor[0] - crop_sz[0] / 2)
        y0 = int(crop_coor[1] - crop_sz[1] / 2)
        x1 = int(x0 + crop_sz[0])
        y1 = int(y0 + crop_sz[1])

        pad_w_left = max(0 - y0, 0)
        pad_w_right = max(y1 - im_size[1], 0)
        pad_h_up = max(0 - x0, 0)
        pad_h_bottom = max(x1 - im_size[0], 0)

        x0 += pad_h_up
        x1 += pad_h_up
        y0 += pad_w_left
        y1 += pad_w_left

        img_data = np.array(im, dtype=np.float32)
        img_data = cv.copyMakeBorder(img_data, pad_h_up, pad_h_bottom, pad_w_left, pad_w_right, cv.BORDER_CONSTANT, value=[122.675, 116.669, 104.008])
        img_data = img_data[x0:x1, y0:y1, :]
        img_data = np.transpose(img_data, (2, 0, 1))

        for i in range(3):
            img_data[i] -= rgb_mean[i]

        img_label = np.array(Image.open(frags[1]))
        img_label = cv.resize(img_label, (im_size[1], im_size[0]), interpolation=cv.INTER_NEAREST)
        img_label = np.array(img_label, dtype=np.float32)
        img_label = cv.copyMakeBorder(img_label, pad_h_up, pad_h_bottom, pad_w_left, pad_w_right, cv.BORDER_CONSTANT, value=255)
        img_label = img_label[x0:x1, y0:y1]

        img_label = cv.resize(img_label, (int(crop_sz[1] / cell_width), int(crop_sz[0] / cell_width)), interpolation=cv.INTER_NEAREST)
        
        converted = np.ones(img_label.shape, dtype=np.float32) * 255
        id2trainId = {label.id: label.trainId for label in labels}
        for id in id2trainId:
            trainId = id2trainId[id]
            converted[img_label == id] = trainId

        feat_height = int(math.ceil(float(crop_sz[0]) / stride))
        feat_width = int(math.ceil(float(crop_sz[1]) / stride))
        converted = converted.reshape((feat_height, int(stride / cell_width), int(feat_width), int(stride / cell_width)))
        converted = np.transpose(converted, (1, 3, 0, 2))
        converted = converted.reshape((-1, feat_height, feat_width))
        converted = converted.reshape(-1)
        return img_data.astype('float32'), converted.astype('float32')

class IoU:
    def __init__(self, ignore_label=255, label_num=19):
        self._ignore_label = ignore_label
        self._label_num = label_num
        self.num_inst = 0
        self._tp = [0.0] * self._label_num
        self._denom = [0.0] * self._label_num

    def reset(self):
        self._tp = [0.0] * self._label_num
        self._denom = [0.0] * self._label_num

    def update(self, preds, labels):
        for i in range(len(labels)):
            pred_label = preds[i]
            label = labels[i].astype('int32')
            pred_label = np.argmax(pred_label, axis=1).astype('int32')
            iou = 0
            eps = 1e-6
            for j in range(self._label_num):
                pred_cur = (pred_label.flat == j)
                gt_cur = (label.flat == j)
                tp = np.logical_and(pred_cur, gt_cur).sum()
                denom = np.logical_or(pred_cur, gt_cur).sum() - np.logical_and(pred_cur, label.flat == self._ignore_label).sum()
                assert tp <= denom
                self._tp[j] += tp
                self._denom[j] += denom
                iou += self._tp[j] / (self._denom[j] + eps)
            iou /= self._label_num
            self.sum_metric = iou
            self.num_inst = 1

    def result(self):
        if self.num_inst == 0:
            return 0
        else:
            return self.sum_metric / self.num_inst
        

if __name__ == "__main__":
    model = onnx.load(args.model_path)
    args.data_path = args.data_path.replace('\\', '/')
    label_path = os.path.join(args.data_path.split('/leftImg8bit/val')[0], 'gtFine', 'val')    
    dataloader  = Dataloader(args.data_path, label_path, batch_size=args.batch_size)
    metric = IoU()

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
        from neural_compressor import quantization, PostTrainingQuantConfig
        from neural_compressor.config import AccuracyCriterion
        accuracy_criterion = AccuracyCriterion()
        accuracy_criterion.absolute = 0.01
        config = PostTrainingQuantConfig(approach='static', 
                                         quant_format=args.quant_format,
                                         accuracy_criterion=accuracy_criterion)
        q_model = quantization.fit(model, config, calib_dataloader=dataloader, eval_func=eval_func)
        q_model.save(args.output_model)


        
