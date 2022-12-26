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
import os
import re
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.metrics import accuracy_score
import cv2
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

def _topk_shape_validate(preds, labels):
    # preds shape can be Nxclass_num or class_num(N=1 by default)
    # it's more suitable for 'Accuracy' with preds shape Nx1(or 1) output from argmax
    if isinstance(preds, int):
        preds = [preds]
        preds = np.array(preds)
    elif isinstance(preds, np.ndarray):
        preds = np.array(preds)
    elif isinstance(preds, list):
        preds = np.array(preds)
        preds = preds.reshape((-1, preds.shape[-1]))

    # consider labels just int value 1x1
    if isinstance(labels, int):
        labels = [labels]
        labels = np.array(labels)
    elif isinstance(labels, tuple):
        labels = np.array([labels])
        labels = labels.reshape((labels.shape[-1], -1))
    elif isinstance(labels, list):
        if isinstance(labels[0], int):
            labels = np.array(labels)
            labels = labels.reshape((labels.shape[0], 1))
        elif isinstance(labels[0], tuple):
            labels = np.array(labels)
            labels = labels.reshape((labels.shape[-1], -1))
        else:
            labels = np.array(labels)
    # labels most have 2 axis, 2 cases: N(or Nx1 sparse) or Nxclass_num(one-hot)
    # only support 2 dimension one-shot labels
    # or 1 dimension one-hot class_num will confuse with N

    if len(preds.shape) == 1:
        N = 1
        class_num = preds.shape[0]
        preds = preds.reshape([-1, class_num])
    elif len(preds.shape) >= 2:
        N = preds.shape[0]
        preds = preds.reshape([N, -1])
        class_num = preds.shape[1]

    label_N = labels.shape[0]
    assert label_N == N, 'labels batch size should same with preds'
    labels = labels.reshape([N, -1])
    # one-hot labels will have 2 dimension not equal 1
    if labels.shape[1] != 1:
        labels = labels.argsort()[..., -1:]
    return preds, labels

class TopK:
    def __init__(self, k=1):
        self.k = k
        self.num_correct = 0
        self.num_sample = 0

    def update(self, preds, labels, sample_weight=None):
        preds, labels = _topk_shape_validate(preds, labels)
        preds = preds.argsort()[..., -self.k:]
        if self.k == 1:
            correct = accuracy_score(preds, labels, normalize=False)
            self.num_correct += correct

        else:
            for p, l in zip(preds, labels):
                # get top-k labels with np.argpartition
                # p = np.argpartition(p, -self.k)[-self.k:]
                l = l.astype('int32')
                if l in p:
                    self.num_correct += 1

        self.num_sample += len(labels)

    def reset(self):
        self.num_correct = 0
        self.num_sample = 0

    def result(self):
        if self.num_sample == 0:
            logger.warning("Sample num during evaluation is 0.")
            return 0
        elif getattr(self, '_hvd', None) is not None:
            allgather_num_correct = sum(self._hvd.allgather_object(self.num_correct))
            allgather_num_sample = sum(self._hvd.allgather_object(self.num_sample))
            return allgather_num_correct / allgather_num_sample
        return self.num_correct / self.num_sample

class Dataloader:
    def __init__(self, dataset_location, image_list):
        self.batch_size = 1
        self.image_list = []
        self.label_list = []
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(dataset_location, image_name)
                if not os.path.exists(src):
                    continue
                with Image.open(src) as image:
                    image = np.array(image.convert('RGB')).astype(np.float32)
                    image = image / 255.
                    image = cv2.resize(image, (256,256))
                    h, w = image.shape[0], image.shape[1]
                    if h + 1 < 224 or w + 1 < 224:
                        raise ValueError(
                            "Required crop size {} is larger then input image size {}".format(
                                (224, 224), (h, w)))

                    if 224 != h or 224 != w:
                        y0 = (h - 224) // 2
                        x0 = (w - 224) // 2
                        image = image[y0:y0 + 224, x0:x0 + 224, :]
                    
                    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225] 
                    image = image.transpose((2, 0, 1))
                    image = np.expand_dims(image, axis=0)
                self.image_list.append(image.astype(np.float32))
                self.label_list.append(int(label))

    def __iter__(self):
        for image, label in zip(self.image_list, self.label_list):
            yield image, label

def eval_func(model, dataloader, metric, postprocess):
    metric.reset()
    sess = ort.InferenceSession(model.SerializeToString(), providers=ort.get_available_providers())
    ort_inputs = {}
    input_names = [i.name for i in sess.get_inputs()]
    for input_data, label in dataloader:
        output = sess.run(None, dict(zip(input_names, [input_data])))
        output, label = postprocess((output, label))
        metric.update(output, label)
    return metric.result()

class Squeeze:
    def __call__(self, sample):
        preds, labels = sample
        return np.squeeze(preds), labels

if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        description="Densenet fine-tune examples for image classification tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        '--dataset_location',
        type=str,
        help="Imagenet data path"
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
    model = onnx.load(args.model_path)
    data_path = os.path.join(args.dataset_location, 'ILSVRC2012_img_val')
    label_path = os.path.join(args.dataset_location, 'val.txt')
    dataloader = Dataloader(data_path, label_path)
    top1 = TopK()
    postprocess = Squeeze()
    def eval(onnx_model):
        return eval_func(onnx_model, dataloader, top1, postprocess)

    if args.benchmark:
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(warmup=10, iteration=1000, cores_per_instance=4, num_of_instance=1)
            fit(model, conf, b_dataloader=dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval(model)
            print("Batch size = %d" % dataloader.batch_size)
            print("Accuracy: %.5f" % acc_result)
    if args.tune:
        from neural_compressor import quantization, PostTrainingQuantConfig
        from neural_compressor.config import AccuracyCriterion
        accuracy_criterion = AccuracyCriterion()
        accuracy_criterion.relative = 0.02
        config = PostTrainingQuantConfig(
            accuracy_criterion=accuracy_criterion,
            op_name_list={'Conv_nc_rename_431': {'activation': {'dtype': ['fp32']}, 'weight': {'dtype': ['fp32']}}})
 
        q_model = quantization.fit(model, config, calib_dataloader=dataloader,
			     eval_func=eval)

        q_model.save(args.output_model)
