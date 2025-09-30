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
import cv2
import numpy as np
import onnx
import re
import os
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)

class Squeeze:
    def __call__(self, sample):
        preds, labels = sample
        return np.squeeze(preds), labels
    
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
    def __init__(self, dataset_location, image_list, batch_size=1):
        self.batch_size = batch_size
        self.image_list = []
        self.label_list = []
        self.resize_side = 256
        self.crop_size = 224
        self.mean_value = [0.485, 0.456, 0.406]
        self.std_value = [0.229, 0.224, 0.225]
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(dataset_location, image_name)
                if not os.path.exists(src):
                    continue

                self.image_list.append(src)
                self.label_list.append(int(label))

    def __iter__(self):
        batched_image = None
        batched_label = None
        for index, (src, label) in enumerate(zip(self.image_list, self.label_list)):
            with Image.open(src) as image:
                image = np.array(image.convert('RGB')).astype(np.float32)
                height, width = image.shape[0], image.shape[1]
                scale = self.resize_side / width if height > width else self.resize_side / height
                new_height = int(height*scale)
                new_width = int(width*scale)
                image = cv2.resize(image, (new_height, new_width))
                image = image / 255.
                shape = image.shape
                y0 = (shape[0] - self.crop_size) // 2
                x0 = (shape[1] - self.crop_size) // 2
                if len(image.shape) == 2:
                    image = np.array([image])
                    image = np.repeat(image, 3, axis=0)
                    image = image.transpose(1, 2, 0)
                image = image[y0:y0+self.crop_size, x0:x0+self.crop_size, :]
                image = ((image - self.mean_value)/self.std_value).astype(np.float32)
                image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            if batched_label is None:
                batched_image = image
                batched_label = label
            else:
                batched_image = np.append(batched_image, image, axis=0)
                batched_label = np.append(batched_label, label, axis=0)
            if (index + 1) % self.batch_size == 0:
                yield batched_image, batched_label
                batched_image = None
                batched_label = None
        if (index + 1) % self.batch_size != 0:
            yield batched_image, batched_label

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

if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        description="Googlenet fine-tune examples for image classification tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Pre-trained model on onnx file"
    )
    parser.add_argument(
        '--dataset_location',
        type=str,
        help="Imagenet data path"
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
        '--output_model',
        type=str,
        help="output model path"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help="batch_size of dataloader"
    )
    parser.add_argument(
        '--iters',
        type=int,
        help="iters of dataloader"
    )
    parser.add_argument(
        '--mode',
        type=str,
        help="benchmark mode of performance or accuracy"
    )
    parser.add_argument(
        '--quant_format',
        type=str,
        default='default', 
        choices=['default', 'QDQ', 'QOperator'],
        help="quantization format"
    )
    args = parser.parse_args()

    model = onnx.load(args.model_path)
    data_path = os.path.join(args.dataset_location, 'ILSVRC2012_img_val')
    label_path = os.path.join(args.dataset_location, 'val.txt')
    dataloader = Dataloader(data_path, label_path, args.batch_size)
    top1 = TopK()
    postprocess = Squeeze()
    def eval(onnx_model):
        return eval_func(onnx_model, dataloader, top1, postprocess)

    if args.benchmark:
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(
                warmup=10, 
                iteration=args.iters, 
                cores_per_instance=4, 
                num_of_instance=1
            )
            fit(model, conf, b_dataloader=dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval(model)
            print("Batch size = %d" % dataloader.batch_size)
            print("Accuracy: %.5f" % acc_result)
    if args.tune:
        from neural_compressor import quantization, PostTrainingQuantConfig
        config = PostTrainingQuantConfig(quant_format=args.quant_format)
 
        q_model = quantization.fit(model, config, calib_dataloader=dataloader,
			     eval_func=eval)

        q_model.save(args.output_model)
