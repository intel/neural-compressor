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
import collections
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import accuracy_score

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
        return self.num_correct / self.num_sample

class Dataloader:
    def __init__(self, dataset_location, image_list, batch_size):
        self.batch_size = batch_size
        self.image_list = []
        self.label_list = []
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(dataset_location, image_name)
                if not os.path.exists(src):
                    continue

                self.image_list.append(src)
                self.label_list.append(int(label))

    def _preprpcess(self, src):
        with Image.open(src) as image:
            image = np.array(image.convert('RGB')).astype(np.float32)
            image = image / 255.
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

            h, w = image.shape[0], image.shape[1]

            y0 = (h - 224) // 2
            x0 = (w - 224) // 2
            image = image[y0:y0 + 224, x0:x0 + 224, :]
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            image = image.transpose((2, 0, 1))
        return image.astype('float32')

    def __iter__(self):
        return self._generate_dataloader()

    def _generate_dataloader(self):
        sampler = iter(range(0, len(self.image_list), 1))

        def collate(batch):
            """Puts each data field into a pd frame with outer dimension batch size"""
            elem = batch[0]
            if isinstance(elem, collections.abc.Mapping):
                return {key: collate([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, collections.abc.Sequence):
                batch = zip(*batch)
                return [collate(samples) for samples in batch]
            elif isinstance(elem, np.ndarray):
                try:
                    return np.stack(batch)
                except:
                    return batch
            else:
                return batch

        def batch_sampler():
            batch = []
            for idx in sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0:
                yield batch

        def fetcher(ids):
            data = [self._preprpcess(self.image_list[idx]) for idx in ids]
            label = [self.label_list[idx] for idx in ids]
            return collate(data), label

        for batched_indices in batch_sampler():
            try:
                data = fetcher(batched_indices)
                yield data
            except StopIteration:
                return

def eval_func(model, dataloader, metric, backend):
    metric.reset()
    provider = 'DmlExecutionProvider' if backend == 'onnxrt_dml_ep' else 'CPUExecutionProvider'
    sess = ort.InferenceSession(model.SerializeToString(), providers=[provider])
    input_names = [i.name for i in sess.get_inputs()]
    for input_data, label in dataloader:
        output = sess.run(None, dict(zip(input_names, [input_data])))
        metric.update(output, label)
    return metric.result()

if __name__ == "__main__":
    logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
    parser = argparse.ArgumentParser(
        description="Shufflenet fine-tune examples for image classification tasks.",
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
        '--label_path',
        type=str,
        help="Imagenet label path"
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
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'npu'],
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )
    args = parser.parse_args()

    model = onnx.load(args.model_path)
    dataloader = Dataloader(args.dataset_location, args.label_path, args.batch_size)
    top1 = TopK()
    backend = 'onnxrt_dml_ep' if args.device == 'npu' else 'default'
    def eval(onnx_model):
        return eval_func(onnx_model, dataloader, top1, backend)

    if args.benchmark:
        if args.mode == 'performance':
            from neural_compressor.benchmark import fit
            from neural_compressor.config import BenchmarkConfig
            conf = BenchmarkConfig(warmup=10,
                                   iteration=1000,
                                   cores_per_instance=4,
                                   num_of_instance=1,
                                   device=args.device,
                                   backend=backend)
            fit(model, conf, b_dataloader=dataloader)
        elif args.mode == 'accuracy':
            acc_result = eval(model)
            print("Batch size = %d" % dataloader.batch_size)
            print("Accuracy: %.5f" % acc_result)
    if args.tune:
        from neural_compressor import quantization, PostTrainingQuantConfig
        config = PostTrainingQuantConfig(quant_format=args.quant_format,
                                         device=args.device,
                                         backend=backend)
 
        q_model = quantization.fit(model, config, calib_dataloader=dataloader, eval_func=eval)
        q_model.save(args.output_model)
