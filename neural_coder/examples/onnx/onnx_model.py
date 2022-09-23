# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import argparse

import onnx
import yaml

from pycocotools.coco import COCO
from pycocotools.mask import iou, encode
import numpy as np
from torchvision import transforms
from PIL import Image
from onnx import numpy_helper
import os
import onnxruntime

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.WARN)
logger.info("Evaluating ONNXRuntime full precision accuracy and performance:")
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--model_path',
    type=str,
    help="Pre-trained model on onnx file"
)
parser.add_argument(
    '--label_path',
    type=str,
    help="Annotation file path"
)
parser.add_argument(
    '--data_path',
    type=str,
    help="Path to val2017 of COCO"
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

# key = COCO id, value = Pascal VOC id
COCO_TO_VOC = {
    1: 15,  # person
    2: 2,   # bicycle
    3: 7,   # car
    4: 14,  # motorbike
    5: 1,   # airplane
    6: 6,   # bus
    7: 19,  # train
    9: 4,   # boat
    16: 3,  # bird
    17: 8,  # cat
    18: 12, # dog
    19: 13, # horse
    20: 17, # sheep
    21: 10, # cow
    44: 5,  # bottle
    62: 9,  # chair
    63: 18, # couch/sofa
    64: 16, # potted plant
    67: 11, # dining table
    72: 20, # tv
}
VOC_CAT_IDS = list(COCO_TO_VOC.keys())
cocoGt = COCO(str(args.label_path))

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset:
    def __init__(self):
        imgIds = self.getImgIdsUnion(cocoGt, VOC_CAT_IDS)
        self.data = []
        for imgId in imgIds:
            img_path = os.path.join(args.data_path, cocoGt.imgs[imgId]['file_name'])
            if os.path.exists(img_path):
                input_tensor = self.load_image(img_path)
                
                _, height, width = input_tensor.shape
                output_tensor = np.zeros((21, height, width), dtype=np.uint8)
                
                annIds = cocoGt.getAnnIds(imgId, VOC_CAT_IDS)
                for ann in cocoGt.loadAnns(annIds):
                    mask = cocoGt.annToMask(ann)
                    output_tensor[COCO_TO_VOC[ann['category_id']]] |= mask
                    
                # Set everything not labeled to be background
                output_tensor[0] = 1 - np.max(output_tensor, axis=0)
                self.data.append((input_tensor, output_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def getImgIdsUnion(self, gt, catIds):
        """
        Returns all the images that have *any* of the categories in `catIds`,
        unlike the built-in `gt.getImgIds` which returns all the images containing
        *all* of the categories in `catIds`.
        """
        imgIds = set()
        for catId in catIds:
            imgIds |= set(gt.catToImgs[catId])
        return list(imgIds)

    def load_image(self, img_path):
        input_image = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(input_image)
        input_tensor = input_tensor.detach().cpu().numpy()
        return input_tensor
    
def iou(model_tensor, target_tensor):
    # Don't include the background when summing
    model_tensor = model_tensor[:, 1:, :, :]
    target_tensor = target_tensor[:, 1:, :, :]
    
    intersection = np.sum(np.logical_and(model_tensor, target_tensor))
    union = np.sum(np.logical_or(model_tensor, target_tensor))
    
    if union == 0:
        # Can only happen if nothing was there and nothing was predicted,
        # which is a perfect score
        return 1
    else:
        return intersection / union

def evaluate(model, dataloader):    
    totalIoU = 0
    sess = onnxruntime.InferenceSession(model.SerializeToString(), None)
    idx = 1
    for input_tensor, target_tensor in dataloader:
        input_tensor = input_tensor[np.newaxis, ...]
        target_tensor = target_tensor[np.newaxis, ...]
        model_tensor = sess.run(["out"], {"input": input_tensor})[0]
        
        batch_size, nclasses, height, width = model_tensor.shape
        raw_labels = np.argmax(model_tensor, axis=1).astype(np.uint8)
        
        output_tensor = np.zeros((nclasses, batch_size, height, width), dtype=np.uint8)
        for c in range(nclasses):
            output_tensor[c][raw_labels==c] = 1

        output_tensor = np.transpose(output_tensor, [1, 0, 2, 3])          
        totalIoU += iou(output_tensor, target_tensor)    
        idx += 1
    return totalIoU / idx

if __name__ == "__main__":
    from neural_compressor.experimental import common
    ds = Dataset()
    dataloader = common.DataLoader(ds)
    model = onnx.load(args.model_path)
    def eval(model):
        return evaluate(model, ds)

    if args.benchmark and args.mode == "accuracy":
        results = eval(model)
        print("Batch size = 1")
        print("Accuracy: %.5f" % results)

    if args.benchmark and args.mode == "performance":
        from neural_compressor.experimental import Benchmark, common
        evaluator = Benchmark(args.config)
        evaluator.model = common.Model(model)
        evaluator.b_dataloader = common.DataLoader(ds)
        evaluator(args.mode)
