from segment_anything import SamPredictor, sam_model_registry
import torchvision
import torch
from PIL import Image

import numpy as np
import os
import xml.etree.ElementTree as ET
from statistics import mean
from torch.nn.functional import threshold, normalize
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
from typing import List, Tuple

# Pad image - based on SAM 
def pad_image(x: torch.Tensor, square_length = 1024) -> torch.Tensor:
    # C, H, W
    h, w = x.shape[-2:]
    padh = square_length - h
    padw = square_length - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

# Custom dataset
class INC_SAMVOC2012Dataset(object):
    def __init__(self, voc_root, type):
        self.voc_root = voc_root
        self.num_of_data = -1
        self.dataset = {}  # Item will be : ["filename", "class_name", [4x bounding boxes coordinates], etc)
        self.resizelongestside = ResizeLongestSide(target_length=1024)
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

        # Read through all the samples and output a dictionary 
        # Key of the dictionary will be idx
        # Item of the dictioanry will be filename, class id and bounding boxes
        annotation_dir = os.path.join(voc_root, "Annotations")
        files = os.listdir(annotation_dir)
        files = [f for f in files if os.path.isfile(annotation_dir+'/'+f)] #Filter directory
        annotation_files = [os.path.join(annotation_dir, x) for x in files]

        # Get the name list of the segmentation files
        segmentation_dir = os.path.join(voc_root, "SegmentationObject")
        files = os.listdir(segmentation_dir)
        files = [f for f in files if os.path.isfile(segmentation_dir+'/'+f)] #Filter directory
        segmentation_files = [x for x in files]
    

        # Based on the type (train/val) to select data
        train_val_dir = os.path.join(voc_root, 'ImageSets/Segmentation/')
        if type == 'train':
            txt_file_name = 'train.txt'
        elif type =='val':
            txt_file_name = 'val.txt'
        else:
            print('Error! Type of dataset should be ''train'' or ''val'' ')

        with open(train_val_dir + txt_file_name, 'r') as f:
            permitted_files = []
            for row in f:
                permitted_files.append(row.rstrip('\n'))
        
        for file in annotation_files:
            file_name = file.split('/')[-1].split('.xml')[0]

            if not(file_name in permitted_files): 
                continue #skip the file
            
            if file_name + '.png' in segmentation_files: # check that if there is any related segmentation file for this annoation
                tree = ET.parse(file)
                root = tree.getroot()
                for child in root:
                    if child.tag == 'object':
                        details = [file_name]
                        for node in child:
                            if node.tag == 'name':
                                object_name = node.text
                            if node.tag == 'bndbox':
                                for coordinates in node:
                                    if coordinates.tag == 'xmax':
                                        xmax = int(coordinates.text)
                                    if coordinates.tag == 'xmin':
                                        xmin = int(coordinates.text)
                                    if coordinates.tag == 'ymax':
                                        ymax = int(coordinates.text)
                                    if coordinates.tag == 'ymin':
                                        ymin = int(coordinates.text)
                                boundary = [xmin, ymin, xmax, ymax]
                        details.append(object_name)
                        details.append(boundary)
                        self.num_of_data += 1
                        self.dataset[self.num_of_data] = details

    def __len__(self):
        return self.num_of_data

    # Preprocess the segmentation mask. Output only 1 object semantic information.
    def preprocess_segmentation(self, filename, bounding_box, pad=True):
        
        #read the semantic mask
        segment_mask = Image.open(self.voc_root + 'SegmentationObject/' + filename + '.png')
        segment_mask_np = torchvision.transforms.functional.pil_to_tensor(segment_mask)

        #Crop the segmentation based on the bounding box
        xmin, ymin = int(bounding_box[0]), int(bounding_box[1])
        xmax, ymax = int(bounding_box[2]), int(bounding_box[3])
        cropped_mask = segment_mask.crop((xmin, ymin, xmax, ymax))
        cropped_mask_np = torchvision.transforms.functional.pil_to_tensor(cropped_mask)
                                 
        #Count the majority element
        bincount = np.bincount(cropped_mask_np.reshape(-1))
        bincount[0] = 0 #Remove the black pixel
        if (bincount.shape[0] >= 256):
            bincount[255] = 0 #Remove the white pixel
        majority_element = bincount.argmax()
        
        #Based on the majority element, binary mask the segmentation
        segment_mask_np[np.where((segment_mask_np != 0) & (segment_mask_np != majority_element))] = 0
        segment_mask_np[segment_mask_np == majority_element] = 1

        #Pad the segment mask to 1024x1024 (for batching in dataloader)
        if pad:
            segment_mask_np = pad_image(segment_mask_np)

        return segment_mask_np

    # Preprocess the image to an appropriate foramt for SAM
    def preprocess_image(self, img):
        # ~= predictor.py - set_image()
        img = np.array(img)
        input_image = self.resizelongestside.apply_image(img)
        input_image_torch = torch.as_tensor(input_image, device='cpu')
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std #normalize
        original_size = img.shape[:2]
        input_size = tuple(input_image_torch.shape[-2:])
       
        return pad_image(input_image_torch), original_size, input_size

    def __getitem__(self, idx):
        data = self.dataset[idx]
        filename, classname = data[0], data[1]
        bounding_box = data[2]

        # No padding + preprocessing
        mask_gt = self.preprocess_segmentation(filename, bounding_box, pad=False)

        image, original_size, input_size = self.preprocess_image(Image.open(self.voc_root + 'JPEGImages/' + filename + '.jpg')) # read the image
        prompt  = bounding_box # bounding box - input_boxes x1, y1, x2, y2
        training_data = {}
        training_data['image'] = image
        training_data["original_size"] = original_size
        training_data["input_size"] = input_size
        training_data["ground_truth_mask"] = mask_gt
        training_data["prompt"] = prompt
        return (training_data, mask_gt) #data, label


class INC_SAMVOC2012Dataloader:
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        self.dataset = []
        ds = INC_SAMVOC2012Dataset(kwargs['voc_root'], kwargs['type'])
        # operations to add (input_data, label) pairs into self.dataset
        for i in range(len(ds)):
            self.dataset.append(ds[i])


    def __iter__(self):
        for input_data, label in self.dataset:
            yield input_data, label