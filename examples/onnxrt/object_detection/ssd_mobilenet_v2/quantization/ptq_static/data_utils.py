import numpy as np
import collections
from PIL import Image
import os
import yaml
from pycocotools.coco import COCO
import cv2

class SequentialSampler():
    def __init__(self, dataset):
        self.whole_dataset = dataset

    def __iter__(self):
        self.process_rank = 0 # The default rank is 0, which represents the main process
        self.process_size = 1 # By default, process_size=1, only the main process is running
        return iter(range(self.process_rank, len(self.whole_dataset), self.process_size))

    def __len__(self):
        return len(self.whole_dataset)

class BatchSampler():
    def __init__(self, sampler, batch_size, drop_last=True):
        if isinstance(drop_last, bool):
            self.drop_last = drop_last
        else:
            raise ValueError("last_batch only support bool as input")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class IndexFetcher():
    def __init__(self, dataset, collate_fn, drop_last):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def __call__(self, batched_indices):
        data = [self.dataset[idx] for idx in batched_indices]
        return self.collate_fn(data)


def default_collate(batch):
    """Merge data with outer dimension batch size."""
    elem = batch[0]
    if isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [default_collate(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        try:
            return np.stack(batch)
        except:
            return batch
    else:
        return batch

class COCORawDataloader():
    def __init__(self, dataset, batch_size=1, last_batch='rollover', collate_fn=None,
                 sampler=None, batch_sampler=None, num_workers=0, pin_memory=False,
                 shuffle=False):
        self.dataset = dataset
        self.last_batch = last_batch
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = False if last_batch == 'rollover' else True
        if self.collate_fn == None:
            self.collate_fn = default_collate

    def __iter__(self):
        """Yield data in iterative order."""
        return self._generate_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            last_batch=self.last_batch,
            collate_fn=self.collate_fn,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle)

    def _generate_dataloader(self, dataset, batch_size, last_batch, collate_fn, sampler,
                             batch_sampler, num_workers, pin_memory, shuffle):

        sampler = self._generate_sampler(dataset)
        self.batch_sampler = BatchSampler(sampler, batch_size, self.drop_last)
        self.fetcher = IndexFetcher(dataset, collate_fn, self.drop_last)

        for batched_indices in self.batch_sampler:
            try:
                data = self.fetcher(batched_indices)
                yield data
            except StopIteration:
                return

    def _generate_sampler(self, dataset):
        if hasattr(dataset, "__getitem__"):
            self.dataset_type = 'index'
            return SequentialSampler(dataset)
        else:
            raise ValueError("dataset type only support (index, iter)")


class COCORawDataset():
    """Coco raw dataset.
    Please arrange data in this way:
        root
          ├──  1.jpg
          ├──  2.jpg
               ...
          └──  n.jpg
        anno_dir
    Please use Resize transform when batch_size > 1
    Args: root (str): Root directory of dataset.
          anno_dir (str, default='annotations/instances_val2017.json'): annotation file directory.
          transform (transform object, default=None):  transform to process input data.
          filter (Filter objects, default=None): filter out examples according 
                                                 to specific conditions.
    """

    def __init__(self, root, \
            anno_dir='annotations/instances_val2017.json', transform=None, filter=None):
        """Initialize the attributes of class."""
        self.batch_size = 1
        self.image_list = []
        self.transform = transform
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
                bbox = [bbox[0]/float(pic_width), bbox[1]/float(pic_height),\
                    bbox[2]/float(pic_width), bbox[3]/float(pic_height)]
                bboxes.append([bbox[1], bbox[0], bbox[1]+bbox[3], bbox[0]+bbox[2]])
                labels.append(coco.cats[ann['category_id']]['name'].encode('utf8'))
            img_file = os.path.join(img_path, img_detail['file_name'])
            if not os.path.exists(img_file) or len(bboxes) == 0:
                continue

            if filter and not filter(None, bboxes):
                continue

            with Image.open(img_file) as image:
                image = np.array(image.convert('RGB'))
            self.image_list.append(
                (image, [np.array(bboxes), np.array(labels), np.array([]),\
                 np.array(img_detail['file_name'].encode('utf-8'))]))

    def __len__(self):
        """Length of the dataset."""
        return len(self.image_list)

    def __getitem__(self, index):
        """Magic method.
        x[i] is roughly equivalent to type(x).__getitem__(x, index)
        """
        sample = self.image_list[index]
        if self.transform is not None:
            sample= self.transform(sample)
        return sample

interpolation_map = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
}

class ResizeTransform():
    def __init__(self, size, interpolation='bilinear'):
        if isinstance(size, int):
            self.size = size, size
        elif isinstance(size, list):
            if len(size) == 1:
                self.size = size[0], size[0]
            elif len(size) == 2:
                self.size = size[0], size[1]

        if interpolation in interpolation_map.keys():
            self.interpolation = interpolation_map[interpolation]
        else:
            raise ValueError("Undefined interpolation type")

    def __call__(self, sample):
        image, label = sample
        image = cv2.resize(image, self.size, interpolation=self.interpolation)
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)
        return (image, label)

class RescaleTransform():
    """Scale the values of image to [0,1].
    Returns:
        tuple of processed image and label
    """

    def __call__(self, sample):
        """Scale the values of the image in sample."""
        image, label = sample
        if isinstance(image, np.ndarray):
            image = image.astype('float32') / 255.
        return (image, label)

class NormalizeTransform():
    def __init__(self, mean=[0.0], std=[1.0]):
        self.mean = mean
        self.std = std
        for item in self.std:
            if item < 10**-6:
                raise ValueError("Std should be greater than 0")

    def __call__(self, sample):
        image, label = sample
        assert len(self.mean) == image.shape[-1], 'Mean channel must match image channel'
        image = (image - self.mean) / self.std
        return (image, label)

class TransposeTransform():
    def __init__(self, perm):
        self.perm = perm

    def __call__(self, sample):
        image, label = sample
        assert len(image.shape) == len(self.perm), "Image rank doesn't match Perm rank"
        image = np.transpose(image, axes=self.perm)
        return (image, label)

np_dtype_map = {'int8': np.int8, 'uint8': np.uint8, 'complex64': np.complex64,
           'uint16': np.uint16, 'int32': np.int32, 'uint32': np.uint32,
           'int64': np.int64, 'uint64': np.uint64, 'float32': np.float32,
           'float16': np.float16, 'float64': np.float64, 'bool': bool,
           'string': str, 'complex128': np.complex128, 'int16': np.int16}

class CastTransform():
    def __init__(self, dtype='float32'):
        assert dtype in np_dtype_map.keys(), 'Unknown dtype'
        self.dtype = dtype

    def __call__(self, sample):
        image, label = sample
        image = image.astype(np_dtype_map[self.dtype])
        return (image, label)

class ComposeTransform():
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample):
        for transform in self.transform_list:
            sample = transform(sample)
        return sample

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
        preds[0][0][:, [0, 1, 2, 3]] = preds[0][0][:, [1, 0, 3, 2]]
        return preds, labels

class LabelBalanceCOCORawFilter(object):
    """The label balance filter for COCO raw data."""

    def __init__(self, size=1):
        """Initialize the attribute of class."""
        self.size = size

    def __call__(self, image, label):
        """Execute the filter.

        Args:
            image: Not used.
            label: label of a sample.
        """
        return len(label) == self.size