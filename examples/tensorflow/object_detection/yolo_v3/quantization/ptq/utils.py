import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.data.experimental import parallel_interleave
from tensorflow.python.data.experimental import map_and_batch

def read_graph(input_graph):
    if not gfile.Exists(input_graph):
      print("Input graph file '" + input_graph + "' does not exist!")
      exit(-1)

    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(input_graph, "rb") as f:
      data = f.read()
    input_graph_def.ParseFromString(data)
    
    return input_graph_def

def parse_and_preprocess(serialized_example):
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.compat.v1.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/object/class/text': tf.compat.v1.VarLenFeature(dtype=tf.string),
      'image/source_id': tf.compat.v1.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
  }
  sparse_float32 = tf.compat.v1.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.compat.v1.parse_single_example(serialized_example, feature_map)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  encoded_image = features['image/encoded']
  image_tensor = tf.image.decode_image(encoded_image, channels=3)
  image_tensor.set_shape([None, None, 3])

  label = features['image/object/class/text'].values

  image_id = features['image/source_id']

  return image_tensor, bbox[0], label, image_id

def get_input(data_location, batch_size=1):
    tfrecord_paths = [data_location]
    ds = tf.data.TFRecordDataset.list_files(tfrecord_paths)

    ds = ds.apply(
        parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=28, block_length=5,
          sloppy=True,
          buffer_output_elements=10000, prefetch_input_elements=10000))
    ds = ds.prefetch(buffer_size=10000)
    ds = ds.apply(
        map_and_batch(
          map_func=parse_and_preprocess,
          batch_size=batch_size,
          num_parallel_batches=28,
          num_parallel_calls=None))
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds_iter = tf.compat.v1.data.make_one_shot_iterator(ds)
    images, bbox, label, image_id = ds_iter.get_next()

    return images, bbox, label, image_id

def letter_box_image(image, output_height, output_width, fill_value):
    """
    Fit image with final image with output_width and output_height.
    :param image: PILLOW Image object.
    :param output_height: width of the final image.
    :param output_width: height of the final image.
    :param fill_value: fill value for empty area. Can be uint8 or np.ndarray
    :return: numpy image fit within letterbox. dtype=uint8, shape=(output_height, output_width)
    """

    height_ratio = float(output_height) / image.size[1]
    width_ratio = float(output_width) / image.size[0]
    fit_ratio = min(width_ratio, height_ratio)
    fit_height = int(image.size[1] * fit_ratio)
    fit_width = int(image.size[0] * fit_ratio)
    fit_image = np.asarray(image.resize((fit_width, fit_height), resample=Image.BILINEAR))

    if isinstance(fill_value, int):
        fill_value = np.full(fit_image.shape[2], fill_value, fit_image.dtype)

    to_return = np.tile(fill_value, (output_height, output_width, 1))
    pad_top = int(0.5 * (output_height - fit_height))
    pad_left = int(0.5 * (output_width - fit_width))
    to_return[pad_top: pad_top+fit_height, pad_left: pad_left+fit_width] = fit_image

    return to_return

def _iou(box1, box2):
    """
    Computes Intersection over Union value for 2 bounding boxes

    :param box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    :param box2: same as box1
    :return: IoU
    """
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = max(int_x1 - int_x0, 0) * max(int_y1 - int_y0, 0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
    """
    Applies Non-max suppression to prediction boxes.

    :param predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    :param confidence_threshold: the threshold for deciding if prediction is valid
    :param iou_threshold: the threshold for deciding if two boxes overlap
    :return: dict: class -> [(box, score)]
    """
    conf_mask = np.expand_dims(
        (predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        tmp = image_pred
        sum_t = np.sum(tmp, axis=1)
        non_zero_idxs = sum_t != 0
        image_pred = image_pred[non_zero_idxs, :]
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if cls not in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result

def letter_box_pos_to_original_pos(letter_pos, current_size, ori_image_size)-> np.ndarray:
    """
    Parameters should have same shape and dimension space. (Width, Height) or (Height, Width)
    :param letter_pos: The current position within letterbox image including fill value area.
    :param current_size: The size of whole image including fill value area.
    :param ori_image_size: The size of image before being letter boxed.
    :return:
    """
    letter_pos = np.asarray(letter_pos, dtype=np.float)
    current_size = np.asarray(current_size, dtype=np.float)
    ori_image_size = np.asarray(ori_image_size, dtype=np.float)
    final_ratio = min(current_size[0] / ori_image_size[0], current_size[1] / ori_image_size[1])
    pad = 0.5 * (current_size - final_ratio * ori_image_size)
    pad = pad.astype(np.int32)
    to_return_pos = (letter_pos - pad) / final_ratio

    return to_return_pos

