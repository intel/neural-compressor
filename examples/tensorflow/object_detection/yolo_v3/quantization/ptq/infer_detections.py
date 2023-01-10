import time
import numpy as np
import tensorflow as tf

from absl import app, flags
from neural_compressor.metric import COCOmAPv2
from neural_compressor.data import TensorflowResizeWithRatio, ParseDecodeCocoTransform
from neural_compressor.data import Postprocess, LabelBalanceCOCORecordFilter
from tensorflow.python.client import timeline
from coco_constants import LABEL_MAP
from utils import read_graph, non_max_suppression

flags.DEFINE_integer('batch_size', 1, "batch size")

flags.DEFINE_string("ground_truth", None, "ground truth file")

flags.DEFINE_string("input_graph", None, "input graph")

flags.DEFINE_string("output_graph", None, "input graph")

flags.DEFINE_string("config", None, "Neural Compressor config file")

flags.DEFINE_string("dataset_location", None, "Location of Dataset")

flags.DEFINE_float("conf_threshold", 0.5, "confidence threshold")

flags.DEFINE_float("iou_threshold", 0.4, "IoU threshold")

flags.DEFINE_integer("num_intra_threads", 0, "number of intra threads")

flags.DEFINE_integer("num_inter_threads", 1, "number of inter threads")

flags.DEFINE_boolean("benchmark", False, "benchmark mode")

flags.DEFINE_boolean("profiling", False, "Signal of profiling")

FLAGS = flags.FLAGS


class NMS():
    def __init__(self, conf_threshold, iou_threshold):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def __call__(self, sample):
        preds, labels = sample
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        filtered_boxes = non_max_suppression(preds,
                                             self.conf_threshold,
                                             self.iou_threshold)

        det_boxes = []
        det_scores = []
        det_classes = []
        for cls, bboxs in filtered_boxes.items():
            det_classes.extend([LABEL_MAP[cls + 1]] * len(bboxs))
            for box, score in bboxs:
                rect_pos = box.tolist()
                y_min, x_min = rect_pos[1], rect_pos[0]
                y_max, x_max = rect_pos[3], rect_pos[2]
                height, width = 416, 416
                det_boxes.append(
                    [y_min / height, x_min / width, y_max / height, x_max / width])
                det_scores.append(score)

        if len(det_boxes) == 0:
            det_boxes = np.zeros((0, 4))
            det_scores = np.zeros((0, ))
            det_classes = np.zeros((0, ))

        return [np.array([det_boxes]), np.array([det_scores]), np.array([det_classes])], labels


def create_tf_config():
    config = tf.compat.v1.ConfigProto()
    config.intra_op_parallelism_threads = FLAGS.num_intra_threads
    config.inter_op_parallelism_threads = FLAGS.num_inter_threads
    return config


def run_benchmark():
    config = create_tf_config()

    graph_def = read_graph(FLAGS.input_graph)

    tf.import_graph_def(graph_def, name='')

    input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('inputs:0')
    output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('output_boxes:0')

    dummy_data_shape = list(input_tensor.shape)
    dummy_data_shape[0] = FLAGS.batch_size
    dummy_data = np.random.random(dummy_data_shape).astype(np.float32)

    if FLAGS.profiling != True:
        num_warmup = 200
        total_iter = 1000
    else:
        num_warmup = 20
        total_iter = 100

    total_time = 0.0

    with tf.compat.v1.Session(config=config) as sess:
        print("Running warm-up")
        for i in range(num_warmup):
            sess.run(output_tensor, {input_tensor: dummy_data})
        print("Warm-up complete")

        for i in range(1, total_iter + 1):
            start_time = time.time()
            sess.run(output_tensor, {input_tensor: dummy_data})
            end_time = time.time()

            if i % 10 == 0:
                print(
                    "Steps = {0}, {1:10.6f} samples/sec".format(i, FLAGS.batch_size / duration))

            duration = end_time - start_time
            total_time += duration

        if FLAGS.profiling:
            options = tf.compat.v1.RunOptions(
                trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()

            sess.run(output_tensor, {input_tensor: dummy_data},
                     options=options, run_metadata=run_metadata)

            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open("timeline_%s.json" % (time.time()), 'w') as f:
                f.write(chrome_trace)

    throughput = total_iter * FLAGS.batch_size / total_time
    print("Batch size = {}".format(FLAGS.batch_size))
    print("Latency: {} ms".format(1 / throughput * 1000))
    print("Throughput: {} samples/sec".format(throughput))

def evaluate(model):
    """Custom evaluate function to estimate the accuracy of the model.

    Args:
        model (tf.Graph_def): The input model graph
        
    Returns:
        accuracy (float): evaluation result, the larger is better.
    """
    from neural_compressor.model import Model
    model = Model(model)
    model.input_tensor_names = ["inputs"]
    model.output_tensor_names = ["output_boxes:0"]
    input_tensor = model.input_tensor
    output_tensor = model.output_tensor if len(model.output_tensor)>1 else \
                        model.output_tensor[0]
    iteration = -1
    if args.benchmark and args.mode == 'performance':
        iteration = 100
    kwargs = {'conf_threshold': FLAGS.conf_threshold,
                  'iou_threshold': FLAGS.iou_threshold}
    postprocess = Postprocess(NMS, 'NMS', **kwargs)
    metric = COCOmAPv2(map_key='DetectionBoxes_Precision/mAP@.50IOU')

    def eval_func(dataloader):
        latency_list = []
        for idx, (inputs, labels) in enumerate(dataloader):
            # dataloader should keep the order and len of inputs same with input_tensor
            inputs = np.array([inputs])
            feed_dict = dict(zip(input_tensor, inputs))

            start = time.time()
            predictions = model.sess.run(output_tensor, feed_dict)
            end = time.time()

            predictions, labels = postprocess((predictions, labels))
            metric.update(predictions, labels)
            latency_list.append(end-start)
            if idx + 1 == iteration:
                break
        latency = np.array(latency_list).mean() / args.batch_size
        return latency

    eval_dataset = COCORecordDataset(root=FLAGS.dataset_location, filter=LabelBalanceCOCORecordFilter(size=1), \
        transform=ComposeTransform(transform_list=[ParseDecodeCocoTransform(), 
            TensorflowResizeWithRatio(min_dim=416, max_dim=416, padding=True, constant_value=128)]))
    eval_dataloader=DataLoader(framework='tensorflow', dataset=eval_dataset, batch_size=args.batch_size)

    latency = eval_func(eval_dataloader)
    if args.benchmark and args.mode == 'performance':
        print("Batch size = {}".format(args.batch_size))
        print("Latency: {:.3f} ms".format(latency * 1000))
        print("Throughput: {:.3f} images/sec".format(1. / latency))
    acc = metric.result()
    return acc 

def main(_):
    calib_dataset = COCORecordDataset(root=FLAGS.dataset_location, filter=LabelBalanceCOCORecordFilter(size=1), \
        transform=ComposeTransform(transform_list=[ParseDecodeCocoTransform(), 
                            TensorflowResizeWithRatio(min_dim=416, max_dim=416, padding=True)]))
    calib_dataloader = DataLoader(framework='tensorflow', dataset=calib_dataset, batch_size=FLAGS.batch_size)

    if FLAGS.benchmark:
        run_benchmark()
    else:
        from neural_compressor import quantization
        from neural_compressor.config import PostTrainingQuantConfig
        op_name_list={
            'detector/yolo-v3/Conv_6/Conv2D': {'activation':  {'dtype': ['fp32']}},
            'detector/yolo-v3/Conv_14/Conv2D': {'activation':  {'dtype': ['fp32']}},
            'detector/yolo-v3/Conv_22/Conv2D': {'activation':  {'dtype': ['fp32']}}
            }
        config = PostTrainingQuantConfig(
            inputs=["inputs"],
            outputs=["output_boxes"],
            calibration_sampling_size=[2],
            op_name_list=op_name_list)
        q_model = quantization.fit(model=FLAGS.input_graph, conf=config, 
                                    calib_dataloader=calib_dataloader, eval_func=evaluate)
        q_model.save(FLAGS.output_graph)


if __name__ == '__main__':
    app.run(main)
