import time
import numpy as np
import tensorflow as tf

from absl import app, flags

from tensorflow.python.client import timeline
from coco_constants import LABEL_MAP
from utils import read_graph, non_max_suppression

flags.DEFINE_integer('batch_size', 1, "batch size")

flags.DEFINE_string("ground_truth", None, "ground truth file")

flags.DEFINE_string("input_graph", None, "input graph")

flags.DEFINE_string("output_graph", None, "input graph")

flags.DEFINE_string("config", None, "Neural Compressor config file")

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


def main(_):
    if FLAGS.benchmark:
        run_benchmark()
    else:
        FLAGS.batch_size = 1
        from neural_compressor.experimental import Quantization, common
        quantizer = Quantization(FLAGS.config)
        quantizer.model = common.Model(FLAGS.input_graph)
        kwargs = {'conf_threshold': FLAGS.conf_threshold,
                  'iou_threshold': FLAGS.iou_threshold}
        # from neural_compressor.data.transforms.postprocess import Postprocess
        # quantizer.postprocess = Postprocess(NMS, 'NMS', **kwargs)
        quantizer.postprocess = common.Postprocess(NMS, 'NMS', **kwargs)
        q_model = quantizer.fit()
        q_model.save(FLAGS.output_graph)


if __name__ == '__main__':
    app.run(main)
