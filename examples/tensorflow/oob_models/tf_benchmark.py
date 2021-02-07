import os
import sys
import time
import logging
import argparse
import math
import yaml
import numpy as np

from tensorflow.python.client import timeline
from tensorflow.python.tools import optimize_for_inference_lib
from find_outputs import get_input_output
from utils import *

logging.basicConfig(level=logging.INFO,
                    datefmt='[%H:%M:%S]',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OOB-Benchmark")


if "RUN_PROFILING" in os.environ and int(os.environ["RUN_PROFILING"]) == 1:
    RUN_PROFILING = True
else:
    RUN_PROFILING = False

if "BS" in os.environ:
    BATCH_SIZE = int(os.environ["BS"])
else:
    BATCH_SIZE = 1

if "OMP_NUM_THREADS" in os.environ:
    NUM_THREADS = int(os.environ["OMP_NUM_THREADS"])
else:
    NUM_THREADS = 28

def metrics_generator(array, tolerance):
    max_diff = np.max(array)
    mean_diff = np.mean(array)
    median_diff = np.median(array)
    success_rate = np.sum(array < tolerance) / array.size
    return max_diff, mean_diff, median_diff, success_rate

def initialize_graph(model_details, disable_optimize_for_inference):
    graph = tf_v1.Graph()
    with graph.as_default():

        od_graph_def = tf_v1.GraphDef()
        with tf_v1.gfile.GFile(os.path.join(os.getcwd(), model_details['model_path']), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            od_graph_def = delete_assign(od_graph_def)

        # optimize for inference
        if not disable_optimize_for_inference:
            # optimize graph for inference
            input_list = [in_name for in_name,
                          val in model_details['input'].items()]
            output_list = [
                out_name for out_name in model_details['output']]
            input_data_type = [tf_v1.convert_to_tensor(item).dtype.as_datatype_enum
                               for item in model_details['input'].values()]

            od_graph_def = optimize_for_inference_lib.optimize_for_inference(
                od_graph_def,  # inputGraph,
                input_list,  # an array of the input nodes
                output_list,  # an array of output nodes
                input_data_type)

        tf_v1.import_graph_def(od_graph_def, name='')

    return graph


def create_tf_config():
    config = tf_v1.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = NUM_THREADS
    config.inter_op_parallelism_threads = 1
    return config


def run_benchmark(model_details, max_reps, num_warmup,
                  disable_optimize_for_inference, batch_size,timeline_dir):
    tf_config = create_tf_config()
    graph = initialize_graph(model_details, disable_optimize_for_inference)
    run_options = tf_v1.RunOptions(trace_level=tf_v1.RunOptions.FULL_TRACE)
    run_metadata = tf_v1.RunMetadata()

    with tf_v1.Session(config=tf_config, graph=graph) as sess:
        output_dict = {out_name: graph.get_tensor_by_name(out_name + ':0')
                       for out_name in model_details['output']}

        input_dict = {
            graph.get_tensor_by_name(in_name + ':0'): val
            for in_name, val in model_details['input'].items()}

        total_time = 0.0
        reps_done = 0
        for rep in range(max_reps):
            if rep < num_warmup:
                _ = sess.run(output_dict, feed_dict=input_dict)
                continue
            start = time.time()
            if RUN_PROFILING:
                _ = sess.run(output_dict, feed_dict=input_dict, options=run_options,
                             run_metadata=run_metadata)
            else:
                _ = sess.run(output_dict, feed_dict=input_dict)

            end = time.time()
            delta = end - start
            total_time += delta
            reps_done += 1
            if RUN_PROFILING:
                if rep==max_reps/2:
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    timeline_name=model_detail['model_path'].split('/')[-1].split('.')[0]
                    timeline_path=timeline_dir+'/'+timeline_name
                    if not os.path.exists(timeline_path):
                        os.makedirs(timeline_path)
                    with open(timeline_path+ '/' + timeline_name + '_' + str(rep)
                              + '_' + str(os.getpid())+ '.json', 'w') as trace_file:
                        trace_file.write(
                            trace.generate_chrome_trace_format(show_memory=False))
        avg_time = total_time / reps_done
        latency = avg_time * 1000
        throughput = 1.0 / avg_time * batch_size

        print('Batch size = %d' % batch_size)
        print('Latency: %.3f ms' % (latency))
        print('Throughput: %.3f images/sec' % throughput)

def _write_inputs_outputs_to_yaml(yaml_path, output_yaml_path, inputs, outputs):
    # deal with the inputs/outputs at yaml
    with open(yaml_path, 'r') as f:
        # content = f.read()
        # content = yaml.load(content, yaml.Loader)
        content = yaml.load(f)

        tmp_i = ''
        tmp_o = ''
        for item in inputs:
            tmp_i = tmp_i + str(item) + ','
        for item in outputs:
            tmp_o = tmp_o + str(item) + ','
        content['model'].update({'inputs': tmp_i[:-1]})
        content['model'].update({'outputs': tmp_o[:-1]})
        print(content)

    with open(output_yaml_path, 'w') as nf:
        yaml.dump(content, nf)

def oob_collate_data_func(batch):
    """Puts each data field into a pd frame with outer dimension batch size"""
    elem = batch[0]
    import collections
    if isinstance(elem, collections.abc.Mapping):
        return {key: oob_collate_data_func([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, collections.abc.Sequence):
        batch = zip(*batch)
        return [oob_collate_data_func(samples) for samples in batch]
    elif isinstance(elem, np.ndarray):
        return np.stack(batch)
    elif elem in (True,False):
        return elem
    else:
        return batch
class DataLoader(object):
    def __init__(self, inputs_tensor, total_samples, batch_size):
        """dataloader generator

        Args:
            data_location (str): tf recorder local path
            batch_size (int): dataloader batch size
        """
        self.batch_size = batch_size
        self.inputs_tensor = inputs_tensor
        # self.input_dtypes = input_dtypes
        self.total_samples = total_samples
        self.n = math.ceil(float(self.total_samples) / self.batch_size)
        # assert len(input_shapes) == len(input_dtypes)
        print("batch size is " + str(self.batch_size) + "," + str(self.n) + " iteration")

    def __iter__(self):
        for _ in range(self.n):
            if len(self.inputs_tensor.values()) > 1:
                data = [list(self.inputs_tensor.values())]
            else:
                data = list(self.inputs_tensor.values())
            yield data

def eval_func(graph):
    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--model_path", help="path of model")
    parser.add_argument("--model_name", help="name of model")
    parser.add_argument("-n", "--num_iter", type=int, default=500,
                        help="numbers of inference iteration, default is 500")
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="numbers of warmup iteration, default is 10")
    parser.add_argument("--disable_optimize", action='store_true',
                        help="use this to disable optimize_for_inference")
    parser.add_argument("--is_meta", action='store_true',
                        help="input a meta file")
    parser.add_argument("--tune", action='store_true', help="Do lpot optimize.")
    parser.add_argument("--benchmark", action='store_true', help="Do lpot optimize.")
    parser.add_argument("--yaml", type=str, help="config yaml file of lpot.",
                        default='./config.yaml')
    parser.add_argument("--timeline", type=str, help="timeline of lpot convert model",
                        default='./timeline_json')
    parser.add_argument("--topology", type=str, help="topology")
    parser.add_argument("--output_path", help="path of lpot convert model")

    args = parser.parse_args()

    num_iter = args.num_iter
    num_warmup = args.num_warmup
    disable_optimize = args.disable_optimize
    is_meta = args.is_meta

    # benchmark PB model directly
    if args.model_path and not args.model_name:
        # generate model detail
        model_path = args.model_path
        model_detail = {}
        model_input_output = get_input_output(model_path, is_meta)
        # ckpt/meta model will save freezed pb in the same dir
        model_path = model_path if not is_meta else args.model_path[:-5] + "_freeze.pb"
        output = model_input_output['outputs']
        input_dic = {}
        for _input in model_input_output['inputs']:
            # deal with bool dtype input
            if model_input_output['inputs'][_input]['type'] == 'bool':
                input_dic[_input] = model_input_output['inputs'][_input]['value']
                logger.info("Find benchmark input name: {}, dtype: {}".format(
                            _input, model_input_output['inputs'][_input]['type']))
            elif _input == 'dropout_keep_prob':
                input_dic[_input] = np.array([0.5,], dtype='float32')
            else:
                dtype = model_input_output['inputs'][_input]['type']
                dshape = model_input_output['inputs'][_input]['shape']
                dummy_input = generate_data(dshape, dtype, BATCH_SIZE)
                input_dic[_input] = dummy_input
                logger.info("Find benchmark input name: {}, dtype: {}, shape: {}"
                            .format(_input, dtype, dummy_input.shape))
        model_detail['model_path'] = model_path
        model_detail['input'] = input_dic
        model_detail['output'] = output
        model_detail['ckpt'] = is_meta

    # benchmark with input/output
    elif args.model_name:
        assert args.model_path is not None, "Model path is undefined."
        from model_detail import models
        model_detail = None
        for model in models:
            if model['model_name'] == args.model_name:
                model_detail = model
                model_detail['model_path'] = args.model_path
                model_detail['ckpt'] = is_meta
                break

        if not model_detail:
            logger.error("Model undefined.")
            sys.exit(1)

    inputs_shape = []
    inputs_dtype = []
    for input_tensor in model_detail['input'].values():
        if not isinstance(input_tensor, bool):
            inputs_shape.append(input_tensor.shape)
            inputs_dtype.append(str(input_tensor.dtype))
        else:
            # TODO: wait scalar support in dummy dataset
            inputs_shape.append((1,))
            inputs_dtype.append('bool')
    logger.info("***** Final benchmark input name: {}, shape: {}".format( \
                model_detail['input'].keys(), inputs_shape))
    logger.info("***** Final benchmark output name: {}".format(model_detail['output']))
    batch_size = inputs_shape[0][0]

    if args.tune:
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        from lpot import Quantization
        inputs = model_detail['input']
        outputs = model_detail['output']
        _write_inputs_outputs_to_yaml(args.yaml, "./config_tmp.yaml", list(inputs.keys()), outputs)

        quantizer = Quantization("./config_tmp.yaml")
        # generate dummy data
        dataset = quantizer.dataset(dataset_type='dummy', shape=inputs_shape,
                                low=1.0, high=20.0, dtype=inputs_dtype, label=True)
        data_loader = quantizer.dataloader(dataset=dataset,
                                           batch_size=batch_size,
                                           collate_fn=oob_collate_data_func)
        q_model = quantizer(args.model_path, q_dataloader=data_loader)
        q_model.save(args.output_path)

    else:
        run_benchmark(model_detail, num_iter, num_warmup,
                      disable_optimize, batch_size, args.timeline)
