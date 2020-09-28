import os
import sys
import time
import logging
import argparse
import yaml
import math
import numpy as np

from tensorflow.python.client import timeline
from tensorflow.python.tools import optimize_for_inference_lib
from find_outputs import get_input_output
from utils import *

logging.basicConfig(level=logging.INFO,
                    datefmt='[%H:%M:%S]',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OOB-Benchmark")


if "PRINT_LATENCY" in os.environ and int(os.environ["PRINT_LATENCY"]) == 1:
    PRINT_LATENCY = True
else:
    PRINT_LATENCY = False

if "RUN_PROFILING" in os.environ and int(os.environ["RUN_PROFILING"]) == 1:
    RUN_PROFILING = True
else:
    RUN_PROFILING = False

if "CHECK_ACCURACY" in os.environ and os.environ["CHECK_ACCURACY"] == "1":
    CHECK_ACCURACY = True
else:
    CHECK_ACCURACY = False

if "BS" in os.environ:
    BATCH_SIZE = int(os.environ["BS"])
else:
    BATCH_SIZE = 1

if "NUM_INSTANCES" in os.environ:
    NUM_INSTANCES = int(os.environ["NUM_INSTANCES"])
else:
    NUM_INSTANCES = 1

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


def initialize_graph(model_details, enable_optimize_for_inference):
    graph = tf_v1.Graph()
    with graph.as_default():

        od_graph_def = tf_v1.GraphDef()
        with tf_v1.gfile.GFile(os.path.join(os.getcwd(), model_details['model_dir']), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            od_graph_def = delete_assign(od_graph_def)
        
        # optimize for inference
        if enable_optimize_for_inference:
            # optimize graph for inference
            input_list = [in_name for in_name,
                            val in model_details['input'].items()]
            output_list = [
                out_name for out_name in model_details['output']]
            input_data_type = [tf_v1.convert_to_tensor(item).dtype.as_datatype_enum for item in model_details['input'].values()]

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
                  enable_optimize_for_inference):
    tf_config = create_tf_config()
    graph = initialize_graph(model_details, enable_optimize_for_inference)
    run_options = tf_v1.RunOptions(trace_level=tf_v1.RunOptions.FULL_TRACE)
    run_metadata = tf_v1.RunMetadata()
    # graph = initialize_graph(model_details, enable_optimize_for_inference)
    with tf_v1.Session(config=tf_config, graph=graph) as sess:
        output_dict = {out_name: graph.get_tensor_by_name(out_name + ':0')
                       for out_name in model_details['output']}

        input_dict = {
            graph.get_tensor_by_name(in_name + ':0'): val
            for in_name, val in model_details['input'].items()}

        # sess.run(tf_v1.global_variables_initializer())

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

            if RUN_PROFILING:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                with open('timeline_' + str(rep + 1) + '.json', 'w') as trace_file:
                    trace_file.write(
                        trace.generate_chrome_trace_format(show_memory=False))

            end = time.time()
            delta = end - start
            total_time += delta
            reps_done += 1

        avg_time = total_time / reps_done
        latency = avg_time * 1000
        throughput = 1.0 / avg_time * BATCH_SIZE * NUM_INSTANCES

        print('Batch size = %d' % BATCH_SIZE) 
        print('Latency: %.3f ms' % (latency)) 
        print('Throughput: %.3f images/sec' % throughput)

def _write_inputs_outputs_to_yaml(yaml_path, inputs, outputs):
    # deal with the inputs/outputs at yaml
    with open(yaml_path, 'r') as f:
        # content = f.read()
        # content = yaml.load(content, yaml.Loader)
        content = yaml.load(f)

        print(type(content))
        print(content)
        tmp_i = ''
        tmp_o = ''
        for item in inputs:
            tmp_i = tmp_i + str(item) + ','
        for item in outputs:
            tmp_o = tmp_o + str(item) + ','
        content['framework'].update({'inputs': tmp_i[:-1]})
        content['framework'].update({'outputs': tmp_o[:-1]})
        print(content)

    with open(yaml_path, 'w') as nf:
        yaml.dump(content, nf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--model_path", help="path of model")
    parser.add_argument("--model_name", help="name of model")
    parser.add_argument("-n", "--num_iter", type=int, default=500,
                        help="numbers of inference iteration, default is 500")
    parser.add_argument("--num_warmup", type=int, default=10,
                        help="numbers of warmup iteration, default is 10")
    parser.add_argument("--disable_optimize", action='store_false',
                        help="use this to disable optimize_for_inference")
    parser.add_argument("--is_meta", action='store_true',
                        help="input a meta file")
    parser.add_argument("--tune", action='store_true', help="Do ilit optimize.")
    parser.add_argument("--benchmark", action='store_true', help="Do ilit optimize.")
    parser.add_argument("--ilit_config_file", type=str, help="config file path", 
                        default='./config.yaml')
    parser.add_argument("--topology", type=str, help="topology")
    parser.add_argument("--output_path", help="path of ilit convert model")
    
    args = parser.parse_args()

    num_iter = args.num_iter
    num_warmup = args.num_warmup
    enable_optimize = args.disable_optimize
    # model_path = args.model_path
    is_meta = args.is_meta

    # benchmark PB model directly
    if args.model_path and not args.model_name:
        # generate model detail
        model_dir = args.model_path
        model_detail = {}
        model_input_output = get_input_output(model_dir, is_meta)
        # ckpt/meta model will save freezed pb in the same dir
        model_dir = model_dir if not is_meta else args.model_path[:-5] + "_freeze.pb"
        output = model_input_output['outputs']
        input_dic = {}
        for _input in model_input_output['inputs']:
            # deal with bool dtype input
            if model_input_output['inputs'][_input]['type'] == 'bool':
                input_dic[_input] = model_input_output['inputs'][_input]['value']
                logger.info("Find benchmark input name: {}, dtype: {}".format(_input, model_input_output['inputs'][_input]['type']))
            elif _input == 'dropout_keep_prob':
                input_dic[_input] = np.array([0.5,], dtype='float32')
            else:
                dtype = model_input_output['inputs'][_input]['type']
                dshape = model_input_output['inputs'][_input]['shape']
                dummy_input = generate_data(dshape, dtype, BATCH_SIZE)
                input_dic[_input] = dummy_input
                logger.info("Find benchmark input name: {}, dtype: {}, shape: {}"
                            .format(_input, dtype, dummy_input.shape))
        model_detail['model_dir'] = model_dir
        model_detail['input'] = input_dic
        model_detail['output'] = output
        model_detail['ckpt'] = args.is_meta


    # benchmark with input/output
    elif args.model_name:
        assert args.model_path is not None, "Model path is undefined."
        from model_detail import models
        model_detail = None
        for model in models:
            if model['model_name'] == args.model_name:
                model_detail = model
                model_detail['model_dir'] = args.model_path
                model_detail['ckpt'] = args.is_meta
                break

        if not model_detail:
            logger.error("Model undefined.")
            sys.exit(1)

    input_shapes = []
    for input_tensor in model_detail['input'].values():
        if not isinstance(input_tensor, bool):
            input_shapes.append(list(input_tensor.shape))
        else:
            input_shapes.append(['bool'])

    logger.info("***** Final benchmark input name: {}, shape: {}".format( \
                model_detail['input'].keys(), input_shapes))
    logger.info("***** Final benchmark output name: {}".format(model_detail['output']))

    if args.tune:
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        import ilit
        from ilit.adaptor.tf_utils.util import write_graph

        inputs = model_detail['input']
        outputs = model_detail['output']
        _write_inputs_outputs_to_yaml(args.ilit_config_file, list(inputs.keys()), outputs)

        graph = initialize_graph(model_detail, enable_optimize)
        tuner = ilit.Tuner(args.ilit_config_file)

        inputs_shape = [item.shape for item in inputs.values()]
        inputs_dtype = [str(item.dtype) for item in inputs.values()]        
        
        # generate dummy data
        dataset = ilit.data.datasets.DATASETS('tensorflow')['dummy'](shape=inputs_shape, 
                                                low=1.0, high=20.0, dtype=inputs_dtype)
        label_dataset = ilit.data.datasets.DATASETS('tensorflow')['dummy'](shape=[(1, 5)])
        data_loader=ilit.data.DataLoader('tensorflow', dataset=(dataset, label_dataset), batch_size=1)

        q_model = tuner.tune(graph, q_dataloader=data_loader)
        write_graph(q_model.as_graph_def(), args.output_path)
        
        # benchmark generator ilit int8 model
        model_detail['model_dir'] = args.output_path
        run_benchmark(model_detail, num_iter, num_warmup, enable_optimize)

    else:
        run_benchmark(model_detail, num_iter, num_warmup, enable_optimize)
