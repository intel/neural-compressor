import subprocess  # nosec B404
import mlperf_loadgen as lg
import argparse
import os
import logging
import sys
import hashlib
import time
from SUT import SUT, SUTServer

sys.path.insert(0, os.getcwd())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["Offline", "Server", "SingleStream"], default="Offline", help="Scenario")
    parser.add_argument("--model-path", type=str, default="meta-llama/Llama-2-70b-chat-hf", help="Model name")
    parser.add_argument("--workload-name", type=str, default="llama2-70b")
    parser.add_argument("--dataset-path", type=str, default="/software/users/mlperf/datasets/open_orca_gpt4_tokenized_llama.sampled_24576.pkl", help="") # TODO: Reset default before submission
    parser.add_argument("--accuracy", action="store_true", help="Run accuracy mode")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="data type of the model, choose from float16, bfloat16 and float32")
    parser.add_argument("--device", type=str, default="hpu", help="device to use")
    parser.add_argument("--audit-conf", type=str, default="audit.conf", help="audit config for LoadGen settings during compliance runs")
    parser.add_argument("--mlperf-conf", type=str, default="mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-conf", type=str, default="user.conf", help="user config for user LoadGen settings such as target QPS")
    parser.add_argument("--total-sample-count", type=int, default=24576, help="Number of samples to use in benchmark.") # TODO: This interpretation of 'total-sample-count' is a little misleading. Fix it
    parser.add_argument("--output-log-dir", type=str, default="build/logs", help="Where logs are saved")
    parser.add_argument("--enable-log-trace", action="store_true", help="Enable log tracing. This file can become quite large")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers to process queries")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quantized", action='store_true', help="If using a AWQ quantized model")
    parser.add_argument("--warmup", action='store_true', help="Do warmup")

    args = parser.parse_args()

    print("### Benchmark parameters")
    params_list = [action.dest for action in parser._actions]
    for param in params_list:
        try:
            value = getattr(args, param)
            print(f"{param} = {value}")
        except AttributeError:
            continue
    # Define grep patterns for environment and package info
    env_patterns = ['VLLM', 'PT_', 'HL_', 'MAX_']
    pip_patterns = ['vllm', 'neural', 'loadgen', 'intel', 'triton', 'ccl']
    
    # Get environment variables
    for pattern in env_patterns:
        try:
            # Using standard system command - trusted input
            result = subprocess.run(['env'], capture_output=True, text=True, check=True)  # nosec B607 B603
            filtered_output = '\n'.join([line for line in result.stdout.split('\n') if pattern in line])
            if filtered_output:
                print(filtered_output)
        except Exception:
            # Environment variable lookup failed, continue
            continue
    
    # Get pip package info
    for pattern in pip_patterns:
        try:
            # Using standard system command - trusted input
            result = subprocess.run(['pip', 'list'], capture_output=True, text=True, check=True)  # nosec B607 B603
            filtered_output = '\n'.join([line for line in result.stdout.split('\n') if pattern in line])
            if filtered_output:
                print(filtered_output)
        except Exception:
            # Package listing failed, continue
            continue

    with open('user.conf', 'r') as file:
        for line in file:
            if line.strip():
                print(line, end='')
    print("########################")
    return args

scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "singlestream": lg.TestScenario.SingleStream
    }

def main():
    args = get_args()

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(args.workload_name + "-MAIN")

    sut_map = {
        "offline": SUT,
        "server": SUTServer,
        "singlestream": SUTServer
    }

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    # Need to update the conf
    settings.FromConfig(args.user_conf, args.workload_name, args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
        log.warning("Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet")
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    partial_output_dir_list = []
    cur_output_log_dir = args.output_log_dir

    os.makedirs(cur_output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = cur_output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    sut_cls = sut_map[args.scenario.lower()]

    sut = sut_cls(
        model_path=args.model_path,
        workload_name=args.workload_name,
        lg_settings=settings,
        dtype=args.dtype,
        dataset_path=args.dataset_path,
        total_sample_count=args.total_sample_count,
        device=args.device,
        workers=args.num_workers,
        tp=args.tensor_parallel,
        pp=args.pipeline_parallel,
        batch_size=args.batch_size,
        quantized=args.quantized,
        warmup=args.warmup,
        partial_output_dir_list=partial_output_dir_list
    )
    # import pdb;pdb.set_trace()

    # Start sut before loadgen starts
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("Starting Benchmark run")
    t_start = time.time()
    lg.StartTestWithLogSettings(lgSUT, sut.qsl, settings, log_settings, args.audit_conf)
    t_end = time.time()
    log.info("Test took {:.1f} sec".format(t_end - t_start))

    # Stop sut after completion
    sut.stop()

    log.info("Run Completed!")

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()
