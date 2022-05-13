"""This script is for NLP Executor benchmaking."""
import os
import subprocess
import re
import argparse
import shutil

def utils_bak(path):
    """backup utils.py."""
    src = path + "/utils.py"
    src_bak = path + "/utils.py.bak"
    shutil.copyfile(src, src_bak)

def utils_copy_after_benchmark(path):
    """copy backup file to original."""
    src = path + "/utils.py.bak"
    src_bak = path + "/utils.py"
    shutil.copyfile(src, src_bak)



def modify_sequence(path, seq_len, dataset_reorder):
    """change sequence len and dataset order."""
    utils_bak(path)
    src = path + "/utils.py"
    src_tmp = path + "/utils_tmp.py"
    with open(src, "r") as src_fp:
        with open(src_tmp, "w") as dst_fp:
            for line in src_fp.readlines():
                line_replace = line
                if line.find("np.array(segment_ids_data)") >= 0 and dataset_reorder == 1:
                    line_replace = line.replace("segment_ids_data", "input_mask_data")
                elif line.find("np.array(input_mask_data)") >= 0 and dataset_reorder == 1:
                    line_replace = line.replace("input_mask_data", "segment_ids_data")
                elif line.find("max_length=128") >= 0 and seq_len > 0:
                    line_replace = line.replace("max_length=128", "max_length={}".format(seq_len))
                elif line.find("F_max_seq_length = 128") >= 0 and seq_len > 0:
                    line_replace = line.replace("F_max_seq_length = 128", "F_max_seq_length = {}" \
                                  .format(seq_len))
                elif line.find("F_max_seq_length = 384") >= 0 and seq_len > 0:
                    line_replace = line.replace("F_max_seq_length = 384", "F_max_seq_length = {}" \
                                  .format(seq_len))
                dst_fp.write(line_replace)

    dst_fp.close()
    src_fp.close()
    shutil.copyfile(src_tmp, src)

def modify_yaml(
        path,
        framework,
        instance,
        cores,
        warmup,
        iteration,
        label_file,
        vocab_file):
    """we copy bert.yaml and change attribute."""
    with open(path + "/bert_static.yaml", "r") as src_fp:
        with open(path + "/bert_tmp.yaml", "w") as dst_fp:
            for line in src_fp.readlines():
                if line.find("num_of_instance") >= 0:
                    dst_fp.write(
                        "      num_of_instance: {}\n".format(instance))
                elif line.find("cores_per_instance") >= 0:
                    dst_fp.write(
                        "      cores_per_instance: {}\n".format(cores))
                elif line.find("warmup") >= 0:
                    dst_fp.write("    warmup: {}\n".format(warmup))
                elif line.find("iteration") >= 0:
                    dst_fp.write("    iteration: {}\n".format(iteration))
                elif line.find("label_file") >= 0:
                    dst_fp.write(
                        "          label_file: {}\n".format(label_file))
                elif line.find("vocab_file") >= 0:
                    dst_fp.write(
                        "          vocab_file: {}\n".format(vocab_file))
                elif line.find("framework") >= 0:
                    dst_fp.write(
                        "  framework: {}\n".format(framework))
                else:
                    dst_fp.write(line)

    dst_fp.close()
    src_fp.close()


def numbers_to_strings(argument):
    """allocator mode num to str."""
    switcher = {
        0: "direct",
        1: "cycle",
        2: "unified",
        3: "je_direct",
        4: "je_cycle",
        5: "je_unified",
    }
    return switcher.get(argument, "cycle")


def concat_allocator_cmd(allocator, cmd):
    """add env variable for different allocator modes."""
    new_cmd = cmd
    if allocator == "direct":
        new_cmd = "DIRECT_BUFFER=1 " + cmd
    elif allocator == "unified":
        new_cmd = "UNIFIED_BUFFER=1 " + cmd
    elif allocator == "je_direct":
        new_cmd = "JEMALLOC=1 DIRECT_BUFFER=1 " + cmd
    elif allocator == "je_cycle":
        new_cmd = "JEMALLOC=1 " + cmd
    elif allocator == "je_unified":
        new_cmd = "JEMALLOC=1 UNIFIED_BUFFER=1 " + cmd
    return new_cmd


def grab_log(is_performance, path, instance, cores, log_fp):
    """extract performance from logs."""
    latency = float(0)
    throughput = float(0)
    if is_performance:
        i = 0
        throughput_str = ""
        latency_str = ""
        while i < instance:
            log_path = "{}/{}_{}_{}.log".format(path, instance, cores, i)
            i += 1
            try:
                with open(log_path, 'r') as src_fp:
                    for line in src_fp.readlines():
                        if line.find("Throughput") >= 0:
                            throughput_str = line
                        elif line.find("Latency") >= 0:
                            latency_str = line
                float_re = re.compile(r'\d+\.\d+')
                floats = [float(i) for i in float_re.findall(throughput_str)]
                floats_latency = [float(i) for i in float_re.findall(latency_str)]

                throughput += floats[0]
                latency += floats_latency[0]
            except OSError as ex:
                print(ex)
        latency = latency / instance
    else:
        print("========please check acc with screen messages=============")
    try:
        if is_performance:
            log_fp.write("Troughput: {} images/sec\n".format(throughput))
            log_fp.write("Latency: {} ms\n".format(latency))
        log_fp.write("--------------------------------------\n")
    except OSError as ex:
        print(ex)


def execute_and_grab(is_performance, model_file, model_path, batch, allocator):
    """execute the run_engine.py."""
    cmd = ""
    if is_performance:
        cmd = "GLOG_minloglevel=2 python run_engine.py --input_model={}/{}" \
              " --config={}/bert_tmp.yaml --benchmark --mode=performance --batch_size={}" \
              .format(model_path, model_file, model_path, batch)
    else:

        cmd = "GLOG_minloglevel=2 ONEDNN_VERBOSE=1 python run_engine.py --input_model={}/{}" \
              " --config={}/bert_tmp.yaml --benchmark --mode=accuracy --batch_size={}" \
              .format(model_path, model_file, model_path, batch)

    cmd = concat_allocator_cmd(allocator, cmd)

    try:
        with open("tmp.sh", "w") as file_p:
            file_p.write("cd {}\n".format(model_path))
            file_p.write(cmd)
        pro = subprocess.Popen(
            "bash tmp.sh",
            shell=True)
        pro.wait()
        file_p.close()

    except OSError as ex:
        print(ex)


def test_all(
        is_performance=True,
        support_models=None,
        batch=None,
        instance_cores=None,
        allocator_mode=None,
        sequence=128,
        warmup=5,
        iterations=10,
        is_int8=False,
        label_file="",
        vocab_file="",
        output_file=""):
    """find model and do benchmark."""
    print("search start")
    print("performance mode is {}".format(is_performance))
    print("search for int8 model {}".format(is_int8))
    benchmark_models = []
    benchmark_path = []
    if allocator_mode is None:
        allocator_mode = [1]
    if batch is None:
        batch = [1]
    if instance_cores is None:
        instance_cores = [1, 28]
    if support_models is None:
        support_models = ["bert_mini_mrpc"]


    for task in os.listdir(os.getcwd()):
        task_path = os.path.join(os.getcwd(), task)
        if os.path.isdir(task_path):
            for model in os.listdir(task_path):
                model_path = os.path.join(task_path, model)
                model_file = ""
                if not is_int8:
                    for file in os.listdir(model_path):
                        if file.endswith("onnx") or file.endswith("pb"):
                            model_file = file
                            benchmark_models = (*benchmark_models, model_file)
                            benchmark_path = (*benchmark_path, model_path)
                            print(model_file, " fp32 exist!!")
                            break
                else:
                    int8_model_path = os.path.join(model_path, "ir")
                    if os.path.exists(int8_model_path):
                        for file in os.listdir(int8_model_path):
                            if file.endswith("model.bin"):
                                model_file = file
                                benchmark_models = (*benchmark_models, "ir")
                                benchmark_path = (*benchmark_path, model_path)
                                break

                if model_file == "":
                    print("{}_{} not find model file!!!".format(model, task))
                else:
                    if "{}_{}".format(model, task) not in support_models:
                        last_element_index = len(benchmark_models)-1
                        benchmark_models = benchmark_models[: last_element_index]
                        last_element_index = len(benchmark_path)-1
                        benchmark_path = benchmark_path[:last_element_index]
                        continue

    print("search end")
    if not benchmark_models:
        print("============no .onnx or .pb for fp32, no ir folder for int8==============\n")
        return 0
    allocator = []

    instance = []
    cores = []
    instance, cores = zip(*instance_cores)
    dataset_reorder = 0
    framework = "engine"
    # this reorder and framework change only support for onnx model
    # tf model you need to use fp32 ir, so you should remvove snippet here
    # when model is tf, but we will not add arg to control, only bert base
    # and bert large use tf now
    if not is_int8 and not is_performance:
        dataset_reorder = 1
        framework = "onnxrt_integerops"
    print("============benchmark start==================")

    try:
        with open(output_file, "w") as file_p:
            for enabled_model_id, enabled_model_val in enumerate(
                    benchmark_models):
                print(enabled_model_val, "exist!!")
                bench_model_path = benchmark_path[enabled_model_id]
                bench_model_file = enabled_model_val
                modify_sequence(bench_model_path, sequence, dataset_reorder)
                for alloc_mode_id, alloc_mode_val in enumerate(allocator_mode):
                    allocator = numbers_to_strings(alloc_mode_val)
                    file_p.write(
                        "Model_{}_Allocator_{}-{}\n".format(
                            bench_model_file, alloc_mode_id, allocator))

                    for ins_idx, ins_val in enumerate(instance):
                        modify_yaml(
                            bench_model_path,
                            framework,
                            ins_val,
                            cores[ins_idx],
                            warmup,
                            iterations,
                            label_file,
                            vocab_file)
                        for ba_s in batch:
                            file_p.write("Path {}\n".format(bench_model_path))
                            file_p.write(
                                "Instance_{}_Cores_{}_Batch_{}\n".format(
                                    ins_val, cores[ins_idx], ba_s))
                            execute_and_grab(
                                is_performance, bench_model_file, bench_model_path, ba_s, allocator)
                            grab_log(
                                is_performance,
                                bench_model_path,
                                ins_val,
                                cores[ins_idx],
                                file_p)

            file_p.close()
            utils_copy_after_benchmark(bench_model_path)

    except OSError as ex:
        print(ex)

        return 1


def main():
    """parsing user arg."""
    is_performance = True
    is_int8 = False
    output_file = "benchmark.txt"
    sequence_len = 0
    iterations = 10
    warmup = 5
    batch_size = [16, 32]
    instance_cores = [[4, 7]]
    allocator_mode = [1]
    model_list = [
        "bert_mini_mrpc",
        "distilroberta_base_wnli",
        "distilbert_base_uncased_sst2",
        "roberta_base_mrpc",
        "bert_base_nli_mean_tokens_stsb",
        "bert_base_sparse_mrpc",
        "distilbert_base_uncased_mrpc",
        "bert_mini_sst2",
        "bert_base_mrpc",
        "minilm_l6_h384_uncased_sst2",
        "distilbert_base_uncased_emotion",
        "paraphrase_xlm_r_multilingual_v1_stsb",
        "finbert_financial_phrasebank",
        "bert_large_squad"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        '--batch',
        '-b',
        help="batch size 1,2,3: --batch 1 2 3 ",
        type=int,
        nargs='+',
        dest='batch')
    parser.add_argument(
        '--allocator',
        '-a',
        help="allocator 1,5: --allocator 1 5" +
        "(0:direct 1:cycle,this one is default 2:unified 3:jemalloc+direct 4:jemalloc+cycle " +
        " 5:jemalloc+unified)",
        type=int,
        nargs='+',
        dest='allocator')
    parser.add_argument(
        '--instance_cores',
        '-i',
        help="--instance_cores 4x7 1x28 , it means 4instance 7 cores and 1 instance 28 cores",
        type=str,
        nargs='+',
        dest='i_c')
    parser.add_argument(
        '--model',
        '-m',
        help="--model bert_mini_mrpc,distilbert_base_uncased_sst2,roberta_base_mrpc,"+
        "bert_base_nli_mean_tokens_stsb,bert_base_sparse_mrpc,distilbert_base_uncased_mrpc,"+
        "bert_mini_sst2,bert_base_mrpc,minilm_l6_h384_uncased_sst2,"+
        "distilbert_base_uncased_emotion,paraphrase_xlm_r_multilingual_v1_stsb,"+
        "finbert_financial_phrasebank,bert_large_squad",
        type=str,
        nargs='+',
        dest='model_name')

    parser.add_argument(
        '--warmup',
        '-w',
        help="warmup 10 times: --warmup 10 ",
        type=int,
        dest='warmup')
    parser.add_argument(
        '--iterations',
        '-e',
        help="execute 50 times: --iterations 50 ",
        type=int,
        dest='iterations')
    parser.add_argument(
        '--seq_len',
        '-s',
        help="you can only input one int",
        type=int,
        dest='seq_len')
    parser.add_argument('--int8', type=int, dest='int8')
    parser.add_argument(
        '--is_performance',
        '-p',
        help="1: performance mode, 0: accuracy mode",
        type=int,
        dest='is_performance')
    parser.add_argument(
        '--label_file',
        '-l',
        help="--only bert large need this path",
        type=str,
        dest='label_file')
    parser.add_argument(
        '--vocab_file',
        '-v',
        help="--only bert large need this path",
        type=str,
        dest='vocab_file')
    parser.add_argument(
        '--output_file',
        '-o',
        help="outputfile: --output_file benchmark.txt",
        type=str,
        dest='output_file')

    args = parser.parse_args()
    if args.batch:
        batch_size = []
        for batch_val in args.batch:
            batch_size.append(batch_val)

    if args.allocator:
        allocator_mode = []
        for allocator_val in args.allocator:
            allocator_mode.append(allocator_val)

    if args.i_c:
        instance_cores = []
        ic_val = []
        for ic_val in args.i_c:
            ic_value = ic_val.split("x")
            tmp_list = [int(ic_value[0]), int(ic_value[1])]
            instance_cores.append(tmp_list)

    if args.model_name:
        model_list = []
        for model_val in args.model_name:
            model_list.append(model_val)

    if args.warmup:
        warmup = args.warmup
    if args.iterations:
        iterations = args.iterations

    if args.int8 == 1:
        is_int8 = True

    if args.is_performance == 0:
        is_performance = False

    label_file = ""
    if args.label_file:
        label_file = args.label_file
    vocab_file = ""
    if args.vocab_file:
        vocab_file = args.vocab_file

    if args.output_file:
        output_file = args.output_file

    if args.seq_len:
        sequence_len = args.seq_len


    test_all(
        is_performance,
        model_list,
        batch_size,
        instance_cores,
        allocator_mode,
        sequence_len,
        warmup,
        iterations,
        is_int8,
        label_file,
        vocab_file,
        output_file)


if __name__ == "__main__":
    main()
