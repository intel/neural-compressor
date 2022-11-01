import argparse
import os
import re

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--framework", type=str, required=True)
parser.add_argument("--fwk_ver", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--logs_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str, default=".")
parser.add_argument("--build_id", type=str, default="3117")
args = parser.parse_args()
print('===== collecting log model =======')
print('build_id: '+args.build_id)
OS='linux'
PLATFORM='icx'
URL ='https://dev.azure.com/lpot-inc/neural-compressor/_build/results?buildId='+args.build_id+'&view=artifacts&pathAsName=false&type=publishedArtifacts'

print(args)


def get_model_tuning_results():
    tuning_result_dict = {}

    if os.path.exists(tuning_log):
        print('tuning log found')
        tmp = {'fp32_acc': 0, 'int8_acc': 0, 'tuning_trials': 0}
        with open(tuning_log, "r") as f:
            for line in f:
                parse_tuning_line(line, tmp)
        print(tmp)
        # set model status failed
        if tmp['fp32_acc'] == 0 or tmp['int8_acc'] == 0:
            os.system('echo "##vso[task.setvariable variable=' + args.framework + '_' + args.model + '_failed]true"')

        tuning_result_dict = {
            "OS": OS,
            "Platform": PLATFORM,
            "Framework": args.framework,
            "Version": args.fwk_ver,
            "Model": args.model,
            "Strategy": tmp['strategy'],
            "Tune_time": tmp['tune_time'],
        }
        benchmark_accuracy_result_dict = {
            'int8': {
                "OS": OS,
                "Platform": PLATFORM,
                "Framework": args.framework,
                "Version": args.fwk_ver,
                "Model": args.model,
                "Mode": "Inference",
                "Type": "Accuracy",
                "BS": 1,
                "Value": tmp['int8_acc'],
                "Url": URL,
            },
            'fp32': {
                "OS": OS,
                "Platform": PLATFORM,
                "Framework": args.framework,
                "Version": args.fwk_ver,
                "Model": args.model,
                "Mode": "Inference",
                "Type": "Accuracy",
                "BS": 1,
                "Value": tmp['fp32_acc'],
                "Url": URL,
            }
        }

    return tuning_result_dict, benchmark_accuracy_result_dict


def get_model_benchmark_results():
    benchmark_performance_result_dict = {'int8': {}, 'fp32': {}}
    for precision in ['int8', 'fp32']:
        throughput = 0.0
        bs = 1
        for root, dirs, files in os.walk(args.logs_dir):
            for name in files:
                file_name = os.path.join(root, name)
                print(file_name)
                if 'performance-' + precision in name:
                    for line in open(file_name, "r"):
                        result= parse_perf_line(line)
                        if result.get("throughput"):
                            throughput += result.get("throughput")
                        if result.get("batch_size"):
                            bs = result.get("batch_size")

        # set model status failed
        if throughput==0.0:
            os.system('echo "##vso[task.setvariable variable='+args.framework+'_'+args.model+'_failed]true"')
        benchmark_performance_result_dict[precision] = {
            "OS": OS,
            "Platform": PLATFORM,
            "Framework": args.framework,
            "Version": args.fwk_ver,
            "Model": args.model,
            "Mode": "Inference",
            "Type": "Performance",
            "BS": 1,
            "Value":throughput,
            "Url":URL,
        }

    return benchmark_performance_result_dict


def main():
    results = []
    tuning_infos = []
    print("tuning log dir is {}".format(tuning_log))
    # get model tuning results
    if os.path.exists(tuning_log):
        print('tuning log found')
        tmp = {'fp32_acc': 0, 'int8_acc': 0, 'tuning_trials': 0}
        with open(tuning_log, "r") as f:
            for line in f:
                parse_tuning_line(line, tmp)
        print(tmp)
        # set model status failed
        if tmp['fp32_acc']==0 or tmp['int8_acc']==0:
            os.system('echo "##vso[task.setvariable variable='+args.framework+'_'+args.model+'_failed]true"')
        results.append('{};{};{};{};FP32;{};Inference;Accuracy;1;{};{}\n'.format(OS, PLATFORM, args.framework, args.fwk_ver, args.model, tmp['fp32_acc'], URL))
        results.append('{};{};{};{};INT8;{};Inference;Accuracy;1;{};{}\n'.format(OS, PLATFORM, args.framework,  args.fwk_ver, args.model, tmp['int8_acc'], URL))
        tuning_infos.append(';'.join([OS, PLATFORM, args.framework,  args.fwk_ver, args.model, tmp['strategy'], str(tmp['tune_time']), str(tmp['tuning_trials']), URL, f"{round(tmp['max_mem_size'] / tmp['total_mem_size'] * 100, 4)}%"])+'\n')
    # get model benchmark results
    for precision in ['int8', 'fp32']:
        throughput = 0.0
        bs = 1
        for root, dirs, files in os.walk(args.logs_dir):
            for name in files:
                file_name = os.path.join(root, name)
                print(file_name)
                if 'performance-'+precision in name:
                    for line in open(file_name, "r"):
                        result= parse_perf_line(line)
                        if result.get("throughput"):
                            throughput += result.get("throughput")
                        if result.get("batch_size"):
                            bs = result.get("batch_size")
        # set model status failed
        if throughput==0.0:
            os.system('echo "##vso[task.setvariable variable='+args.framework+'_'+args.model+'_failed]true"')
        results.append('{};{};{};{};{};{};Inference;Performance;{};{};{}\n'.format(OS, PLATFORM, args.framework, args.fwk_ver, precision.upper(), args.model, bs, throughput, URL))
    # write model logs
    f = open(args.output_dir+'/'+args.framework+'_'+args.model+'_summary.log', "a")
    f.writelines("OS;Platform;Framework;Version;Precision;Model;Mode;Type;BS;Value;Url\n")
    for result in results:
        f.writelines(str(result))
    f2 = open(args.output_dir + '/'+args.framework+'_'+args.model+'_tuning_info.log', "a")
    f2.writelines("OS;Platform;Framework;Version;Model;Strategy;Tune_time\n")
    for tuning_info in tuning_infos:
        f2.writelines(str(tuning_info))


def parse_tuning_line(line, tmp):
    tuning_strategy = re.search(r"Tuning strategy:\s+([A-Za-z]+)", line)
    if tuning_strategy and tuning_strategy.group(1):
        tmp['strategy'] = tuning_strategy.group(1)

    baseline_acc = re.search(r"FP32 baseline is:\s+\[Accuracy:\s(\d+(\.\d+)?), Duration \(seconds\):\s*(\d+(\.\d+)?)\]",
                             line)
    if baseline_acc and baseline_acc.group(1):
        tmp['fp32_acc'] = float(baseline_acc.group(1))

    tuned_acc = re.search(r"Best tune result is:\s+\[Accuracy:\s(\d+(\.\d+)?), Duration \(seconds\):\s(\d+(\.\d+)?)\]", line)
    if tuned_acc and tuned_acc.group(1):
        tmp['int8_acc'] = float(tuned_acc.group(1))

    tune_trial = re.search(r"Tune \d*\s*result is:", line)
    if tune_trial:
        tmp['tuning_trials'] += 1

    tune_time = re.search(r"Tuning time spend:\s+(\d+(\.\d+)?)s", line)
    if tune_time and tune_time.group(1):
        tmp['tune_time'] = int(tune_time.group(1))

    fp32_model_size = re.search(r"The input model size is:\s+(\d+(\.\d+)?)", line)
    if fp32_model_size and fp32_model_size.group(1):
        tmp['fp32_model_size'] = int(fp32_model_size.group(1))

    int8_model_size = re.search(r"The output model size is:\s+(\d+(\.\d+)?)", line)
    if int8_model_size and int8_model_size.group(1):
        tmp['int8_model_size'] = int(int8_model_size.group(1))

    total_mem_size = re.search(r"Total resident size\D*([0-9]+)", line)
    if total_mem_size and total_mem_size.group(1):
        tmp['total_mem_size'] = float(total_mem_size.group(1))

    max_mem_size = re.search(r"Maximum resident set size\D*([0-9]+)", line)
    if max_mem_size and max_mem_size.group(1):
        tmp['max_mem_size'] = float(max_mem_size.group(1))


def parse_perf_line(line) -> float:
    perf_data = {}

    throughput = re.search(r"Throughput:\s+(\d+(\.\d+)?)", line)
    if throughput and throughput.group(1):
        perf_data.update({"throughput": float(throughput.group(1))})

    batch_size = re.search(r"Batch size = ([0-9]+)", line)
    if batch_size and batch_size.group(1):
        perf_data.update({"batch_size": int(batch_size.group(1))})

    return perf_data


if __name__ == '__main__':
    tuning_log = os.path.join(args.logs_dir, f"{args.framework}-{args.model}-tune.log")
    main()
