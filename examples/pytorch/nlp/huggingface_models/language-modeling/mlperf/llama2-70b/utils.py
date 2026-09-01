import subprocess  # nosec B404
import json
import os
import io

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_start_cores():
    # Use safer subprocess call without shell=True - single process approach
    try:
        # Using standard system command - trusted input
        result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)  # nosec B607 B603
        start_cores = []
        
        # Process each line to find NUMA node CPU information
        for line in result.stdout.split('\n'):
            if 'NUMA node' in line and 'CPU' in line:
                # Extract the CPU range (e.g., "0-15" from "NUMA node0 CPU(s): 0-15")
                parts = line.split()
                if len(parts) >= 3:
                    cpu_range = parts[-1]  # Last part should be the CPU range
                    # Get the start core (before the '-')
                    if '-' in cpu_range:
                        start_core = int(cpu_range.split('-')[0])
                        start_cores.append(start_core)
        
        return start_cores if start_cores else [0]
    except Exception:
        # Fallback to basic approach
        return [0]

def get_total_runtime(log_detail_file):
    import re
    from datetime import datetime

    with open(log_detail_file) as fid:
        text = fid.read()

    bpat = '"power_begin", "value": "(.*?)",'
    epat = '"power_end", "value": "(.*?)",'

    pbegin = re.search(bpat, text).group(1)
    pend = re.search(epat, text).group(1)

    print(f"Test started at {pbegin}")
    print(f"Test ended at {pend}")

    start_time = datetime.strptime(pbegin, "%m-%d-%Y %H:%M:%S.%f")
    end_time = datetime.strptime(pend, "%m-%d-%Y %H:%M:%S.%f")
    time_difference = end_time - start_time
    total_time_in_seconds = time_difference.total_seconds()
    print(f"Total time taken is {total_time_in_seconds}")
    return total_time_in_seconds

def estimate_performance(log_detail_file, acc_result_dict):
    total_time_in_seconds = get_total_runtime(log_detail_file)
    tokens_per_sample = acc_result_dict["tokens_per_sample"]
    num_samples = acc_result_dict["gen_num"]
    samples_per_second = round(num_samples / total_time_in_seconds, 3)
    tokens_per_second = round(tokens_per_sample * samples_per_second, 2)
    return tokens_per_second