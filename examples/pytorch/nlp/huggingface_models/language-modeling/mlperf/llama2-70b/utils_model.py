import os
from vllm import SamplingParams
import numpy as np
import time
import types

# Disable prints if progress bar is active
def void(*args, **kwargs):
    pass


PBAR = int(os.environ.get("PBAR", "1"))
# Update frequency of the progress bar
PBAR_FREQ = int(os.environ.get("PBAR_FREQ", "100"))
XPU_COUNT = int(os.environ.get("XPU_COUNT", "0"))
WORKLOAD_NAME= os.environ.get("WORKLOAD_NAME", "llama2-70b")
GPU_MEMORY_UTILIZATION= float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.95"))
if PBAR==1:
    print = void

#***********************Values to be imported into SUT.py

SAMPLING_PARAMS = SamplingParams(
    max_tokens=1024 if WORKLOAD_NAME=="llama2-70b" else 128,
    min_tokens=1,
    temperature=0.0,
    detokenize=False
)

ADDITIONAL_MODEL_KWARGS = {
    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION
}

if XPU_COUNT>0:
    ########################XPU-specific
    #ADDITIONAL_MODEL_KWARGS["enforce_eager"] = True
    pass
else:
    ########################CPU-specific

    from numa import schedule, memory
    import os
    import math
    import subprocess  # nosec B404
    cores_per_inst = int(os.environ.get("CORES_PER_INST", "1"))
    num_numa_nodes = int(os.environ.get("NUM_NUMA_NODES", "1"))
    nodes_per_inst = int(os.environ["NUM_NUMA_NODES"])/int(os.environ["NUM_INSTS"])
    insts_per_node = int(os.environ["INSTS_PER_NODE"])
    # Only start workers in allowed NUMA nodes. Useful when node sharing
    ALLOWED_NODES = os.environ.get("ALLOWED_NODES", "all")

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

    def SUT_OVERRIDE(self):
        node_start_cores = get_start_cores()
        core_lists = []
        if insts_per_node>0:
            for i in range(num_numa_nodes):
                for j in range(insts_per_node):
                    core_lists.append(tuple(range(node_start_cores[i]+j*cores_per_inst, node_start_cores[i]+(j+1)*cores_per_inst)))

        node_list = list(range(len(node_start_cores)))
        allowed_nodes = [str(node) for node in node_list]
        if ALLOWED_NODES!="all":
            allowed_nodes = ALLOWED_NODES.split(",")
        self.individual_worker_kwargs["core_list"] = core_lists

        instance_node_list = []
        for j in range(self.num_workers):
            cur_node = math.floor(j*nodes_per_inst)
            instance_node_list.append(tuple([cur_node]))
        self.individual_worker_kwargs["node_list"] = instance_node_list
        

    def INSTANCE_OVERRIDE(self):
        print("self.node_list", self.node_list)
        os.environ["VLLM_CPU_OMP_THREADS_BIND"]=f"{self.core_list[0]}-{self.core_list[-1]}"
        memory.set_membind_nodes(*self.node_list)
