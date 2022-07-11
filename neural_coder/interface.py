# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import subprocess
import logging
import time

from . import globals

if not os.path.exists("neural_coder_workspace"):
    os.makedirs("neural_coder_workspace")


def enable(
    code,
    features,
    target_batch_size=1,  # effective for feature "pytorch_change_batch_size"
    num_benchmark_iteration=30,  # effective for feature "pytorch_benchmark"
    generate_patch=True,
    overwrite=False,
    # TO-ADD: return a folder with user-input code and artificial pip folders
    brutal_mode=False,
    save_patch_path="",
    patch_suffix=".diff",
    remove_copy=True,
    consider_imports=True,
    patch_imports=False,
    logging_level="info",
    run_bench=False,
    entry_code="",
    entry_code_args="",
    mode="throughput",
    cpu_set_env=True,
    ncore_per_instance=-1,  # only for "self_defined" mode
    ninstances=-1,  # only for "self_defined" mode
    bench_batch_size=-1,  # only for "self_defined" mode
):

    # Preparation

    # set up workspace
    ws_path = "neural_coder_workspace/" + \
        "enable" + str(int(time.time())) + "/"
    os.makedirs(ws_path)

    # user parameters
    globals.consider_imports = consider_imports
    logging_var = "logging." + logging_level.upper()
    globals.logging_level = eval(logging_var)

    # set up logging
    logger = logging.getLogger(ws_path)
    logger.setLevel(globals.logging_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(ws_path+'enable.log')
    fh.setLevel(globals.logging_level)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(globals.logging_level)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # print key inputs
    logger.info(f"Enabling started ...")
    logger.info(f"code: {code}")
    logger.info(f"features: {features}")

    # feature list for reference
    '''
    feature_list = ["pytorch_jit_script",
                    "pytorch_jit_script_ofi",
                    "pytorch_inc_dynamic_quant",
                    "pytorch_inc_static_quant",
                    "pytorch_ipex_fp32",
                    "pytorch_ipex_bf16",
                    "pytorch_ipex_int8_static_quant",
                    "pytorch_ipex_int8_dynamic_quant",
                    "pytorch_channels_last",
                    "pytorch_mixed_precision_cpu",
                    "pytorch_mixed_precision_cuda",
                    "pytorch_torchdynamo_jit_script",
                    "pytorch_torchdynamo_jit_script_ofi",
                    "pytorch_torchdynamo_jit_trace",
                    "pytorch_torchdynamo_jit_trace_ofi",
                    "pytorch_benchmark",
                    "pytorch_change_batch_size",
                    "pytorch_cuda_to_cpu",
                    "pytorch_lightning_bf16_cpu",
                    "tensorflow_amp",
                    "keras_amp",]
    '''

    # Benchmark
    if run_bench:
        # add "pytorch_change_batch_size" to features
        from .utils.cpu_info import get_num_cpu_cores
        ncores = get_num_cpu_cores()
        if mode == "throughput":
            target_batch_size = 2 * ncores
        elif mode == "multi_instance":
            target_batch_size = 1
        elif mode == "latency":
            target_batch_size = 1
        elif mode == "self_defined":
            target_batch_size = bench_batch_size

        if "pytorch_change_batch_size" not in features:
            features.append("pytorch_change_batch_size")

        # add "pytorch_benchmark" to features
        if "pytorch_benchmark" not in features:
            features.append("pytorch_benchmark")

        logger.info(
            f"Will perform benchmark on [{mode}] mode with batch size [{target_batch_size}] ...")

    # Rearrange Feature Order (due to certain regulations: e.g. channels_last is before IPEX, and IPEX is before JIT)
    from .utils.common import move_element_to_front
    for feature in ["pytorch_benchmark",
                    "pytorch_channels_last",
                    "pytorch_ipex_fp32",
                    "pytorch_ipex_bf16",
                    "pytorch_ipex_int8_static_quant",
                    "pytorch_ipex_int8_dynamic_quant",
                    "pytorch_jit_script",
                    "pytorch_jit_script_ofi",
                    "pytorch_torchdynamo_jit_script",
                    "pytorch_torchdynamo_jit_script_ofi",
                    "pytorch_torchdynamo_jit_trace",
                    "pytorch_torchdynamo_jit_trace_ofi",
                    "pytorch_inc_static_quant",
                    "pytorch_inc_dynamic_quant",
                    "pytorch_mixed_precision_cpu",
                    "pytorch_mixed_precision_cuda", ]:
        features = move_element_to_front(features, feature)

    # Enabling

    transformed_list_code_path = []

    for feature in features:

        # reset globals
        globals.reset_globals()

        from .utils import handle_user_input
        globals.list_code_path, num_user_code_path = handle_user_input.get_all_code_path(
            code)
        if len(transformed_list_code_path) > 0:
            globals.list_code_path = transformed_list_code_path

        # common for all features (transformations),
        list_transformed_code = []
        # in this list, each item stores the transformed code of the corresponding original code
        # by the order in code_path

        # global behaviors
        logger.info(
            f"Performing code transformation for feature: [{feature}] ...")

        from .graphers.code_line import register_code_line
        from .graphers.model import register_nnModule_class, register_nnModule_instance_definition
        from .graphers.function import register_func_wrap_pair
        from .coders.transform import execute_insert_transformation, execute_indenting_transformation

        register_code_line()
        register_func_wrap_pair()
        register_nnModule_class()
        register_nnModule_instance_definition()

        # AMP
        if "pytorch_mixed_precision_cpu" == feature:
            from .coders.pytorch.amp import PTAMP
            opt = PTAMP("cpu")
            opt.register_transformation()
        elif "pytorch_mixed_precision_cuda" == feature:
            from .coders.pytorch.amp import PTAMP
            opt = PTAMP("cuda")
            opt.register_transformation()

        # TorchDynamo
        if "pytorch_torchdynamo_jit_script" == feature:
            from .coders.pytorch.torchdynamo import TorchDynamo
            opt = TorchDynamo()
            opt.register_transformation()
            from .coders.pytorch.torchdynamo import TorchDynamoJITScript
            opt = TorchDynamoJITScript(
                globals.list_model_def_instance, "plain")
            opt.register_transformation()
        elif "pytorch_torchdynamo_jit_script_ofi" == feature:
            from .coders.pytorch.torchdynamo import TorchDynamo
            opt = TorchDynamo()
            opt.register_transformation()
            from .coders.pytorch.torchdynamo import TorchDynamoJITScript
            opt = TorchDynamoJITScript(globals.list_model_def_instance, "ofi")
            opt.register_transformation()
        elif "pytorch_torchdynamo_jit_trace" == feature:
            from .coders.pytorch.torchdynamo import TorchDynamo
            opt = TorchDynamo()
            opt.register_transformation()
            from .coders.pytorch.torchdynamo import TorchDynamoJITTrace
            opt = TorchDynamoJITTrace(globals.list_model_def_instance, "plain")
            opt.register_transformation()
        elif "pytorch_torchdynamo_jit_trace_ofi" == feature:
            from .coders.pytorch.torchdynamo import TorchDynamo
            opt = TorchDynamo()
            opt.register_transformation()
            from .coders.pytorch.torchdynamo import TorchDynamoJITTrace
            opt = TorchDynamoJITTrace(globals.list_model_def_instance, "ofi")
            opt.register_transformation()

        # Channels Last
        if "pytorch_channels_last" == feature:
            from .coders.pytorch.channels_last import ChannelsLast
            opt = ChannelsLast(globals.list_model_def_instance)
            opt.register_transformation()

        # JIT
        if "pytorch_jit_script" == feature:
            from .coders.pytorch.jit import JITScript
            opt = JITScript(globals.list_model_def_instance, "plain")
            opt.register_transformation()

        elif "pytorch_jit_script_ofi" == feature:
            from .coders.pytorch.jit import JITScript
            opt = JITScript(globals.list_model_def_instance, "ofi")
            opt.register_transformation()

        # IPEX
        if "pytorch_ipex_fp32" == feature:
            from .coders.pytorch.intel_extension_for_pytorch.ipex import IPEX
            opt = IPEX(globals.list_model_def_instance, "fp32")
            opt.register_transformation()

        elif "pytorch_ipex_bf16" == feature:
            from .coders.pytorch.intel_extension_for_pytorch.ipex import IPEX
            opt = IPEX(globals.list_model_def_instance, "bf16")
            opt.register_transformation()
            from .coders.pytorch.amp import PTAMP  # enable amp together
            opt = PTAMP("cpu")
            opt.register_transformation()

        elif "pytorch_ipex_int8_static_quant" == feature:
            from .coders.pytorch.intel_extension_for_pytorch.ipex import IPEX
            opt = IPEX(globals.list_model_def_instance, "int8_static_quant")
            opt.register_transformation()

        elif "pytorch_ipex_int8_dynamic_quant" == feature:
            from .coders.pytorch.intel_extension_for_pytorch.ipex import IPEX
            opt = IPEX(globals.list_model_def_instance, "int8_dynamic_quant")
            opt.register_transformation()

        # INC
        if "pytorch_inc_dynamic_quant" == feature:
            from .coders.pytorch.neural_compressor.dynamic_quant import DynamicQuant
            opt = DynamicQuant(globals.list_model_def_instance)
            opt.register_transformation()

        elif "pytorch_inc_static_quant" == feature:
            from .coders.pytorch.dummy_dataloader import DummyDataLoader  # detect dataloader first
            opt = DummyDataLoader(globals.list_model_def_instance)
            opt.register_transformation()
            from .coders.pytorch.neural_compressor.static_quant import StaticQuant
            opt = StaticQuant(globals.list_model_def_instance)
            opt.register_transformation()

        # Benchmark
        if "pytorch_benchmark" == feature:
            from .coders.pytorch.benchmark import Benchmark
            globals.num_benchmark_iteration = str(num_benchmark_iteration)
            opt = Benchmark()
            opt.register_transformation()

        # transformation execution
        for i in globals.list_code_path:
            list_transformed_code.append(open(i, 'r').read())
        list_transformed_code = execute_indenting_transformation(
            list_transformed_code)
        list_transformed_code = execute_insert_transformation(
            list_transformed_code)

        # other features (which use direct line transform instead of register-and-execute transform,
        # these features will be transformed here)
        for i in range(len(list_transformed_code)):
            # Batch Size
            if "pytorch_change_batch_size" == feature:
                from .coders.pytorch.batch_size import BatchSizeCoder
                globals.target_batch_size = str(target_batch_size)
                list_transformed_code[i] = BatchSizeCoder(
                    list_transformed_code[i]).transform()

            # CUDA to CPU
            if "pytorch_cuda_to_cpu" == feature:
                from .coders.pytorch.cuda_to_cpu import CudaToCpu
                list_transformed_code[i] = CudaToCpu(
                    list_transformed_code[i]).transform()

            # Lightning
            if "pytorch_lightning_bf16_cpu" == feature:
                from .coders.pytorch.lightning import Lightning
                list_transformed_code[i] = Lightning(
                    list_transformed_code[i]).transform()

            # TF & Keras AMP
            if "tensorflow_mixed_precision" == feature:
                from .coders.tensorflow.amp import TensorFlowKerasAMP
                list_transformed_code[i] = TensorFlowKerasAMP(
                    list_transformed_code[i]).transform()

        logger.info(f"Code transformation for feature: [{feature}] finished.")

        for path in globals.list_code_path:
            if path[-14:] == "_nc_enabled.py":
                path_transformed = path
            else:
                path_transformed = path[:-3] + "_nc_enabled.py"
            open(path_transformed, "w").write(
                list_transformed_code[globals.list_code_path.index(path)])
            globals.list_code_path[globals.list_code_path.index(
                path)] = path_transformed
        transformed_list_code_path = globals.list_code_path

    # Output of Enabling

    globals.list_code_path, num_user_code_path = handle_user_input.get_all_code_path(
        code)

    if save_patch_path == "":
        save_patch_path = ws_path

    if generate_patch:
        whole_patch_user_code = ""
        for path in globals.list_code_path[0:num_user_code_path]:
            path_transformed = path[:-3] + "_nc_enabled.py"
            cmd_gen_patch = "diff -up " + path + " " + path_transformed
            sp_gen_patch = subprocess.Popen(
                cmd_gen_patch, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
            sp_gen_patch.wait()
            this_patch, _ = sp_gen_patch.communicate()
            this_patch = str(this_patch)[2:-1]
            whole_patch_user_code += this_patch
        open(save_patch_path + "neural_coder_patch" + patch_suffix, "w").write(
            whole_patch_user_code.replace(r'\n', '\n').replace(r'\t', '\t').replace(r"\'", "\'"))
        abs_patch_path = os.path.abspath(
            save_patch_path + "neural_coder_patch" + patch_suffix)
        logger.info(f"The patch is saved to: [{abs_patch_path}]")

        if overwrite:
            sp_overwrite = subprocess.Popen(
                "patch -d/ -p0 < " + abs_patch_path, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
            sp_overwrite.wait()
            os.remove(abs_patch_path)  # remove patch after overwrite

        if patch_imports:
            whole_patch_import_modules = ""
            for path in globals.list_code_path[num_user_code_path:]:
                path_transformed = path[:-3] + "_nc_enabled.py"
                cmd_gen_patch = "diff -up " + path + " " + path_transformed
                sp_gen_patch = subprocess.Popen(
                    cmd_gen_patch, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
                sp_gen_patch.wait()
                this_patch, _ = sp_gen_patch.communicate()
                this_patch = str(this_patch)[2:-1]
                whole_patch_import_modules += this_patch
            open(save_patch_path + "neural_coder_patch_import_modules" + patch_suffix, "w").write(
                whole_patch_import_modules.replace(r'\n', '\n').replace(r'\t', '\t').replace(r"\'", "\'"))
            abs_patch_path = os.path.abspath(
                save_patch_path + "neural_coder_patch_import_modules" + patch_suffix)
            logger.info(
                f"The patch for imported modules is saved to: [{abs_patch_path}]")

    # remove copy for imports
    if remove_copy:
        for path in globals.list_code_path:
            try:
                path_transformed = path[:-3] + "_nc_enabled.py"
                os.remove(path_transformed)
            except:
                pass

    # Benchmark
    if run_bench:
        bench_performance, bench_mode, bench_ws_path = bench(
            code=code,
            entry_code=entry_code,
            entry_code_args=entry_code_args,
            patch_path=abs_patch_path,
            mode=mode,
            cpu_set_env=cpu_set_env,
            ncore_per_instance=ncore_per_instance,  # only for "self_defined" mode
            ninstances=ninstances,  # only for "self_defined" mode
            bench_batch_size=bench_batch_size,  # only for "self_defined" mode
        )

        return bench_performance, bench_mode, bench_ws_path


'''
bench API works on either "optimized code", or "patch" + "original code"
it does not enable benchmark, or enable change of batch size, all the enabling is done in enable API
which means the "optimized code" should already have "pytorch_benchmark" and "pytorch_change_batch_size" enabled
or the "patch" should already have the code modification for "pytorch_benchmark" and 
"pytorch_change_batch_size" in it
'''


def bench(
    code,
    entry_code="",
    entry_code_args="",
    patch_path="",
    mode="throughput",  # throughput, latency, multi_instance or self_defined
    logging_level="info",
    cpu_set_env=True,
    ncore_per_instance=-1,  # only for "self_defined" mode
    ninstances=-1,  # only for "self_defined" mode
    bench_batch_size=-1,  # only for "self_defined" mode
):

    # set up workspace
    ws_path = "neural_coder_workspace/" + "bench" + str(int(time.time())) + "/"
    os.makedirs(ws_path)

    # set up logging
    logging_var = "logging." + logging_level.upper()
    globals.logging_level = eval(logging_var)

    logger = logging.getLogger(ws_path)
    logger.setLevel(globals.logging_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(ws_path+'bench.log')
    fh.setLevel(globals.logging_level)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(globals.logging_level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # print key inputs
    logger.info(f"Benchmarking started ...")
    logger.info(f"code: {code}")
    logger.info(f"mode: {mode}")

    # entry code
    if entry_code == "":
        # if not specify entry_code, then code has to be a list of one element,
        # or a single string of single path, otherwise quit
        if type(code) == list and len(code) == 1:
            entry_code = code[0]
        elif type(code) == str:
            entry_code = code
        else:
            logger.error(
                f"You have to specify an entry_code of your code: [{code}]")
            quit()

    # patch
    if patch_path != "":
        sp_patch = subprocess.Popen("patch -d/ -p0 < " + patch_path,
                                    env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
        sp_patch.wait()

    # if mode is "self_defined", user must specify ncpi, nins and bs
    if mode == "self_defined":
        if ncore_per_instance == -1 or ninstances == -1 or bench_batch_size == -1:
            logger.error(
                f"You have to specify ncore_per_instance, ninstances and bench_batch_size for \
                    self-defined benchmark mode.")
            quit()

    # numactl
    from . import numa_launcher

    from .utils.cpu_info import get_num_cpu_cores
    ncores = get_num_cpu_cores()

    # numactl setup for different modes
    if mode == "throughput":
        ncore_per_instance = ncores
        ninstances = 1
        bench_batch_size = 2 * ncores
    elif mode == "multi_instance":
        ncore_per_instance = 4
        ninstances = int(ncores / ncore_per_instance)
        bench_batch_size = 1
    elif mode == "latency":
        ncore_per_instance = 1
        ninstances = ncores
        bench_batch_size = 1
    elif mode == "self_defined":
        ncore_per_instance = ncore_per_instance
        ninstances = ninstances
        bench_batch_size = bench_batch_size

    # set cpu env variables
    if cpu_set_env:
        cmd_env = ''
        cmd_env += 'export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so'
        cmd_env += ' && '
        cmd_env += 'export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so'
        cmd_env += ' && '
        cmd_env += 'export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,\
            dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"'
        cmd_env += ' && '
        cmd_env += 'export KMP_AFFINITY="granularity=fine,compact,1,0"'
        cmd_env += ' && '
        cmd_env += 'export KMP_BLOCKTIME=1'
        cmd_env += ' && '
        cmd_env += 'export DNNL_PRIMITIVE_CACHE_CAPACITY=1024'
        cmd_env += ' && '
        cmd_env += 'export KMP_SETTINGS=1'

        sp_set_env = subprocess.Popen(
            cmd_env, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
        sp_set_env.wait()

    # benchmark
    logger.info(f"Start benchmark on the code ...")

    bench_log_path = ws_path + "performance.log"
    os.remove(bench_log_path) if os.path.exists(bench_log_path) else 0

    entry_code_args = [entry_code_args]
    numa_launcher.exec_launcher(
        ncore_per_instance, ninstances, entry_code, entry_code_args, bench_log_path)

    # get performance (throughput and latency)
    bench_log = open(bench_log_path, "r").read().split('\n')
    IPS = 0
    MSPI = 0
    count_MSPI = 0
    P50 = 0
    count_P50 = 0
    P90 = 0
    count_P90 = 0
    P99 = 0
    count_P99 = 0
    for line in bench_log:
        if "Neural_Coder_Bench_IPS" in line:
            try:
                IPS += float(line[line.find(":")+3:])
            except ValueError as ve:
                pass
        if "Neural_Coder_Bench_MSPI" in line:
            try:
                MSPI += float(line[line.find(":")+3:])
                count_MSPI += 1
            except ValueError as ve:
                pass
        if "Neural_Coder_Bench_P50" in line:
            try:
                P50 += float(line[line.find(":")+3:])
                count_P50 += 1
            except ValueError as ve:
                pass
        if "Neural_Coder_Bench_P90" in line:
            try:
                P90 += float(line[line.find(":")+3:])
                count_P90 += 1
            except ValueError as ve:
                pass
        if "Neural_Coder_Bench_P99" in line:
            try:
                P99 += float(line[line.find(":")+3:])
                count_P99 += 1
            except ValueError as ve:
                pass

    FPS = round(IPS * bench_batch_size, 3)
    try:
        MSPI = round(MSPI / count_MSPI, 3)
    except:
        MSPI = 0
    try:
        P50 = round(P50 / count_P50, 3)
    except:
        MSPI = 0
    try:
        P90 = round(P90 / count_P90, 3)
    except:
        MSPI = 0
    try:
        P99 = round(P99 / count_P99, 3)
    except:
        MSPI = 0

    logger.info(f"Collected throughput on the code is: [{FPS}] (fps)")
    logger.info(f"Collected latency on the code is: [{MSPI}] (mspi)")
    logger.info(f"Collected latency_p50 on the code is: [{P50}] (mspi)")
    logger.info(f"Collected latency_p90 on the code is: [{P90}] (mspi)")
    logger.info(f"Collected latency_p99 on the code is: [{P99}] (mspi)")

    # unpatch
    if patch_path != "":
        sp_unpatch = subprocess.Popen(
            "patch -R -d/ -p0 < " + patch_path, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
        sp_unpatch.wait()

    return [FPS, MSPI, P50, P90, P99], mode, os.path.abspath(ws_path)


def superbench(
    code,
    entry_code="",
    entry_code_args="",
    sweep_objective="feature",
    bench_feature=[],
    mode="throughput",
    num_benchmark_iteration=30,
    logging_level="info",
    cpu_conversion=True,
    cpu_set_env=True,
    ncore_per_instance=-1,  # only for "self_defined" mode
    ninstances=-1,  # only for "self_defined" mode
    bench_batch_size=-1,  # only for "self_defined" mode
):

    # set up workspace
    ws_path = "neural_coder_workspace/" + \
        "superbench" + str(int(time.time())) + "/"
    os.makedirs(ws_path)

    # set up logging
    logging_var = "logging." + logging_level.upper()
    globals.logging_level = eval(logging_var)

    logger = logging.getLogger(ws_path)
    logger.setLevel(globals.logging_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(ws_path+'superbench.log')
    fh.setLevel(globals.logging_level)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(globals.logging_level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # print key inputs
    logger.info(f"Superbench started ...")
    logger.info(f"code: {code}")
    logger.info(f"mode: {mode}")
    logger.info(f"sweep_objective: {sweep_objective}")
    logger.info(f"num_benchmark_iteration: {num_benchmark_iteration}")

    # entry code
    if entry_code == "":
        # if not specify entry_code, then code has to be a list of one element,
        # or a single string of single path, otherwise quit
        if type(code) == list and len(code) == 1:
            entry_code = code[0]
        elif type(code) == str:
            entry_code = code
        else:
            logger.error(
                f"You have to specify an entry_code of your code: [{code}]")
            quit()

    if sweep_objective == "feature":
        list_FPS = []
        list_features = []
        list_mode = []
        list_ws_path = []
        result = []

        # features that is a "backend":
        backends = ["",
                    "pytorch_ipex_fp32",
                    "pytorch_ipex_bf16",
                    "pytorch_ipex_int8_static_quant",
                    "pytorch_ipex_int8_dynamic_quant",
                    "pytorch_inc_static_quant",
                    "pytorch_inc_dynamic_quant",
                    ]

        # features that can be standalone (either use alone or use with "backend"):
        standalones_pool = ["pytorch_channels_last",
                            "pytorch_mixed_precision_cpu",
                            "pytorch_jit_script",
                            "pytorch_jit_script_ofi",
                            "pytorch_torchdynamo_jit_script",
                            "pytorch_torchdynamo_jit_script_ofi",
                            "pytorch_torchdynamo_jit_trace",
                            "pytorch_torchdynamo_jit_trace_ofi",
                            ]

        if logging_level == "debug":
            # features that is a "backend":
            backends = ["",
                        "pytorch_ipex_fp32",
                        "pytorch_inc_static_quant",
                        ]

            # features that can be standalone (either use alone or use with "backend"):
            standalones_pool = ["pytorch_channels_last",
                                ]

        standalones = []
        standalones.append("")
        from itertools import combinations
        for num_items in range(len(standalones_pool)):
            list_comb = list(combinations(standalones_pool, num_items + 1))
            for item in list_comb:
                jit_feature_count = 0
                for i in list(item):
                    if "jit" in i:
                        jit_feature_count += 1
                if jit_feature_count <= 1:
                    standalones.append(list(item))  # only appends one JIT item

        for backend in backends:
            for standalone in standalones:
                features = []
                features.append(backend)
                features += standalone

                # exclude conflict features (like jit and jit_ofi)
                if "pytorch_ipex_fp32" in features and "pytorch_mixed_precision_cpu" in features:
                    continue
                if "pytorch_ipex_bf16" in features and "pytorch_mixed_precision_cpu" in features:
                    continue
                if "pytorch_ipex_int8_static_quant" in features and "pytorch_mixed_precision_cpu" in features:
                    continue
                if "pytorch_ipex_int8_dynamic_quant" in features and "pytorch_mixed_precision_cpu" in features:
                    continue
                if "pytorch_inc_static_quant" in features and "pytorch_mixed_precision_cpu" in features:
                    continue
                if "pytorch_inc_dynamic_quant" in features and "pytorch_mixed_precision_cpu" in features:
                    continue

                if cpu_conversion:
                    features.append("pytorch_cuda_to_cpu")

                if features[0] == "" and len(features) > 1:
                    features = features[1:]  # remove ""

                bench_performance, bench_mode, bench_ws_path = enable(
                    code=code,
                    entry_code=entry_code,
                    entry_code_args=entry_code_args,
                    features=features,
                    mode=mode,
                    run_bench=True,
                    num_benchmark_iteration=num_benchmark_iteration,
                    cpu_set_env=cpu_set_env,
                    ncore_per_instance=ncore_per_instance,
                    ninstances=ninstances,
                    bench_batch_size=bench_batch_size,
                )

                def remove_if_have(list, element):
                    if element in list:
                        list.remove(element)
                    return list

                features = remove_if_have(features, "pytorch_benchmark")
                features = remove_if_have(
                    features, "pytorch_change_batch_size")
                features = remove_if_have(features, "pytorch_cuda_to_cpu")

                logger.info(
                    f"Benchmark result of acceleration set [{features}] is [{bench_performance[0]}] (FPS)")

                d = {}  # initialize dict
                d["features"] = features
                d["FPS"] = bench_performance[0]
                d["mode"] = bench_mode
                d["workspace_path"] = bench_ws_path
                result.append(d)

                list_FPS.append(bench_performance[0])
                list_features.append(features)
                list_mode.append(bench_mode)
                list_ws_path.append(bench_ws_path)

        # print result
        logger.info(
            f"Superbench result of sweeping [{sweep_objective}] printed below with sorted FPS: ")
        print("{:<20} {:<20} {:<120}".format(
            'Numactl Mode', 'Performance (FPS)', 'Features Applied'))
        sort_index = sorted(range(len(list_FPS)),
                            key=lambda k: list_FPS[k], reverse=True)
        for i in sort_index:
            if list_FPS[i] != 0:
                print("{:<20} {:<20} {:<120}".format(
                    str(list_mode[i]), str(list_FPS[i]), str(list_features[i])))

        # for superbench report generation
        list_optimization_set_top3 = []
        list_performance_top3 = []
        count_top3 = 0
        for i in sort_index:
            if list_FPS[i] != 0:
                list_performance_top3.append(list_FPS[i])
                list_optimization_set_top3.append(list_features[i])
                count_top3 += 1
                if count_top3 == 3:
                    break

        original_model_performance = 0
        original_model_ranking = 0
        for i in sort_index:
            if list_FPS[i] != 0:
                original_model_ranking += 1
                if list_features[i] == []:
                    original_model_performance = list_FPS[i]
                    break

        return list_optimization_set_top3, \
            list_performance_top3, original_model_ranking, original_model_performance

    elif sweep_objective == "bench_config":
        result_ncpi = []
        result_nins = []
        result_bs = []
        result_regular_thp = []
        result_p50_thp = []
        result_p90_thp = []
        result_p99_thp = []
        if bench_feature == []:
            logger.error(
                f'You must specify a feature (optimization set) for benchmark when "sweep_objective" \
                    is "bench_config"')
            quit()
        else:
            from .utils.cpu_info import get_num_cpu_cores
            ncores = get_num_cpu_cores()
            list_ncpi = [1, 2, 4, 8]
            for i in [1, 2, 4, 8]:
                list_ncpi.append(int(ncores / i))
            list_ncpi = list(set(list_ncpi))
            list_ncpi.sort()
            logger.debug(f"list_ncpi = {list_ncpi}")

            for this_ncpi in list_ncpi:
                ncore_per_instance = this_ncpi
                ninstances = int(ncores / this_ncpi)
                list_bs = [1, 2, 4, 8, this_ncpi * 1, this_ncpi * 2, this_ncpi *
                           4, this_ncpi * 8, this_ncpi * 16, this_ncpi * 32, this_ncpi * 64]
                list_bs = list(set(list_bs))
                list_bs.sort()
                if logging_level == "debug":
                    list_bs = [list_bs[-5]]
                logger.debug(f"this_ncpi = {this_ncpi}")
                logger.debug(f"list_bs = {list_bs}")
                for this_bs in list_bs:
                    bench_batch_size = this_bs
                    try:
                        bench_performance, bench_mode, bench_ws_path = enable(
                            code=code,
                            entry_code=entry_code,
                            entry_code_args=entry_code_args,
                            features=bench_feature,
                            mode="self_defined",  # sweep bench_config, so mode set to "self_defined"
                            run_bench=True,
                            num_benchmark_iteration=num_benchmark_iteration,
                            cpu_set_env=cpu_set_env,
                            ncore_per_instance=ncore_per_instance,
                            ninstances=ninstances,
                            bench_batch_size=bench_batch_size,
                        )

                        socket_regular_thp = bench_performance[0]
                        socket_p50_thp = round(
                            1000 / bench_performance[2] * ninstances * bench_batch_size, 3)
                        socket_p90_thp = round(
                            1000 / bench_performance[3] * ninstances * bench_batch_size, 3)
                        socket_p99_thp = round(
                            1000 / bench_performance[4] * ninstances * bench_batch_size, 3)

                        result_ncpi.append(ncore_per_instance)
                        result_nins.append(ninstances)
                        result_bs.append(bench_batch_size)
                        result_regular_thp.append(socket_regular_thp)
                        result_p50_thp.append(socket_p50_thp)
                        result_p90_thp.append(socket_p90_thp)
                        result_p99_thp.append(socket_p99_thp)

                        logger.info(
                            f"ncpi: {ncore_per_instance}, nins: {ninstances}, bs: {bench_batch_size}, \
                                regular_thp: {socket_regular_thp}, p50_thp: {socket_p50_thp}, \
                                    p90_thp: {socket_p90_thp}, p99_thp: {socket_p99_thp}")

                    except:
                        logger.warning(
                            f"ncpi: {ncore_per_instance}, nins: {ninstances}, bs: {bench_batch_size}, Benchmark \
                                failed. It might be due to HW limitation such as CPU load limit.")
                        continue

        # print result
        for item in [result_regular_thp, result_p50_thp, result_p90_thp, result_p99_thp]:
            if item is result_regular_thp:
                display_item_name = "Throughput"
            elif item is result_p50_thp:
                display_item_name = "Throughput based on P50-Latency"
            elif item is result_p90_thp:
                display_item_name = "Throughput based on P90-Latency"
            elif item is result_p99_thp:
                display_item_name = "Throughput based on P99-Latency"

            print("{:<30} {:<30} {:<30} {:<30}".format(
                'Num Cores Per Instance', 'Num of Instances', 'Batch Size', display_item_name))
            sort_index = sorted(
                range(len(item)), key=lambda k: item[k], reverse=True)
            for i in sort_index:
                print("{:<30} {:<30} {:<30} {:<30}".format(str(result_ncpi[i]), str(
                    result_nins[i]), str(result_bs[i]), str(item[i])))

        list_config_best_ncpi = []
        list_config_best_nins = []
        list_config_best_bs = []
        list_config_best_performance = []
        for item in [result_regular_thp, result_p50_thp, result_p90_thp, result_p99_thp]:
            sort_index = sorted(
                range(len(item)), key=lambda k: item[k], reverse=True)
            for i in sort_index:
                list_config_best_ncpi.append(result_ncpi[i])
                list_config_best_nins.append(result_nins[i])
                list_config_best_bs.append(result_bs[i])
                list_config_best_performance.append(item[i])
                break  # only fetch the top result

        return list_config_best_ncpi, list_config_best_nins, list_config_best_bs, list_config_best_performance
