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
import yaml
import re

from . import globals

if not os.path.exists("neural_coder_workspace"):
    os.makedirs("neural_coder_workspace")


def detect_device_(logger):
    # device detection
    logger.info(f"Device detection started ...")
    from .utils.device import detect_device
    detect_device()
    if globals.device == "cpu_with_amx":
        logger.info(f"Device: CPU with AMX")
    elif globals.device == "cpu_without_amx":
        logger.info(f"Device: CPU without AMX")
    elif globals.device == "intel_gpu":
        logger.info(f"Device: Intel(R) GPU")
    elif globals.device == "cuda":
        logger.info(f"Device: CUDA")
    elif globals.device == "mutli":
        logger.info(f"Device: Multi-Device")


def enable(
    code,
    features,
    target_batch_size=1,  # effective for feature "pytorch_change_batch_size"
    num_benchmark_iteration=10,  # effective for feature "pytorch_benchmark"
    generate_patch=True,
    overwrite=False,
    save_patch_path="",
    patch_suffix=".diff",
    remove_copy=True,
    consider_imports=True,
    patch_imports=False,
    logging_level="info",
    run_bench=False,
    entry_code="",
    args="",
    mode="throughput",
    cpu_set_env=True,
    ncore_per_instance=-1,  # only for "self_defined" mode
    ninstances=-1,  # only for "self_defined" mode
    bench_batch_size=-1,  # only for "self_defined" mode
    test_code_line=False, # print code line info for debug use
    cache_load_transformers=True,
    optimum_quant_config="", # only for HF optimum optimizations, yaml or hub path
    use_inc=False,
    use_modular=False,
    modular_item="",
):
    """enable a feature or a couple of features for the code

    """

    ### Preparation

    # set up workspace
    ws_path = "neural_coder_workspace/" + \
        "enable" + str(time.time()).replace(".","") + "/"
    os.makedirs(ws_path)

    # user parameters
    globals.consider_imports = consider_imports
    logging_var = "logging." + logging_level.upper()
    globals.logging_level = eval(logging_var)

    # set up logging
    logger = logging.getLogger(ws_path)
    logger.setLevel(globals.logging_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(ws_path + 'enable.log')
    fh.setLevel(globals.logging_level)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(globals.logging_level)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # device detection
    detect_device_(logger)

    # print key inputs
    logger.info(f"Enabling started ...")
    logger.info(f"code: {code}")
    logger.info(f"features: {features}")

    # feature list for reference
    '''
    feature_list = [
        "pytorch_jit_script",
        "pytorch_jit_script_ofi",
        "pytorch_jit_trace",
        "pytorch_jit_trace_ofi",
        "pytorch_inc_dynamic_quant",
        "pytorch_inc_static_quant_fx",
        "pytorch_inc_static_quant_ipex",
        "pytorch_inc_bf16",
        "pytorch_inc_huggingface_optimum_static",
        "pytorch_inc_huggingface_optimum_dynamic",
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
        "pytorch_torch_profiler",
        "pytorch_benchmark",
        "pytorch_change_batch_size",
        "pytorch_cuda_to_cpu",
        "pytorch_lightning_bf16_cpu",
        "pytorch_aliblade",
        "tensorflow_amp",
        "keras_amp",
        "tensorflow_inc",
        "keras_inc",
        "onnx_inc_static_quant_qlinear",
        "onnx_inc_static_quant_qdq",
        "onnx_inc_dynamic_quant",
        "inc_auto",
    ]
    '''

    ### Enable Benchmark (if run_bench)
    
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

    #### Feature Enabling

    globals.num_benchmark_iteration = str(num_benchmark_iteration + 10) # 10: warmup iteration number

    globals.cache_load_transformers = cache_load_transformers
    globals.optimum_quant_config = optimum_quant_config

    globals.use_modular = use_modular
    globals.modular_item = modular_item
    
    # move "pytorch_benchmark" to the last
    from .utils.common import move_element_to_last
    features = move_element_to_last(features, "pytorch_benchmark")

    # not in harness scope
    features_outside_harness = [
        "pytorch_change_batch_size",
        "pytorch_cuda_to_cpu",
        "pytorch_lightning_bf16_cpu",
        "tensorflow_mixed_precision",
        "change_trainer_to_nlptrainer",
    ]
    
    # # features that need creating dummy dataloader (when needed) first
    # if "pytorch_inc_static_quant_fx" in features or \
    #     "pytorch_inc_static_quant_ipex" in features:
    #     features = ["pytorch_dummy_dataloader"] + features
    
    # features that need reclaiming inputs first (e.g. for "for step, inputs in enumerate(dataloader)")
    if "pytorch_jit_trace" in features or \
        "pytorch_jit_trace_ofi" in features or \
        "pytorch_inc_static_quant_fx" in features or \
        "pytorch_inc_static_quant_ipex" in features:
        features = ["pytorch_reclaim_inputs"] + features

    # intel_extension_for_transformers
    if "intel_extension_for_transformers" in features:
        features = ["change_trainer_to_nlptrainer"] + features

    transformed_list_code_path = []

    ## Determine Code Domain
    # reset globals
    globals.reset_globals()

    from .utils import handle_user_input
    globals.list_code_path, num_user_code_path = handle_user_input.get_all_code_path(code)

    from .coders.autoinc import domain
    code_domain = domain.determine_domain(globals.list_code_path[0])
    if code_domain == "transformers_trainer":
        if "pytorch_benchmark" in features:
            features = ["pytorch_reclaim_inference_transformers_trainer"] + features
            # for BS
            args += " --per_device_eval_batch_size " + str(target_batch_size)
            globals.batch_size_changed = True

    ## Feature Transformation
    for idx_feature, feature in enumerate(features):

        # "inc_auto" auto selection of feature according to fwk
        if feature == "inc_auto":
            from .coders.autoinc import domain
            code_domain = domain.determine_domain(globals.list_code_path[0])
            if code_domain == "keras_script":
                feature = "keras_inc"
            elif code_domain == "tensorflow_keras_model":
                feature = "tensorflow_inc"
            elif code_domain == "onnx":
                feature = "onnx_inc_dynamic_quant"
            else:
                feature = "pytorch_inc_dynamic_quant"

        # reset globals
        globals.reset_globals()

        from .utils import handle_user_input
        globals.list_code_path, num_user_code_path = handle_user_input.get_all_code_path(code)
        if len(transformed_list_code_path) > 0:
            globals.list_code_path = transformed_list_code_path

        # common for all features (transformations),
        list_transformed_code = []
        # in this list, each item stores the transformed code
        # of the corresponding original code
        # by the order in code_path

        # global behaviors
        logger.info(
            f"Performing code transformation for feature: [{feature}] ...")

        for i in globals.list_code_path:
            list_transformed_code.append(open(i, 'r').read())

        ## 1. Features in Harness Scope
        if feature not in features_outside_harness:
            from .graphers.code_line import register_code_line
            from .graphers.model import register_nnModule_class, register_nnModule_instance_definition
            from .graphers.function import register_func_wrap_pair
            from .coders.transform import execute_insert_transformation, execute_indent_transformation

            # code analysis (call graph, type inference etc)
            register_code_line()
            register_func_wrap_pair()
            register_nnModule_class()
            if cache_load_transformers:
                preload_file = open(os.path.dirname(__file__) +
                    "/graphers/preloads/" + "transformers" + ".yaml")
                preload_dict = yaml.load(preload_file, Loader=yaml.BaseLoader)
                globals.list_class_name += preload_dict["class"]
            register_nnModule_instance_definition()
            # register transformation
            if feature == "pytorch_dummy_dataloader": # is not in harness scope, but needs call graph and type inference
                from .coders.pytorch.dummy_dataloader import DummyDataLoader
                opt = DummyDataLoader(globals.list_model_def_instance)
                opt.register_transformation()
            elif feature == "pytorch_reclaim_inputs":
                from .coders.pytorch.reclaim_inputs import ReclaimInputs
                opt = ReclaimInputs(globals.list_model_def_instance)
                opt.register_transformation()
            elif feature == "pytorch_reclaim_inference_transformers_trainer":
                from .coders.pytorch.reclaim_inference_transformers_trainer import ReclaimInferenceTransformersTrainer
                opt = ReclaimInferenceTransformersTrainer(globals.list_model_def_instance)
                opt.register_transformation()
            elif feature in [
                    "pytorch_inc_dynamic_quant",
                    "pytorch_inc_static_quant_fx",
                    "pytorch_inc_static_quant_ipex",
                    "pytorch_inc_huggingface_optimum_static",
                    "pytorch_inc_huggingface_optimum_dynamic",
                    "onnx_inc_static_quant_qlinear",
                    "onnx_inc_static_quant_qdq",
                    "onnx_inc_dynamic_quant",
                    "intel_extension_for_transformers",
                ]:

                # determine domain
                from .coders.autoinc.domain import determine_domain
                globals.code_domain = determine_domain(globals.list_code_path[0])

                # for transformers code, enable optimum-intel api by default
                # if specify use_inc, then still use INC API
                if "transformers" in globals.code_domain and not use_inc:
                    if "static_quant" in feature:
                        feature = "pytorch_inc_huggingface_optimum_static"
                    elif "dynamic_quant" in feature:
                        feature = "pytorch_inc_huggingface_optimum_dynamic"

                # optimum-intel quantization config for static and dynamic
                if feature == "pytorch_inc_huggingface_optimum_static":
                    globals.optimum_quant_config = "quantization/quant_config_static"
                elif feature == "pytorch_inc_huggingface_optimum_dynamic":
                    globals.optimum_quant_config = "quantization/quant_config_dynamic"
                else:
                    pass

                from .coders.autoinc.autoinc_harness import AutoInc_Harness
                from .coders.autoinc.calib_dataloader import Calib_Dataloader
                from .coders.autoinc.eval_func import Eval_Func
                opt = Calib_Dataloader()
                opt.register_transformation()

                opt = Eval_Func()
                opt.register_transformation()

                opt = AutoInc_Harness(backend=feature)
                opt.register_transformation()
            else:
                from .coders.pytorch.harness import Harness
                opt = Harness(backend=feature)
                opt.register_transformation()

            # execute transformation
            list_transformed_code = execute_indent_transformation(list_transformed_code)
            list_transformed_code = execute_insert_transformation(list_transformed_code)

        ## 2. Features NOT in Harness Scope
        else:
            for i in range(len(list_transformed_code)):
                # Batch Size
                if "pytorch_change_batch_size" in features:
                    if "batch_size" in list_transformed_code[0]:  # entry code has "batch_size"
                        globals.batch_size_changed = True
                    from .coders.pytorch.batch_size import BatchSizeCoder
                    globals.target_batch_size = str(target_batch_size)
                    list_transformed_code[i] = BatchSizeCoder(list_transformed_code[i]).transform()
                # CUDA to CPU
                if "pytorch_cuda_to_cpu" in features:
                    from .coders.pytorch.cuda_to_cpu import CudaToCpu
                    list_transformed_code[i] = CudaToCpu(list_transformed_code[i]).transform()
                # Lightning
                if "pytorch_lightning_bf16_cpu" in features:
                    from .coders.pytorch.lightning import Lightning
                    list_transformed_code[i] = Lightning(list_transformed_code[i]).transform()
                # TF & Keras AMP
                if "tensorflow_mixed_precision" in features:
                    from .coders.tensorflow.amp import TensorFlowKerasAMP
                    list_transformed_code[i] = TensorFlowKerasAMP(list_transformed_code[i]).transform()
                if "tensorflow_inc" in features:
                    from .coders.tensorflow.inc import TensorFlowKerasINC
                    list_transformed_code[i] = TensorFlowKerasINC(list_transformed_code[i]).transform()
                # Change Trainer to NLPTrainer (only for intel_extension_for_pytorch)
                if "change_trainer_to_nlptrainer" in features:
                    from .coders.pytorch.change_trainer_to_nlptrainer import TrainerToNLPTrainer
                    list_transformed_code[i] = TrainerToNLPTrainer(list_transformed_code[i]).transform()

        logger.info(f"Code transformation for feature: [{feature}] finished.")

        for idx_path, path in enumerate(globals.list_code_path):
            if path[-14:] == "_nc_enabled.py":
                path_transformed = path
            else:
                path_transformed = path[:-3] + "_nc_enabled.py"
            if idx_feature != len(features) - 1:
                open(path_transformed, "w").write(list_transformed_code[idx_path])
            else:
                open(path_transformed, "w").write(list_transformed_code[idx_path].replace(" # [coder-enabled]", ""))
            globals.list_code_path[idx_path] = path_transformed
        transformed_list_code_path = globals.list_code_path

    # test code_line.py
    if test_code_line:
        # reset globals
        globals.reset_globals()
        globals.print_code_line_info = True

        from .utils import handle_user_input
        globals.list_code_path, num_user_code_path = handle_user_input.get_all_code_path(code)
        if len(transformed_list_code_path) > 0:
            globals.list_code_path = transformed_list_code_path

        # common for all features (transformations),
        list_transformed_code = []
        # in this list, each item stores the transformed code
        # of the corresponding original code
        # by the order in code_path

        for i in globals.list_code_path:
            list_transformed_code.append(open(i, 'r').read())
        
        from .graphers.code_line import register_code_line
        from .graphers.model import register_nnModule_class, register_nnModule_instance_definition
        from .graphers.function import register_func_wrap_pair
        from .coders.transform import execute_insert_transformation, execute_indent_transformation

        # code analysis (call graph, type inference etc)
        register_code_line()
        register_func_wrap_pair()
        register_nnModule_class()
        register_nnModule_instance_definition()

    ### Output of Enabling
    globals.list_code_path, num_user_code_path = handle_user_input.get_all_code_path(code)

    if generate_patch:
        whole_patch_user_code = ""
        for path in globals.list_code_path[0:num_user_code_path]:
            path_transformed = path[:-3] + "_nc_enabled.py"
            if path_transformed[-25:] == "_nc_enabled_nc_enabled.py":
                continue
            cmd_gen_patch = "diff -up " + path + " " + path_transformed
            sp_gen_patch = subprocess.Popen(
                cmd_gen_patch, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
            sp_gen_patch.wait()
            this_patch, _ = sp_gen_patch.communicate()
            this_patch = str(this_patch)[2:-1]
            whole_patch_user_code += this_patch
        if save_patch_path == "":
            save_patch_path = ws_path + "neural_coder_patch"
        open(save_patch_path + patch_suffix, "w").write(
            whole_patch_user_code.replace(r'\n', '\n').replace(r'\t', '\t').replace(r"\'", "\'"))
        abs_patch_path = os.path.abspath(
            save_patch_path + patch_suffix)
        logger.info(f"The patch is saved to: [{abs_patch_path}]")

        if overwrite:
            sp_overwrite = subprocess.Popen(
                "patch -d/ -p0 < " + abs_patch_path, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
            sp_overwrite.wait()
            # os.remove(abs_patch_path)  # remove patch after overwrite

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
            if save_patch_path == "":
                save_patch_path = ws_path + "neural_coder_patch_import_modules"
            open(save_patch_path + patch_suffix, "w").write(
                whole_patch_import_modules.replace(r'\n', '\n').replace(r'\t', '\t').replace(r"\'", "\'"))
            abs_patch_path = os.path.abspath(
                save_patch_path + patch_suffix)
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

    ### Benchmark
    if run_bench:
        if "github.com" in code and ".py" in code:
            code = globals.list_code_path[0]
            entry_code = globals.list_code_path[0]

        bench_performance, bench_mode, bench_ws_path = bench(
            code=code,
            entry_code=entry_code,
            args=args,
            patch_path=abs_patch_path,
            mode=mode,
            cpu_set_env=cpu_set_env,
            ncore_per_instance=ncore_per_instance,  # only for "self_defined" mode
            ninstances=ninstances,  # only for "self_defined" mode
            bench_batch_size=bench_batch_size,  # only for "self_defined" mode
        )

        return bench_performance, bench_mode, bench_ws_path


def bench(
    code,
    entry_code="",
    args="",
    patch_path="",
    mode="throughput",  # throughput, latency, multi_instance or self_defined
    logging_level="info",
    cpu_set_env=True,
    ncore_per_instance=-1,  # only for "self_defined" mode
    ninstances=-1,  # only for "self_defined" mode
    bench_batch_size=-1,  # only for "self_defined" mode
):
    """benchmark on either "optimized code", or "patch" + "original code"
    it does not enable benchmark code lines, or enable change of batch size
    all the enabling should be done within enable API
    which means the "optimized code" should already have
    "pytorch_benchmark" and "pytorch_change_batch_size" enabled
    or the "patch" should already have the code modification
    for "pytorch_benchmark" and "pytorch_change_batch_size" in it
    """
    # set up workspace
    ws_path = "neural_coder_workspace/" + "bench" + str(time.time()).replace(".","") + "/"
    os.makedirs(ws_path)

    # set up logging
    logging_var = "logging." + logging_level.upper()
    globals.logging_level = eval(logging_var)

    logger = logging.getLogger(ws_path)
    logger.setLevel(globals.logging_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(ws_path + 'bench.log')
    fh.setLevel(globals.logging_level)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(globals.logging_level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # device detection
    detect_device_(logger)

    # print key inputs
    logger.info(f"Benchmarking started ...")
    logger.info(f"code: {code}")
    logger.info(f"mode: {mode}")

    # entry code
    if entry_code == "":
        # if not specify entry_code,
        # then code has to be a list of one element, or a single string of single path, otherwise quit
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
        sp_patch = subprocess.Popen(
            "patch -d/ -p0 < " + patch_path, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
        sp_patch.wait()

    # if mode is "self_defined", user must specify ncpi, nins and bs
    if mode == "self_defined":
        if ncore_per_instance == -1 or ninstances == -1 or bench_batch_size == -1:
            logger.error(
                f"You have to specify ncore_per_instance,"
                f"ninstances and bench_batch_size for self-defined benchmark mode.")
            quit()

    # numactl
    from .utils import numa_launcher

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
        cmd_env += 'export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,'
        cmd_env += 'dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"'
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

    args = [args]
    numa_launcher.exec_launcher(
        ncore_per_instance, ninstances, entry_code, args, bench_log_path)

    # get performance (throughput and latency)
    bench_log = open(bench_log_path, "r", encoding='unicode_escape').read().split('\n')
    IPS = []
    MSPI = 0
    count_MSPI = 0
    P50 = 0
    count_P50 = 0
    P90 = 0
    count_P90 = 0
    P99 = 0
    count_P99 = 0
    acc_delta = 0
    for line in bench_log:
        if "Neural_Coder_Bench_IPS" in line:
            try:
                IPS.append(float(line[line.find(":")+3:]))
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
        if "Accuracy (int8|fp32)" in line:
            try:
                acc_int8 = float(re.search(r"\d+\.\d+", line).group())
                acc_fp32 = float(re.search(r"(?<=\|)\d+\.\d+", line).group())
                acc_delta = round((acc_int8 - acc_fp32) / acc_fp32 * 100, 2) # percent of increase/decrease
            except ValueError as ve:
                pass

    if len(IPS) >= 4:  # handle extreme values
        IPS.sort()
        IPS[0] = IPS[1]
        IPS[-1] = IPS[-2]

    try:
        if globals.batch_size_changed: # only times BS if BS has been modified, otherwise times 1
            FPS = round(sum(IPS) / len(IPS) * ninstances * bench_batch_size, 3)
        else:
            FPS = round(sum(IPS) / len(IPS) * ninstances * 1, 3)
    except:
        FPS = 0
    try:
        MSPI = round(MSPI / count_MSPI, 3)
    except:
        MSPI = 0
    try:
        P50 = round(P50 / count_P50, 3)
    except:
        P50 = 0
    try:
        P90 = round(P90 / count_P90, 3)
    except:
        P90 = 0
    try:
        P99 = round(P99 / count_P99, 3)
    except:
        P99 = 0

    logger.info(f"Collected throughput on the code is: [{FPS}] (fps)")
    logger.info(f"Collected latency on the code is: [{MSPI}] (mspi)")
    logger.info(f"Collected latency_p50 on the code is: [{P50}] (mspi)")
    logger.info(f"Collected latency_p90 on the code is: [{P90}] (mspi)")
    logger.info(f"Collected latency_p99 on the code is: [{P99}] (mspi)")
    logger.info(f"Collected accuracy delta on the code is: [{acc_delta}]")

    # unpatch
    if patch_path != "":
        sp_unpatch = subprocess.Popen(
            "patch -R -d/ -p0 < " + patch_path, env=os.environ, shell=True, stdout=subprocess.PIPE)  # nosec
        sp_unpatch.wait()

    return [FPS, MSPI, P50, P90, P99, acc_delta], mode, os.path.abspath(ws_path)


def superbench(
    code,
    entry_code="",
    args="",
    sweep_objective="feature",  # "feature" or "bench_config"
    specify_features=[],
    bench_feature=[],  # only effective when sweep_objective is "bench_config"
    mode="throughput",
    num_benchmark_iteration=5,
    iteration_dynamic_adjust=True,
    logging_level="info",
    cpu_set_env=True,
    ncore_per_instance=-1,  # only for "self_defined" mode
    ninstances=-1,  # only for "self_defined" mode
    bench_batch_size=-1,  # only for "self_defined" mode
    use_inc=False,
    auto_quant=False,
):

    # set up workspace
    ws_path = "neural_coder_workspace/" + \
        "superbench" + str(time.time()).replace(".","") + "/"
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

    # device detection
    detect_device_(logger)

    # print key inputs
    if auto_quant:
        logger.info(f"Auto-Quant started ...")
        logger.info(f"Code: {code}")
        logger.info(f"Benchmark Mode: {mode} mode")
        logger.debug(f"Number of benchmark iterations: {num_benchmark_iteration}")
    else:
        logger.info(f"SuperBench started ...")
        logger.info(f"Code: {code}")
        logger.info(f"Benchmark Mode: {mode} mode")
        logger.debug(f"Sweep Objective: {sweep_objective}")
        logger.debug(f"Number of benchmark iterations: {num_benchmark_iteration}")

    # entry code
    if entry_code == "":
        # if not specify entry_code,
        # then code has to be a list of one element,
        # or a single string of single path, otherwise quit
        if type(code) == list and len(code) == 1:
            entry_code = code[0]
        elif type(code) == str:
            entry_code = code
        else:
            logger.error(
                f"You have to specify an entry_code of your code: [{code}]")
            quit()

    # detect device compatibility of entry code
    from .utils.device import detect_code_device_compatibility
    detect_code_device_compatibility(entry_code)

    if sweep_objective == "feature":
        list_FPS = []
        list_accuracy = []
        list_features = []
        list_mode = []
        list_ws_path = []
        result = []

        if auto_quant:
            backends = [
                [],
                ["pytorch_inc_dynamic_quant"],
                ["pytorch_inc_static_quant_fx"],
                ["pytorch_inc_static_quant_ipex"],
                ["pytorch_inc_bf16"],
            ]
            standalones_pool = []
        elif len(specify_features) != 0:
            backends = [
                [],
            ]
            for item in specify_features:
                backends.append([item])
            standalones_pool = []
        else:
            # features that is a "backend":
            backends = [
                "",
                "pytorch_ipex_fp32",
                "pytorch_ipex_bf16",
                "pytorch_inc_static_quant_fx",
                "pytorch_inc_static_quant_ipex",
                "pytorch_inc_dynamic_quant",
                "pytorch_ipex_int8_static_quant",
                "pytorch_ipex_int8_dynamic_quant",
            ]
            # features that can be standalone (either use alone or use with "backend"):
            standalones_pool = [
                "pytorch_mixed_precision_cpu",
                "pytorch_channels_last",
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
                    standalones.append(list(item))  # only appends the item with one JIT feature in it

        dry_run = True
        for backend in backends:
            for standalone in standalones:
                features = []
                if auto_quant:
                    features += backend
                elif len(specify_features) != 0:
                    features += backend
                else:
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
                if "pytorch_inc_static_quant_fx" in features and "pytorch_mixed_precision_cpu" in features:
                    continue
                if "pytorch_inc_static_quant_ipex" in features and "pytorch_mixed_precision_cpu" in features:
                    continue
                if "pytorch_inc_dynamic_quant" in features and "pytorch_mixed_precision_cpu" in features:
                    continue

                # device conversion
                if "cpu" in globals.device and "cpu" not in globals.list_code_device_compatibility:
                    features.append("pytorch_cuda_to_cpu")

                if features[0] == "" and len(features) > 1:
                    features = features[1:]  # remove ""

                if dry_run:
                    t_start = time.time()

                bench_performance, bench_mode, bench_ws_path = enable(
                    code=code,
                    entry_code=entry_code,
                    args=args,
                    features=features,
                    mode=mode,
                    run_bench=True,
                    num_benchmark_iteration=num_benchmark_iteration,
                    cpu_set_env=cpu_set_env,
                    ncore_per_instance=ncore_per_instance,
                    ninstances=ninstances,
                    bench_batch_size=bench_batch_size,
                    use_inc=use_inc,
                )

                if dry_run:
                    t_end = time.time()
                    if iteration_dynamic_adjust:
                        num_benchmark_iteration = max(int(300 / (t_end - t_start)), 5)
                        logger.debug(
                            f"Adjusted number of benchmark iterations after dry-run is {num_benchmark_iteration}")
                    dry_run = False

                def remove_if_have(list, element):
                    if element in list:
                        list.remove(element)
                    return list

                features = remove_if_have(features, "pytorch_benchmark")
                features = remove_if_have(features, "pytorch_change_batch_size")
                features = remove_if_have(features, "pytorch_cuda_to_cpu")

                if auto_quant:
                    # convert feature name to display name for better user experience
                    if features == ['pytorch_inc_dynamic_quant']:
                        features_display = "Intel INT8 (Dynamic)"
                    elif features == ['pytorch_inc_static_quant_fx']:
                        features_display = "Intel INT8 (Static)"
                    elif features == ['pytorch_inc_static_quant_ipex']:
                        features_display = "Intel INT8 (IPEX)"
                    elif features == ['pytorch_inc_bf16']:
                        features_display = "Intel BF16"
                    elif features == []:
                        features_display = "The Original Model"

                    logger.info(
                        f"Benchmark result (performance) of {features_display}"
                        f" is {bench_performance[0]} (FPS)")
                    logger.info(
                        f"Benchmark result (accuracy delta) of {features_display} is {bench_performance[5]} %")
                else:
                    logger.info(
                        f"Benchmark result (performance) of optimization set [{features}]"
                        f" is [{bench_performance[0]}] (FPS)")
                    logger.info(
                        f"Benchmark result (accuracy delta) of optimization set [{features}]"
                        f" is [{bench_performance[5]}] %")

                d = {}  # initialize dict
                d["features"] = features
                d["FPS"] = bench_performance[0]
                d["accuracy"] = bench_performance[5]
                d["mode"] = bench_mode
                d["workspace_path"] = bench_ws_path
                result.append(d)

                list_FPS.append(bench_performance[0])
                list_accuracy.append(bench_performance[5])
                list_features.append(features)
                list_mode.append(bench_mode)
                list_ws_path.append(bench_ws_path)

        # print result
        print(f"Superbench result of sweeping [{sweep_objective}] printed below with sorted FPS: ")
        print("{:<20} {:<20} {:<20} {:<120}".format(
            'Numactl Mode', 'Performance (FPS)', 'Accuracy Delta (%)', 'Features Applied'))

        sort_index = sorted(
            range(len(list_FPS)),
            key=lambda k: list_FPS[k],
            reverse=True,
        )

        for i in sort_index:
            if list_FPS[i] != 0:
                print(
                    "{:<20} {:<20} {:<20} {:<120}".format(
                        str(list_mode[i]),
                        str(list_FPS[i]),
                        str(list_accuracy[i]),
                        str(list_features[i]),
                    )
                )

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
        
        if auto_quant:
            # convert feature name to display name for better user experience
            if list_optimization_set_top3[0] == ['pytorch_inc_dynamic_quant']:
                best_optimization_display = "Intel INT8 (Dynamic)"
            elif list_optimization_set_top3[0] == ['pytorch_inc_static_quant_fx']:
                best_optimization_display = "Intel INT8 (Static)"
            elif list_optimization_set_top3[0] == ['pytorch_inc_static_quant_ipex']:
                best_optimization_display = "Intel INT8 (IPEX)"
            elif list_optimization_set_top3[0] == ['pytorch_inc_bf16']:
                best_optimization_display = "Intel BF16"
            elif list_optimization_set_top3[0] == []:
                best_optimization_display = "The Original Model"

            logger.info(f"The best optimization set for your model is {best_optimization_display}")
            logger.info(
                f"You can get up to "
                f"{round(list_performance_top3[0] / original_model_performance, 1)}"
                f" X performance boost."
            )
        else:
            logger.info(f"The best optimization set for your model is: {list_optimization_set_top3[0]}")
            logger.info(
                f"You can get up to "
                f"{round(list_performance_top3[0] / original_model_performance, 1)}"
                f" X performance boost."
            )

        # generate patch for the best optimization
        features_to_generate = list_optimization_set_top3[0]
        features_to_generate.append("pytorch_cuda_to_cpu")
        enable(
            code=code,
            features=features_to_generate,
            save_patch_path="intel_optimization",
            use_inc=use_inc,
        )
        logger.info('The optimization patch was saved to "intel_optimziation.diff"')

        return list_optimization_set_top3, list_performance_top3, original_model_ranking, original_model_performance

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
                f'You must specify a feature (optimization set) '
                f'for benchmark when "sweep_objective" is "bench_config"')
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

            dry_run = True
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

                        if dry_run:
                            t_start = time.time()

                        bench_performance, bench_mode, bench_ws_path = enable(
                            code=code,
                            entry_code=entry_code,
                            args=args,
                            features=bench_feature,
                            mode="self_defined",  # sweep bench_config, so mode set to "self_defined"
                            run_bench=True,
                            num_benchmark_iteration=num_benchmark_iteration,
                            cpu_set_env=cpu_set_env,
                            ncore_per_instance=ncore_per_instance,
                            ninstances=ninstances,
                            bench_batch_size=bench_batch_size,
                            use_inc=use_inc,
                        )

                        if dry_run:
                            t_end = time.time()
                            if iteration_dynamic_adjust:
                                num_benchmark_iteration = max(int(300 / (t_end - t_start)), 5)
                                logger.debug(
                                    f"Adjusted number of benchmark iterations after dry-run is "
                                    f"{num_benchmark_iteration}")
                            dry_run = False

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
                            f"ncpi: {ncore_per_instance}, nins: {ninstances}, bs: {bench_batch_size}, "
                            f"regular_thp: {socket_regular_thp}, p50_thp: {socket_p50_thp}, "
                            f"p90_thp: {socket_p90_thp}, p99_thp: {socket_p99_thp}"
                        )

                    except:
                        logger.warning(
                            f"ncpi: {ncore_per_instance}, nins: {ninstances}, bs: {bench_batch_size}, "
                            f"Benchmark failed. It might be due to HW limitation such as CPU load limit."
                        )
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


# def superreport(
#     code,
#     save_path="superbench_report.pdf",
#     logging_level="info",
#     platform="bare_metal",
#     bare_metal_machine_type="SPR",
# ):

#     from .utils.pdf_report import PDFReport

#     logging_level = logging_level
#     res1, res2, res3, res4 = superbench(
#         code=code,
#         sweep_objective="feature",
#         mode="throughput",
#         logging_level=logging_level,
#     )

#     res5, res6, res7, res8 = superbench(
#         code=code,
#         sweep_objective="bench_config",
#         bench_feature=res1[0],
#         logging_level=logging_level,
#     )
#     res1[0] = res1[0][0:-2]

#     if platform == "AWS":
#         # get AWS cloud_vendor and cloud_instance_type
#         # pricing: https://aws.amazon.com/ec2/pricing/on-demand/
#         import subprocess
#         res = subprocess.Popen(
#             "grep 'DMI' /var/log/dmesg", 
#             shell=True,                  
#             stdout=subprocess.PIPE,      
#             stderr=subprocess.PIPE,      
#         )
#         res.wait()
#         result = res.stdout.read()       
#         result = str(result, encoding="utf-8")
#         cloud_vendor = result.split()[4] + ' ' + result.split()[5]
#         if cloud_vendor == 'Amazon EC2':
#             cloud_vendor = 'AWS'
#         cloud_instance_type = result.split()[6].strip(',').strip('/')

#         # pricing to get automatically from AWS website
#         import pandas as pd
#         url = 'https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-2/index.csv'
#         df = pd.read_csv(url, skiprows=5, delimiter=',')
#         for i in range(len(df)):
#             if df['Location Type'][i] == "AWS Region" and df['Instance Type'][i] == cloud_instance_type \
#                 and df['Tenancy'][i] == "Shared" and df['Operating System'][i] == "Linux" \
#                 and df['operation'][i] == "RunInstances" and df['CapacityStatus'][i] == "UnusedCapacityReservation":
#                 cloud_unit_price = float(df['PricePerUnit'][i])
#     elif platform == "bare_metal":
#         cloud_vendor="Intel internal machine"
#         cloud_instance_type=bare_metal_machine_type
#         cloud_unit_price="1"
            
#     report = PDFReport(
#         path=save_path,
#         list_optimization_set_top3=res1,
#         list_performance_top3=res2,
#         original_model_ranking=res3,
#         original_model_performance=res4,
#         list_config_best_ncpi=res5,
#         list_config_best_nins=res6,
#         list_config_best_bs=res7,
#         list_config_best_performance=res8,
#         TCO_unit_pricing=cloud_unit_price,  # 2.448
#         cloud_vendor=cloud_vendor,  # "AWS"
#         cloud_instance_type=cloud_instance_type,  # "c6i"
#     )


def auto_quant(
    code,
    entry_code="",
    args="",
    sweep_objective="feature",
    bench_feature=[],
    mode="throughput",
    num_benchmark_iteration=30,
    iteration_dynamic_adjust=False,
    logging_level="info",
    cpu_set_env=True,
    ncore_per_instance=-1,  # only for "self_defined" mode
    ninstances=-1,  # only for "self_defined" mode
    bench_batch_size=-1,  # only for "self_defined" mode
    use_inc=False,
):
    return superbench(
        code,
        entry_code=entry_code,
        args=args,
        sweep_objective=sweep_objective,
        bench_feature=bench_feature,
        mode=mode,
        num_benchmark_iteration=num_benchmark_iteration,
        iteration_dynamic_adjust=iteration_dynamic_adjust,
        logging_level=logging_level,
        cpu_set_env=cpu_set_env,
        ncore_per_instance=ncore_per_instance,  # only for "self_defined" mode
        ninstances=ninstances,  # only for "self_defined" mode
        bench_batch_size=bench_batch_size,  # only for "self_defined" mode
        use_inc=use_inc,
        auto_quant=True,
    )
