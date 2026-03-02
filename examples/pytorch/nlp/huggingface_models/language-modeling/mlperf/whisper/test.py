# from numa import schedule, memory
from vllm import LLM, SamplingParams
import os
import time

import librosa
import torch
from vllm.config import CompilationConfig, CompilationLevel
# NODE_LIST = [5]
# OMP_NUM_THREADS = 6
# START_CORE = 214
# OMP_THREADS_BIND = f"{START_CORE}-{START_CORE+OMP_NUM_THREADS-1}"
MODEL_PATH="/model/whisper-large-v3/"
# MODEL_PATH = "/model/whisper-large-v3-w4a8g-1"
SAMPLE = "/data/dev-all-repack/116-288045_0.wav"
PREFILL_BATCH_SIZE=1
DECODE_BATCH_SIZE=1

# os.environ["VLLM_CPU_OMP_THREADS_BIND"]=f"{OMP_THREADS_BIND}"
# os.environ["OMP_NUM_THREADS"]=f"{OMP_NUM_THREADS}"
# os.environ["VLLM_CPU_KVCACHE_SPACE"]="32"

def setup_profiler(enabled):
    if not enabled:
        return None
    schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0)
    DEVICE = 'xpu:0'
    activities = [torch.profiler.ProfilerActivity.XPU]

    profiler = torch.profiler.profile(
        schedule=schedule,
        activities=activities,
        #debug_activities=debug_activities,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler(
        #       'pytorch_profiler_internal',
        #       use_gzip=True),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=True)
    return profiler

class DummyProfiler:
    def start(*args, **kwargs):
        pass
    def step(*args, **kwargs):
        pass
    def stop(*args, **kwargs):
        pass

def main():
    # profiler_prefill = DummyProfiler()
    # profiler_decode = DummyProfiler()
    enabled = False
    profiler_prefill = setup_profiler(enabled)
    profiler_decode = setup_profiler(enabled)
    # memory.set_membind_nodes(*NODE_LIST)
    prompt = {
        "prompt": "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        "multi_modal_data": {
            "audio": librosa.load(SAMPLE, sr=16000)
        },
    }
    print("Audio length", len(prompt["multi_modal_data"]["audio"][0]))
    model = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        skip_tokenizer_init=False,
        # trust_remote_code=True,
        tensor_parallel_size=1,
        max_num_seqs=DECODE_BATCH_SIZE,
        max_model_len=448,
        # max_num_batched_tokens=800,
        gpu_memory_utilization=0.4,
        # num_scheduler_steps=1,
        limit_mm_per_prompt={"audio": 1},
        kv_cache_dtype="auto",
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            use_inductor=True,
            # compile_sizes=[64,128,256],
        )
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=200
    )
    
    req_counter = 0

    # Prefill
    if enabled:
        profiler_prefill.start()
    for i in range(DECODE_BATCH_SIZE):
        for j in range(PREFILL_BATCH_SIZE):
            model.llm_engine.add_request(str(req_counter), prompt, sampling_params)
            req_counter += 1
        end = time.time()
        model.llm_engine.step()
        if enabled:
            profiler_prefill.step()
        print(f"Prefill bs={PREFILL_BATCH_SIZE} time {(time.time()-end)*1000:.2f}ms")
    if enabled:
        profiler_prefill.stop()

    # Decode
    results = []
    if enabled:
        profiler_decode.start()
    while model.llm_engine.has_unfinished_requests():
        end = time.time()
        step_outputs = model.llm_engine.step()
        if enabled:
            profiler_decode.step()
        print(f"Decode bs={DECODE_BATCH_SIZE} time {(time.time()-end)*1000:.2f}ms")
        for output in step_outputs:
            if output.finished:
                id = int(output.request_id)
                print(f"Finished 1, {output.outputs[0].text}")
    if enabled:
        profiler_decode.stop()

if __name__ == "__main__":
    main()