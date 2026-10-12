import os
import time
import numpy as np
import math
import array
import torch
from typing import Tuple
from transformers import AutoTokenizer, AutoConfig, AutoModel

import time
import threading
import torch.multiprocessing as mp
import gc
import types

import logging

import mlperf_loadgen as lg
from dataset import Dataset

import os
from tqdm import tqdm
import sys

from vllm import LLM
from vllm.inputs import TokensPrompt
from vllm.config import CompilationConfig, CompilationLevel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SUT")

#MAX_NUM_BATCHED_TOKENS=int(os.environ.get("MAX_NUM_BATCHED_TOKENS", 32768))
MAX_NUM_BATCHED_TOKENS=4096
MAX_MODEL_LEN=int(os.environ.get("MAX_MODEL_LEN", MAX_NUM_BATCHED_TOKENS))
PREFILL_BLOCK_SIZE = int(os.environ.get("VLLM_PROMPT_SEQ_BUCKET_STEP", "128"))
PREFILL_MIN_SIZE = int(os.environ.get("VLLM_PROMPT_SEQ_BUCKET_MIN", "128"))
PREFILL_MAX_SIZE = int(os.environ.get("VLLM_PROMPT_SEQ_BUCKET_MAX", "1024"))
SERVER_TIME_LIMIT = float(os.environ.get("SERVER_TIME_LIMIT", "2"))
SERVER_COMPUTE_TIME = float(os.environ.get("SERVER_COMPUTE_TIME", "1"))
BATCHED_PREFILL = os.environ.get("BATCHED_PREFILL", "1")=="1"
MAX_BATCHED_PROMPT_LEN = int(os.environ.get('VLLM_MAX_BATCHED_PROMPT_LEN', sys.maxsize))
ENABLE_TORCH_COMPILE = int(os.environ.get("ENABLE_TORCH_COMPILE", "0"))==1

MODEL_NAME = os.environ.get("MODEL_NAME", "").lower()

MODEL_INIT_DIR = os.environ.get("MODEL_INIT_DIR", "")
XPU_COUNT = int(os.environ.get("XPU_COUNT", "0"))

BLOCK_SIZE = 8
sys.path.append(MODEL_INIT_DIR)

from utils_model import *

# Copied from vllm/worker/hpu_model_runner.py
def warmup_range(config: Tuple[int, int, int]):
    """Generate a warmup range.

    Start from bmin and multiply by 2 until you reach bstep.
    Then, increase the values in the range by the value of bstep until you 
    reach bmax.

    Example:
    bmin = 2, bstep = 32, bmax = 64
    => ramp_up = (2, 4, 8, 16)
    => stable = (32, 64)
    => return ramp_up + stable => (2, 4, 8, 16, 32, 64)
    """
    import itertools
    import operator
    bmin, bstep, bmax = config
    if bmin > bmax:
        raise ValueError("Min. batch size cannot be greater than max. "
                        "batch size. If you want to skip warmup, "
                        "set VLLM_SKIP_WARMUP=true")
    base = itertools.repeat(2)
    ramp_up_acc = itertools.accumulate(base, func=operator.mul, initial=bmin)
    ramp_up_tw = itertools.takewhile(lambda x: x < bstep and x <= bmax, \
        ramp_up_acc)
    stable = range(bstep, bmax + 1, bstep)
    buckets = list(ramp_up_tw) + list(stable)
    return list(filter(lambda bucket: bucket >= bmin, buckets))

PREFILL_BUCKETS = warmup_range((PREFILL_MIN_SIZE, PREFILL_BLOCK_SIZE, PREFILL_MAX_SIZE))

def len_to_bucket(length):
    bucket = 0
    while length>PREFILL_BUCKETS[bucket]:
        bucket+=1
    return bucket

def void(*args, **kwargs):
    pass

# Disable prints if progress bar is active
PBAR = int(os.environ.get("PBAR", "1"))
# Update frequency of the progress bar
PBAR_FREQ = int(os.environ.get("PBAR_FREQ", "10"))
if PBAR==1:
    print = void

class Instance(mp.Process):
    def __init__(self,**kwargs):
        super(Instance, self).__init__()
        for key, value in kwargs.items():
            # Set each keyword argument as an attribute of the instance
            setattr(self, key, value)

        if hasattr(self, "lg_settings"):
            self.max_ttft_latency = self.lg_settings.ttft_latency/1e9
            self.max_tpot_latency = self.lg_settings.tpot_latency/1e9
        else:
            self.max_ttft_latency = 2
            self.max_tpot_latency = 0.2
        self.query_idx_mapping = []
        self.qid_mapping = []
        self.start_time_mapping = []
        self.wait_time_mapping = []
        self.first_time_mapping = []
        self.first_token_id = []
        self.finished = False
        
        self.tpot_list = []
        self.tprefill_list = []
        self.nprefill_list = []
        

        # Extend/override Instance methods if it's defined
        try:
            self.instance_override = types.MethodType(INSTANCE_OVERRIDE, self)
        except NameError:
            pass
        else:
            self.instance_override()

    def do_warmup(self):
        dummy_input = torch.zeros((128,), dtype=torch.int32)
        inputs = TokensPrompt(prompt_token_ids=dummy_input.tolist())
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=8,
            min_tokens=8,
            temperature=0.0,
            detokenize=False
        )
        # Using a large value for request_id to avoid collisions
        for i in range(1):
            self.model.llm_engine.add_request(str(int(1e8)),
                inputs,
                sampling_params)

        while self.model.llm_engine.has_unfinished_requests():
            step_outputs = self.model.llm_engine.step()
            for output in step_outputs:
                request_id = int(output.request_id)
                if output.finished:
                    token_ids = output.outputs[0].token_ids
                    log.info(f"[{time.time():.3f}] Warmup finished, "
                        f"Rank {self.rank}, "
                        f"Generated {len(token_ids)} tokens")

        with self.cond_var:
            self.alive_counter.value += 1
            self.cond_var.notify()

    def step_engine_prompt(self, *args):
        self.step_engine(*args)

    def run(self):
        gc.disable()

        # Select correct XPU device
        if XPU_COUNT>0:
            os.environ["ONEAPI_DEVICE_SELECTOR"]=f'level_zero:{self.xpu_devices}'
            pass
        
        self.load_dataset()

        log.info("Loading Model")
        self.load_model()

        self.sampling_params = SAMPLING_PARAMS

        self.warmup_req_idx = set([])
        if self.warmup:
            self.do_warmup()
        else:
            with self.cond_var:
                self.alive_counter.value += 1
                self.cond_var.notify()

        keep_alive = True
        self.req_counter = 0 # Request counter
        
        # Running seq_len array
        self.running_seq_len = np.zeros(self.batch_size)
        # Map ind within a batch to request id
        self.running_ind_to_id = np.zeros(self.batch_size)

        while keep_alive:
            keep_alive = self.process_queries()

        # V1 shutdown engine
        self.model.llm_engine.engine_core.shutdown()

        with self.cond_var:
            self.alive_counter.value -= 1
            # The instance needs to get restarted
            if self.alive_counter.value == 0:
                pass
            else:
                self.input_queue.put(None)
            self.cond_var.notify()

    def load_model(self):
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            use_inductor=True,
            # compile_sizes=[64,128,256],
        )
        
        model_kwargs={
            "model": self.model_path,
            #"dtype": 'float16',
            "dtype": "auto",
            "skip_tokenizer_init": False,
            "tensor_parallel_size": self.tp,
            "pipeline_parallel_size": self.pp,
            "max_num_seqs": self.batch_size,
            "max_model_len": MAX_MODEL_LEN,
            # "enforce_eager": True,
            #"max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            #"enable_prefix_caching": False,
            #"distributed_executor_backend": "spawn",
            # "calculate_kv_scales": True,
        }

        #if XPU_COUNT>0:
        #    model_kwargs["kv_cache_dtype"] = "int8"
        #else:
        #    model_kwargs["kv_cache_dtype"] = "fp8_e5m2"
        # model_kwargs["kv_cache_dtype"] = "fp8"
        if ENABLE_TORCH_COMPILE:
            model_kwargs["compilation_config"] = compilation_config
        if self.quantized:
            pass

        log.info(model_kwargs)
        # Model-specific params
        #model_kwargs = model_kwargs | ADDITIONAL_MODEL_KWARGS

        self.model = LLM(**model_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_dataset(self):
        if "llama3_1-8b" in self.workload_name:
            self.data_object = Dataset(model_name=self.model_path,
                                            dataset_path=self.dataset_path,
                                            total_sample_count=self.total_sample_count)
        else:
            self.data_object = Dataset(self.model_path,
                                        dataset_path=self.dataset_path,
                                        total_sample_count=self.total_sample_count,
                                       )
                                        #device="cpu")

    def step_engine(
        self,
        pred_output_tokens,
        query_ids,
        qid,
        pred_first_tokens,
        query_ids_first,
        qid_first,
        ):
        if self.model.llm_engine.has_unfinished_requests():
            if len(self.first_token_id)>0:
                end = time.time()
            step_outputs = self.model.llm_engine.step()
            if len(self.first_token_id)>0:
                step_time = time.time()-end
            for output in step_outputs:
                request_id = int(output.request_id)

                if request_id in self.warmup_req_idx: # Skip requests that are from the warmup 
                    continue

                if len(self.first_token_id) > 0:
                    # Add step time to everything other than the one being prefilled
                    if not PBAR:
                        for i in range(len(self.tprefill_list)-len(self.first_token_id)):
                            self.tprefill_list[i] += step_time
                            self.nprefill_list[i] += len(self.first_token_id)

                if output.finished:
                    token_ids = output.outputs[0].token_ids
                    pred_output_tokens.append(token_ids)
                    query_ids.append(self.query_idx_mapping[request_id])
                    qid.append(self.qid_mapping[request_id])

                    finish_time = time.time()
                    time_elapsed = finish_time - self.start_time_mapping[request_id]
                    ttft = self.first_time_mapping[request_id]
                    wait = self.wait_time_mapping[request_id]
                    # Rest of the tokens
                    tpot = (time_elapsed - ttft)/(len(token_ids)-1)
                    with self.cond_var:
                        if ttft+wait >= self.max_ttft_latency:
                            self.over_ttft_counter.value += 1
                        if tpot >= self.max_tpot_latency:
                            self.over_tpot_counter.value += 1
                        over_ttft_count = self.over_ttft_counter.value
                        over_tpot_count = self.over_tpot_counter.value
                        self.cond_var.notify()

                    if not PBAR:
                        self.tpot_list.append(tpot)
                        tpot_sorted = np.sort(self.tpot_list)
                        tpot_mean = np.mean(self.tpot_list)
                        print(f"[{time.time():.3f}] Query {request_id} finished after {time_elapsed:.3f}s, "
                            f"Rank {self.rank}, "
                            f"TTFT {ttft:.3f}s ({over_ttft_count}), "
                            f"TPOT {tpot:.3f}s ({over_tpot_count}), "
                            f"prefill time {self.tprefill_list[request_id]:.3f}s, "
                            f"prefill count {self.nprefill_list[request_id]}, "
                            f"wait time {wait:.3f}s, "
                            f"bs_c {self.model.llm_engine.get_num_unfinished_requests()}, "
                            f"p99 tpot {tpot_sorted[int(0.99*len(tpot_sorted))]:.3f}s, "
                            f"p50 tpot {tpot_mean:.3f}s, "
                            f"generated {len(token_ids)} tokens")
                
                if request_id in self.first_token_id:
                    self.first_time_mapping[request_id] = time.time()-self.start_time_mapping[request_id]
                    if self.server:
                        token_ids = output.outputs[0].token_ids
                        pred_first_tokens.append(token_ids)
                        query_ids_first.append(self.query_idx_mapping[request_id])
                        qid_first.append(self.qid_mapping[request_id])
                    self.first_token_id.remove(request_id)
                    
    def fetch_queries(self):
        # For chunked prefill
        # samples_to_fill = self.batch_size + self.prefill_batch_size - self.model.llm_engine.get_num_unfinished_requests()
        samples_to_fill = self.batch_size - self.model.llm_engine.get_num_unfinished_requests()

        add_new_qitem = samples_to_fill>=self.prefill_batch_size
        return_value = True
        qitem = -1
        
        if self.finished:
            add_new_qitem = False
            if not self.model.llm_engine.has_unfinished_requests():
                print(f"Rank {self.rank} setting return_value to False")
                return_value = False

        # Receive request if there is an empty decode slot
        qitem_list = []
        added_new_queries = 0

        # Ensure that after insertion, the running batch size doesn't exceed max batch size
        while add_new_qitem and added_new_queries+self.prefill_batch_size<=samples_to_fill:
            try:
                # Continue decoding if nothing in queue
                qitem = self.input_queue.get(False)
            except Exception:
                qitem = -1
            else:
                pass
            
            if qitem is None:
                self.finished = True
                add_new_qitem = False
            elif type(qitem) == int:
                add_new_qitem = False
            else:
                qitem_list.append(qitem)
                added_new_queries += len(qitem[0])
                # In case main thread cannot keep up
                time.sleep(0.01)
        return qitem_list, return_value

    def run_one_step(self, qitem_list):
        pred_output_tokens = []
        query_ids = []
        qid = []
        pred_first_tokens = []
        query_ids_first = []
        qid_first = []
        step_engine_args = (
            pred_output_tokens,
            query_ids,
            qid,
            pred_first_tokens,
            query_ids_first,
            qid_first
        )

        if len(qitem_list)>0:

            for qitem in qitem_list:
                qitem, start_time = qitem
                # Construct / collate batch

                input_ids_tensor = []
                # q = qitem[0] # All qitems are of length 1
                for i,q in enumerate(qitem):
                    sampling_params = self.sampling_params
                    if "llama3_1-8b" in self.workload_name:
                        input_ids_tensor = [self.data_object.input_ids[q.index]]
                    else:
                        input_ids_tensor = self.data_object.input_ids[q.index].tolist()

                    self.first_token_id.append(len(self.query_idx_mapping))
                    self.wait_time_mapping.append(time.time()-start_time[i])
                    self.query_idx_mapping.append(q.index)
                    self.qid_mapping.append(q.id)
                    self.start_time_mapping.append(time.time())
                    self.first_time_mapping.append(0)
                    self.tprefill_list.append(0)
                    self.nprefill_list.append(0)

                    for prompt_id in input_ids_tensor:
                        if isinstance(prompt_id, torch.Tensor):
                            prompt_id = prompt_id.tolist()
                        inputs = TokensPrompt(prompt_token_ids=prompt_id)#.tolist())

                        self.model.llm_engine.add_request(str(self.req_counter),
                            inputs,
                            sampling_params)

                        self.req_counter += 1

                        # Find the ind that this request can be inserted
                        batch_ind = np.argmax(self.running_ind_to_id==0)
                        self.running_seq_len[batch_ind] = self.data_object.input_lens[q.index]+1
                        self.running_ind_to_id[batch_ind] = q.id
                        # print(f"adding {q.id}. Mapped to {batch_ind}")

            print(f"Step prompt, req_counter {self.req_counter}")
            self.step_engine_prompt(*step_engine_args)
        elif self.model.llm_engine.has_unfinished_requests():
            self.step_engine(*step_engine_args)
            self.running_seq_len = self.running_seq_len + 1
        else:
            # Avoid excessive empty steps
            time.sleep(0.05)
        return pred_output_tokens, query_ids, qid, pred_first_tokens, query_ids_first, qid_first

    def process_step_outputs(self, step_outputs):
        (
            pred_output_tokens,
            query_ids,
            qid,
            pred_first_tokens,
            query_ids_first,
            qid_first
        ) = step_outputs
        # Process first tokens
        if len(pred_first_tokens)>0:
            processed_first = []
            first_time = time.time()
            for pred_first_token in pred_first_tokens:
                processed_first.append(np.array(pred_first_token).reshape(-1))
            self.first_queue.put((qid_first, processed_first, first_time))

        # Process finished requests
        if len(pred_output_tokens)>0:
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                            input_seq_lens=0,
                                                            query_id_list=query_ids)
            
            # Zero-out corresponding entries
            for qid_c in qid:
                ind_c = np.argmax(self.running_ind_to_id==qid_c)
                # print(f"zeroing out {qid_c}. Mapped to {ind_c}")
                self.running_ind_to_id[ind_c] = 0
                self.running_seq_len[ind_c] = 0

            self.output_queue.put((qid,processed_output))

            with self.cond_var:
                self.sample_counter.value += len(query_ids)
                print(f"Samples run: {self.sample_counter.value}, rank {self.rank}, current_count {self.current_counter.value}")
                self.current_counter.value -= len(query_ids)
                self.cond_var.notify()

        # Update number of blocks
        ind_valid = self.running_ind_to_id > 0
        # Don't update when idle
        if np.sum(ind_valid)>0:
            with self.cond_var:
                self.blocks_counter.value = int(np.sum((self.running_seq_len[ind_valid]+BLOCK_SIZE-1)//BLOCK_SIZE))
                # print(f"Rank {self.rank} updated blocks_counter {self.blocks_counter.value}")
                self.cond_var.notify()

    def process_queries(self):
        # Attempt to fetch queries from input queue
        qitem_list, return_value = self.fetch_queries()
        # Run inference (prefill/decode) for one step
        step_outputs = self.run_one_step(qitem_list)
        # Process (finished) step outputs
        self.process_step_outputs(step_outputs)
        return return_value

def find_best_bucket(query_lists, prefill_tokens, input_len):
    query_lists_new = query_lists + [0]
    query_lists_new = (np.array(query_lists_new)+input_len)%(prefill_tokens+1)
    # Try to fill the bucket with the minimum number of total tokens
    best_ind = np.argmax(query_lists_new)
    ind_full = query_lists_new>(prefill_tokens-64)
    if ind_full.any():
        best_ind_full = np.argmax(ind_full)
    else:
        best_ind_full = len(query_lists_new)
    best_ind = min(best_ind, best_ind_full)
    return best_ind

class SUT():
    def __init__(self,
                 model_path=None,
                 workload_name="llama2-70b",
                 lg_settings=None,
                 scenario="offline",
                 dtype="bfloat16",
                 device="cpu",
                 batch_size=None,
                 total_sample_count=24576,
                 dataset_path=None,
                 use_cached_outputs=False,  # Set this to True *only for test accuracy runs* in case your prior session was killed partway through
                 workers=1,
                 tp=1,
                 pp=1,
                 quantized=False,
                 warmup=False,
                 partial_output_dir_list=[]):
        if XPU_COUNT>0: # XPU needs spawn method. Can be problematic when using INSTANCE_OVERRIDE
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                print("Start method can only be set once")

        # If ONEAPI_DEVICE_SELECTOR is set outside, use that to limit device selection within instances
        preset_device_selector = os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
        if preset_device_selector=="":
            self.xpu_devices = range(XPU_COUNT)
        else:
            self.xpu_devices = preset_device_selector.split("level_zero:")[1].split(",")
        self.xpu_devices = [str(d) for d in self.xpu_devices]

        self.tp = tp
        self.pp = pp

        self.model_path = model_path or "meta-llama/Llama-3.1-70B-Instruct"
        self.workload_name = workload_name
        self.lg_settings=lg_settings
        self.scenario = scenario
        self.device = device

        if not batch_size:
            if device == "cpu":
                batch_size = 1
            else:
                batch_size = 32  # Reduce to 8 if using 4 GPUs, 16 for 8.
        self.batch_size = batch_size
        self.total_sample_count = total_sample_count
        #self.tp = tp
        self.quantized=quantized
        self.warmup = warmup

        # dtype
        if dtype == 'bfloat16':
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == 'float16':
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        self.dataset_path = dataset_path
        self.qsl = lg.ConstructQSL(self.total_sample_count, self.total_sample_count,
                                   self.LoadSamplesToRam, self.UnloadSamplesFromRam)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue_list = [mp.JoinableQueue() for _ in range(self.num_workers)]
        self.query_queue_int = mp.Queue()
        self.output_queue = mp.Queue()
        self.alive_counter = mp.Value("i", 0)
        self.dead_counter = mp.Value("i", 0)
        self.cond_var = mp.Condition(lock=mp.Lock())

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = mp.Value("i", 0)
        self.current_counter_list = [mp.Value("i", 0) for _ in range(self.num_workers)]
        self.blocks_counter_list = [mp.Value("i", 0) for _ in range(self.num_workers)]
        self.over_ttft_counter = mp.Value("i", 0)
        self.over_tpot_counter = mp.Value("i", 0)

        # TODO: CHANGE THIS
        if BATCHED_PREFILL:
            # Adjust for chunked prefill
            self.prefill_batch_size = 1
        else:
            self.prefill_batch_size = 1

        self.progress = None
        self.tp_sizes = []
        self.load_dataset() # In case SUT may balance load using query lens
        self.core_lists = [[]]*self.num_workers
        self.completed_queries = self.find_completed_queries(partial_output_dir_list)
        self.first_token_queue = None

        # Shared worker kwargs. These will be passed as is to the workers
        self.shared_worker_kwargs={
            "model_path": self.model_path,
            "workload_name": self.workload_name,
            "num_workers": self.num_workers,
            # "lg_settings": self.lg_settings, # issue with spawn
            "dataset_path": self.dataset_path,
            #"device": self.device,
            "batch_size": self.batch_size,
            "total_sample_count": self.total_sample_count,
            "dtype": self.amp_dtype,
            "output_queue": self.output_queue,
            "first_queue": self.first_token_queue,
            "cond_var": self.cond_var,
            "alive_counter": self.alive_counter,
            "dead_counter": self.dead_counter,
            "sample_counter": self.sample_counter,
            "over_ttft_counter": self.over_ttft_counter,
            "over_tpot_counter": self.over_tpot_counter,
            "server": self.scenario.lower()=="server",
            "tp": self.tp,
            "pp": self.pp,
            "quantized": self.quantized,
            "warmup": self.warmup,
            "prefill_batch_size": self.prefill_batch_size,
        }

        # Individual worker kwargs. These will be lists of size self.num_workers. i-th worker will get i-th element of each list
        self.individual_worker_kwargs={
            "input_queue": self.query_queue_list,
            "current_counter": self.current_counter_list,
            "blocks_counter": self.blocks_counter_list,
            "rank": list(range(self.num_workers))
        }

        if XPU_COUNT>0:
            world_size = self.tp * self.pp
            self.individual_worker_kwargs["xpu_devices"]=[",".join(self.xpu_devices[j*world_size:(j+1)*world_size]) for j in range(self.num_workers)]

        # Extend/override SUT method if it's defined
        try:
            self.sut_override = types.MethodType(SUT_OVERRIDE, self)
        except NameError:
            pass
        else:
            self.sut_override()

    def load_dataset(self):
        if self.workload_name=="llama3_1-8b":
            self.data_object = Dataset(model_name=self.model_path,
                                            dataset_path=self.dataset_path,
                                            total_sample_count=self.total_sample_count)
        else:
            self.data_object = Dataset(self.model_path,
                                        dataset_path=self.dataset_path,
                                        total_sample_count=self.total_sample_count,
                                       )
                                        #device="cpu")
    
    def find_completed_queries(self, partial_output_dir_list):
        import json
        completed_queries = set()
        for directory in partial_output_dir_list:
            with open(os.path.join(directory,"mlperf_log_accuracy.json"), "r", encoding="UTF-8") as f:
                lines = f.readlines()
            combined_line = ""
            for i,line in enumerate(lines):
                if i==len(lines)-1:
                    if line[-1]!="]":
                        line += "]"
                combined_line += line
            log.info(f"End of combined_line {combined_line[-10:]}")
            results = json.loads(combined_line)
            for pred in results:
                qsl_idx = pred["qsl_idx"]
                if qsl_idx not in completed_queries:
                    completed_queries.add(qsl_idx)
        log.info(f"Completed_queries {completed_queries}, partial list {partial_output_dir_list}")
        return completed_queries


    def LoadSamplesToRam(self, query_samples):
        pass

    def UnloadSamplesFromRam(self, query_samples):
        pass

    def start(self):
        cur_copies = self.num_workers
        # Create worker threads
        for j in range(self.num_workers):

            # Build kwargs for individual workers
            private_worker_kwargs = {key_i: self.individual_worker_kwargs[key_i][j] for key_i in self.individual_worker_kwargs}
            private_worker_kwargs = private_worker_kwargs | self.shared_worker_kwargs

            worker = Instance(**private_worker_kwargs)
            worker.start()
            self.worker_threads[j] = worker
            # Prevent host OOM
            if (j%cur_copies==cur_copies-1):
                # For some nodes it takes a long time to load the checkpoint so all instance weight load overlap, causing memory stress
                with self.cond_var:
                    self.cond_var.wait_for(lambda: self.alive_counter.value >= self.num_workers//2, 120)

        with self.cond_var:
            print(f"Waiting for alive_counter to be equal to {self.num_workers}")
            self.cond_var.wait_for(lambda: self.alive_counter.value == self.num_workers)

        log.info(f"Starting internal issue query thread")
        self.query_thread = threading.Thread(target=self.issue_queries_int_merged)
        self.query_thread.daemon = True
        self.query_thread.start()

        log.info(f"Starting Loadgen response thread")
        self.response_thread = threading.Thread(target=self.response_loadgen)
        self.response_thread.daemon = True
        self.response_thread.start()

    def stop(self):
        self.output_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        if self.scenario=="server":
            with self.cond_var:
                log.info(f"[{time.time():.3f}] Test finished "
                        f"TTFT {self.over_ttft_counter.value} "
                        f"TPOT {self.over_tpot_counter.value} ")
                try:
                    log.info(f"Max first time {np.max(self.first_time_list):.6f}s")
                    log.info(f"Average first time {np.mean(self.first_time_list):.6f}s")
                except (AttributeError, ValueError):
                    pass

                self.cond_var.notify()
        if PBAR:
            # Ensure the last progress bar is updated
            time.sleep(0.5)
            self.progress.close()

    def update_pbar(self, tok_count, last_count, update_value):
        postfix_str = f"{tok_count/self.progress.format_dict['elapsed']:.1f}toks/s"
        self.progress.set_postfix_str(postfix_str, refresh=False)
        self.progress.update(update_value)
        self.progress.refresh()
        last_count += update_value
        return last_count
        
    def response_loadgen(self):
        keep_alive = True
        processed_count = 0
        last_count = 0
        tok_count = 0
        while keep_alive:
            result = self.output_queue.get()
            if result is None:
                keep_alive = False
            else:
                qid, processed_output = result

                for i in range(len(processed_output)):
                    n_tokens = processed_output[i].shape[0]
                    response_array = array.array("B", processed_output[i].tobytes())
                    bi = response_array.buffer_info()
                    response = [lg.QuerySampleResponse(qid[i], bi[0], bi[1], n_tokens)]
                    lg.QuerySamplesComplete(response)
                    processed_count += 1
                    tok_count += n_tokens
                
                if PBAR:
                    # Limit prints to once every PBAR_FREQ completed samples
                    if processed_count - last_count >= PBAR_FREQ:
                        last_count = self.update_pbar(tok_count, last_count, PBAR_FREQ)

        # Update progress bar at the end
        if PBAR and (processed_count > last_count):
            last_count = self.update_pbar(tok_count, last_count, processed_count-last_count)

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self,**kwargs):
        raise NotImplementedError

    def get_best_rank(self, value_added):
        current_counters = np.array([(self.current_counter_list[i].value+value_added) for i in range(self.num_workers)]) # Instances priority will be ordered by their respective in-flight queries
        target_rank = np.argmin(current_counters)
        # if current_counters[target_rank]>self.batch_size:
        #     return -1
        # Instead of trying to only fill up to batch size, allow one queued up
        if self.query_queue_list[target_rank].qsize()>0:
            return -1
        else:
            return target_rank

    def issue_queries(self, query_samples):
        """ Receives samples from loadgen and adds them to queue. Users may choose to batch here"""
        start_time = time.time()
        if PBAR and (self.progress is None):
            if len(query_samples)>1:
                self.progress = tqdm(total=len(query_samples), smoothing=0.0)
            else:
                self.progress = tqdm(smoothing=0.0)

        query_sample_list = []
        for query_sample in query_samples:
            # Continuous batching
            self.query_queue_int.put((query_sample, start_time))

    def try_dispatch(
        self,
        query_list,
        delete_list,
        bucket,
        input_len_bucket,
        time_start_bucket,
        server=False): # Inactive for now
        target_rank = -1
        wait_to_dispatch = True
        while wait_to_dispatch:
            with self.cond_var:
                target_rank = self.get_best_rank(1) # With prepacking, prefill batch size is always 1
                if target_rank!=-1:
                    self.current_counter_list[target_rank].value += 1
            # Always wait for an instance with an empty slot(for now)
            if target_rank==-1:
                time.sleep(0.01)
            else:
                wait_to_dispatch = False
        if target_rank!=-1:
            print(f"Sending to rank {target_rank}, num_queries {len(query_list)}, before add {self.current_counter_list[target_rank].value}")
            self.query_queue_list[target_rank].put((query_list, time_start_bucket))
            delete_list.append(bucket)
            return True
        return False

    # Load balancer for merged prefill
    def issue_queries_int_merged(self):
        keep_alive = True

        query_list = []
        time_start_list = []

        # TODO: use the actual latency target instead of the hard-coded numbers
        time_left = 9999# if self.scenario=="offline" else SERVER_TIME_LIMIT
        #time_compute = SERVER_COMPUTE_TIME
        time_compute = 0

        while keep_alive:
            new_query = False
            try:
                query = self.query_queue_int.get(timeout=0.05)
            except Exception:
                # No query available within timeout, continue
                pass
            else:
                if query is None:
                    keep_alive = False
                else:
                    query, start_time = query
                    start_time_c = start_time
                    new_query = True
            
            if new_query:
                time_start_list.append(start_time_c)
                query_list.append(query)
            
            if len(query_list)>0:
                time_wait = time.time()-time_start_list[0]
                time_needed = time_wait + time_compute
                # When to dispatch
                # 1. Time limit passed
                # 2. Reached prefill batch size
                # 3. End of inference
                if (time_needed > time_left) or (len(query_list)==self.prefill_batch_size) or (not keep_alive):
                    target_rank = -1
                    # Continue trying to dispatch until there is an empty spot
                    while target_rank==-1:
                        with self.cond_var:
                            target_rank = self.get_best_rank(len(query_list))
                            if target_rank!=-1:
                                self.current_counter_list[target_rank].value += len(query_list)
                        if target_rank==-1:
                            time.sleep(0.01)
                    print(f"Sending to rank {target_rank}, length {len(query_list)}, wait_time {time_wait:.2f}s")
                    self.query_queue_list[target_rank].put((query_list, time_start_list))
                    query_list = []
                    time_start_list = []
        
        for i in range(self.num_workers):
            print("Putting none in", i)
            self.query_queue_list[i].put(None)
    
    def flush_queries(self):
        self.query_queue_int.put(None)
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(self, **kwargs):
        kwargs["scenario"]="server"
        super().__init__(**kwargs)
        self.first_token_queue = mp.Queue()
        self.shared_worker_kwargs["first_queue"]=self.first_token_queue
        self.first_time_list = []

    def start(self):
        super().start()
        # Create first token response thread
        log.info(f"Starting first-token response thread")
        self.ft_response_thread = threading.Thread(target=self.process_first_tokens)
        self.ft_response_thread.daemon = True
        self.ft_response_thread.start()

    def process_first_tokens(self):
        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                log.info("Exiting First token response thread")
                break

            qid, processed_output, send_time = first_token_item
            
            for i in range(len(processed_output)):
                response_data = array.array("B", processed_output[i].tobytes())
                bi = response_data.buffer_info()
                response = [lg.QuerySampleResponse(qid[i], bi[0], bi[1])]
                lg.FirstTokenComplete(response)
            receive_time = time.time()-send_time
            self.first_time_list.append(receive_time)


