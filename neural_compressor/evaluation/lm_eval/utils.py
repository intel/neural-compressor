# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


try:
    import habana_frameworks.torch.hpex  # pylint: disable=E0401

    _hpex_available = True
except:
    _hpex_available = False


class LMEvalParser:
    def __init__(
        self,
        model="hf",
        tasks="lambada_openai",
        model_args="",
        user_model=None,
        tokenizer=None,
        num_fewshot=None,
        batch_size=1,
        max_batch_size=None,
        device=None,
        output_path=None,
        limit=None,
        use_cache=None,
        cache_requests=None,
        check_integrity=False,
        write_out=False,
        log_samples=False,
        show_config=False,
        include_path=None,
        gen_kwargs=None,
        verbosity="INFO",
        wandb_args="",
        predict_only=False,
        seed=[0, 1234, 1234],
        trust_remote_code=False,
        pad_to_buckets=None,  # used by HPU to align input length for performance.
        buckets=[32, 64, 128, 256, 512, 1024, 2048, 4096],  # used by HPU to limit input length range.
    ):
        self.model = model
        self.tasks = tasks
        self.model_args = model_args
        self.user_model = user_model
        self.tokenizer = tokenizer
        self.num_fewshot = num_fewshot
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.device = device
        self.output_path = output_path
        self.limit = limit
        self.use_cache = use_cache
        self.cache_requests = cache_requests
        self.check_integrity = check_integrity
        self.write_out = write_out
        self.log_samples = log_samples
        self.show_config = show_config
        self.include_path = include_path
        self.gen_kwargs = gen_kwargs
        self.verbosity = verbosity
        self.wandb_args = wandb_args
        self.predict_only = predict_only
        self.seed = seed
        self.trust_remote_code = trust_remote_code
        if pad_to_buckets is None:
            if _hpex_available:
                self.pad_to_buckets = True
            else:
                self.pad_to_buckets = False
        else:
            self.pad_to_buckets = pad_to_buckets
        self.buckets = buckets
